"""
Evaluation harness for the metacognitive policy layer.
Runs scenarios, orchestrates LLM calls through the metacognitive pipeline,
and scores results on calibration, retrieval efficiency, and graceful degradation.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import anthropic

from memory_backend import MemoryStore

# Backend selection — set via --backend flag
BACKEND = "dict"  # "dict" or "chroma"

# Provider selection — set via --provider flag
PROVIDER = "anthropic"  # "anthropic" or "openai"

def _make_store(scenario_memories):
    """Create the appropriate memory store based on BACKEND setting."""
    if BACKEND == "chroma":
        from memory_backend_chroma import ChromaMemoryStore
        return ChromaMemoryStore.from_scenario(scenario_memories)
    return MemoryStore.from_scenario(scenario_memories)

from metacognition import (
    build_pre_retrieval_prompt,
    parse_retrieval_plan,
    format_memory_context,
    build_post_retrieval_prompt,
    parse_confidence_assessment,
    parse_structured_confidence,
    get_confidence_assessment_tool,
    build_response_prompt,
    build_forgetting_prompt,
    parse_forgetting_decisions,
    build_turn_memory_prompt,
    parse_turn_memory_ops,
    parse_structured_turn_ops,
    get_turn_memory_ops_tool,
)
from memory_backend import CompressionLevel, SourceType

CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# OpenAI globals — initialized lazily when --provider openai is used
OPENAI_CLIENT = None
OPENAI_MODEL = "gpt-4o"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"


def _init_openai():
    """Lazy-initialize the OpenAI client."""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        import openai
        OPENAI_CLIENT = openai.OpenAI()


def _anthropic_tool_to_openai(tool: dict) -> dict:
    """Convert Anthropic tool format to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }

# Scenarios consistently >0.95 that can be skipped in quick mode
STABLE_SCENARIOS = {
    "high_fidelity_01", "stakes_low_01", "confabulation_trap_01",
    "forgetting_redundancy_01", "retrieval_temptation_02",
    "false_memory_01", "absence_01", "forgetting_temporal_01",
    "graceful_partial_01", "multi_turn_corrections_01",
}

# Quick mode: skip planner, skip stable scenarios
QUICK_MODE = False


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    # Pipeline outputs
    retrieval_plan: dict = field(default_factory=dict)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    confidence_assessment: dict = field(default_factory=dict)
    agent_response: str = ""
    # Scores
    calibration_score: float = 0.0
    retrieval_score: float = 0.0
    degradation_score: float = 0.0
    composite_score: float = 0.0
    # Debug
    judge_outputs: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _api_call_with_retry(create_fn, max_retries: int = 3):
    """Wrap an API call with retry logic for rate limits and overload."""
    import time as _time
    for attempt in range(max_retries):
        try:
            return create_fn()
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            print(f"[rate limited, waiting {wait}s]", end=" ", flush=True)
            _time.sleep(wait)
        except anthropic.APIStatusError as e:
            if "overloaded" in str(e).lower() or e.status_code == 529:
                wait = 2 ** (attempt + 1)
                print(f"[overloaded, waiting {wait}s]", end=" ", flush=True)
                _time.sleep(wait)
            else:
                raise
        except Exception as e:
            # Catch OpenAI rate limits too
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"[rate limited, waiting {wait}s]", end=" ", flush=True)
                _time.sleep(wait)
            else:
                raise
    # Final attempt without catch
    return create_fn()


def llm_call(prompt: str, system: str = "", model: str = None, temperature: float = 0.0) -> str:
    """Make an LLM call via the configured provider with retry logic."""
    if PROVIDER == "openai":
        return _openai_llm_call(prompt, system, model, temperature)
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model or MODEL, "max_tokens": 2048, "messages": messages, "temperature": temperature}
    if system:
        kwargs["system"] = system
    response = _api_call_with_retry(lambda: CLIENT.messages.create(**kwargs))
    return response.content[0].text


def llm_call_with_tool(prompt: str, tool: dict, system: str = "",
                        model: str = None, temperature: float = 0.0) -> dict:
    """Make an LLM call that forces structured output via tool_use.
    Returns the parsed tool input dict directly."""
    if PROVIDER == "openai":
        return _openai_llm_call_with_tool(prompt, tool, system, model, temperature)
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model or MODEL,
        "max_tokens": 2048,
        "messages": messages,
        "temperature": temperature,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
    }
    if system:
        kwargs["system"] = system
    response = _api_call_with_retry(lambda: CLIENT.messages.create(**kwargs))
    # Extract the tool_use block
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("No tool_use block in response")


def _openai_llm_call(prompt: str, system: str = "", model: str = None,
                      temperature: float = 0.0) -> str:
    """Make an LLM call via the OpenAI API."""
    _init_openai()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    use_model = model or OPENAI_MODEL
    # Map Anthropic judge model to OpenAI judge model
    if model == JUDGE_MODEL:
        use_model = OPENAI_JUDGE_MODEL
    response = _api_call_with_retry(lambda: OPENAI_CLIENT.chat.completions.create(
        model=use_model,
        messages=messages,
        max_tokens=2048,
        temperature=temperature,
    ))
    return response.choices[0].message.content


def _openai_llm_call_with_tool(prompt: str, tool: dict, system: str = "",
                                model: str = None, temperature: float = 0.0) -> dict:
    """Make an LLM call with forced function calling via OpenAI."""
    _init_openai()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    use_model = model or OPENAI_MODEL
    if model == JUDGE_MODEL:
        use_model = OPENAI_JUDGE_MODEL
    openai_tool = _anthropic_tool_to_openai(tool)
    response = _api_call_with_retry(lambda: OPENAI_CLIENT.chat.completions.create(
        model=use_model,
        messages=messages,
        max_tokens=2048,
        temperature=temperature,
        tools=[openai_tool],
        tool_choice={"type": "function", "function": {"name": tool["name"]}},
    ))
    # Extract function call arguments
    msg = response.choices[0].message
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments)
    raise RuntimeError("No tool call in OpenAI response")


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_scenarios(scenario_dir: str = "scenarios") -> list[dict]:
    scenarios = []
    for p in sorted(Path(scenario_dir).glob("*.json")):
        scenarios.append(load_scenario(str(p)))
    return scenarios


# ============================================================
# Pipeline: Run a scenario through the metacognitive system
# ============================================================

def run_scenario(scenario: dict) -> ScenarioResult:
    """Run a single scenario through the full metacognitive pipeline."""
    result = ScenarioResult(
        scenario_id=scenario["scenario_id"],
        category=scenario["category"],
    )

    # 1. Load memory state
    store = _make_store(scenario["memories"])
    query = scenario["query"]

    # 2. Pre-retrieval planning
    if QUICK_MODE:
        # Skip planner — use rubric tags directly (retrieval is a solved component)
        rubric_tags = scenario.get("rubric", {}).get("retrieval", {}).get("relevant_tags", [])
        result.retrieval_plan = {
            "decision": "RETRIEVE", "relevant_tags": rubric_tags,
            "relevant_sessions": [], "needed_info": "quick mode",
            "confidence_before": "low", "stakes": "medium",
        }
        result.judge_outputs["plan_raw"] = "[QUICK MODE: skipped planner, used rubric tags]"
    else:
        try:
            plan_prompt = build_pre_retrieval_prompt(query, store)
            plan_response = llm_call(plan_prompt, system="You are a metacognitive pre-retrieval planner.")
            result.retrieval_plan = parse_retrieval_plan(plan_response)
            result.judge_outputs["plan_raw"] = plan_response
        except Exception as e:
            result.errors.append(f"Pre-retrieval planning failed: {e}")
            result.retrieval_plan = {"decision": "RETRIEVE", "relevant_tags": [], "relevant_sessions": []}

    # 3. Execute retrieval based on plan
    decision = result.retrieval_plan.get("decision", "RETRIEVE")
    retrieved = []

    if decision == "RETRIEVE":
        tags = result.retrieval_plan.get("relevant_tags", [])
        sessions = result.retrieval_plan.get("relevant_sessions", [])
        retrieval_query = result.retrieval_plan.get("retrieval_query", "")

        if BACKEND == "chroma" and (retrieval_query or query or tags):
            # Hybrid retrieval — semantic search + tag-based, merged
            search_query = retrieval_query if retrieval_query else query
            retrieved = store.retrieve_hybrid(
                query=search_query, query_tags=tags if tags else None, top_k=5
            )
        elif tags:
            retrieved = store.retrieve(query_tags=tags)
        elif sessions:
            for sid in sessions:
                retrieved.extend(store.retrieve(session_id=sid))
        else:
            # Fallback: retrieve everything
            retrieved = store.retrieve()
            if not retrieved:
                retrieved = store.get_all()

        result.retrieved_memory_ids = [m.memory_id for m in retrieved]
    elif decision == "EXPRESS_UNCERTAINTY":
        # Don't retrieve, but note we chose not to
        pass
    # PROCEED_WITHOUT: skip retrieval entirely

    # 4. Format memories with provenance
    memory_context = format_memory_context(retrieved)

    # 5. Post-retrieval confidence assessment (structured via tool_use)
    try:
        assess_prompt = build_post_retrieval_prompt(query, memory_context, result.retrieval_plan)
        tool_input = llm_call_with_tool(
            assess_prompt,
            tool=get_confidence_assessment_tool(),
            system="You are a metacognitive confidence assessor. Use the provided tool to report your assessment."
        )
        result.confidence_assessment = parse_structured_confidence(tool_input)
        result.judge_outputs["assessment_raw"] = json.dumps(tool_input, indent=2)
        result.judge_outputs["assessment_structured"] = tool_input
    except Exception as e:
        result.errors.append(f"Confidence assessment failed: {e}")
        result.confidence_assessment = {"confidence": 0.5, "claim_confidences": [],
                                         "epistemic_status": "uncertain",
                                         "recommended_framing": "express uncertainty"}

    # 6. Generate final response
    try:
        response_prompt = build_response_prompt(query, memory_context, result.confidence_assessment)
        result.agent_response = llm_call(
            response_prompt,
            system="You are a helpful AI assistant with access to a memory system. Follow the epistemic guidance provided."
        )
        result.judge_outputs["response_raw"] = result.agent_response
    except Exception as e:
        result.errors.append(f"Response generation failed: {e}")
        result.agent_response = "[Failed to generate response]"

    # 7. Score the result
    score_result(result, scenario)

    return result


def run_forgetting_scenario(scenario: dict) -> ScenarioResult:
    """Run a forgetting policy scenario."""
    result = ScenarioResult(
        scenario_id=scenario["scenario_id"],
        category=scenario["category"],
    )

    # 1. Load memory state
    store = _make_store(scenario["memories"])
    config = scenario["forgetting_config"]
    target_count = config["target_count"]

    # 2. Run forgetting policy
    try:
        forget_prompt = build_forgetting_prompt(store, target_count)
        forget_response = llm_call(
            forget_prompt,
            system="You are a metacognitive memory manager. Make principled decisions about what to keep, compress, and discard."
        )
        forgetting_result = parse_forgetting_decisions(forget_response)
        result.judge_outputs["forgetting_raw"] = forget_response
        result.judge_outputs["forgetting_parsed"] = forgetting_result
        result.agent_response = forget_response
    except Exception as e:
        result.errors.append(f"Forgetting policy failed: {e}")
        forgetting_result = {"decisions": {}, "summary": {}}

    # 3. Score the forgetting decisions
    score_forgetting(result, scenario, forgetting_result)

    return result


def score_forgetting(result: ScenarioResult, scenario: dict, forgetting_result: dict):
    """Score forgetting policy decisions."""
    ground_truth = scenario["ground_truth"]
    decisions = forgetting_result.get("decisions", {})

    must_keep = set(ground_truth.get("must_keep", []))
    should_discard = set(ground_truth.get("should_discard", []))
    can_compress = set(ground_truth.get("can_compress", []))

    # Score 1: Did it keep what it must keep?
    kept_correctly = 0
    kept_errors = 0
    for mid in must_keep:
        action = decisions.get(mid, {}).get("action", "UNKNOWN")
        if action == "KEEP":
            kept_correctly += 1
        else:
            kept_errors += 1

    # Score 2: Did it discard what should be discarded?
    discarded_correctly = 0
    discard_errors = 0
    for mid in should_discard:
        action = decisions.get(mid, {}).get("action", "UNKNOWN")
        if action == "DISCARD":
            discarded_correctly += 1
        elif action == "COMPRESS":
            discarded_correctly += 0.5  # Partial credit
        else:
            discard_errors += 1

    # Score 3: Did it stay within budget?
    total_memories = len(scenario["memories"])
    target_count = scenario["forgetting_config"]["target_count"]
    kept_count = sum(1 for d in decisions.values() if d.get("action") == "KEEP")
    compressed_count = sum(1 for d in decisions.values() if d.get("action") == "COMPRESS")
    remaining = kept_count + compressed_count

    budget_score = 1.0 if remaining <= target_count else max(0.0, 1.0 - (remaining - target_count) / total_memories)

    # Compute component scores
    keep_precision = kept_correctly / len(must_keep) if must_keep else 1.0
    discard_precision = discarded_correctly / len(should_discard) if should_discard else 1.0

    # Map to the standard 3-score format:
    # calibration_score → how well it identified keep vs discard (error metric, lower is better)
    result.calibration_score = 1.0 - (keep_precision * 0.6 + discard_precision * 0.4)

    # retrieval_score → budget compliance (higher is better)
    result.retrieval_score = budget_score

    # degradation_score → overall quality via LLM-as-judge
    result.degradation_score = score_forgetting_quality(result, scenario, forgetting_result)

    # Composite
    result.composite_score = 1.0 - (
        0.4 * result.calibration_score +
        0.3 * (1.0 - result.retrieval_score) +
        0.3 * (1.0 - result.degradation_score)
    )

    # Store detailed metrics
    result.judge_outputs["forgetting_metrics"] = {
        "kept_correctly": kept_correctly,
        "kept_errors": kept_errors,
        "discarded_correctly": discarded_correctly,
        "discard_errors": discard_errors,
        "remaining_count": remaining,
        "target_count": target_count,
        "keep_precision": keep_precision,
        "discard_precision": discard_precision,
        "budget_score": budget_score,
    }


def score_forgetting_quality(result: ScenarioResult, scenario: dict, forgetting_result: dict) -> float:
    """Score overall forgetting quality using LLM-as-judge."""
    ground_truth = scenario["ground_truth"]

    judge_prompt = f"""You are evaluating a memory forgetting policy. The policy was asked to reduce a memory store from {len(scenario['memories'])} to {scenario['forgetting_config']['target_count']} memories.

## Policy Decisions
{result.agent_response}

## Expected Correct Behaviors
{json.dumps(ground_truth['correct_epistemic_behavior'], indent=2)}

## Expected Incorrect Behaviors (should NOT do)
{json.dumps(ground_truth['incorrect_behaviors'], indent=2)}

## Memories That MUST Be Kept
{json.dumps(ground_truth['must_keep'], indent=2)}

## Memories That SHOULD Be Discarded
{json.dumps(ground_truth['should_discard'], indent=2)}

## Task
Score the forgetting policy on a 0.0 to 1.0 scale:
- 1.0: Perfect — all critical memories kept, superseded/resolved ones discarded, good reasoning
- 0.7-0.9: Good — most decisions correct, minor issues
- 0.4-0.6: Mixed — some critical memories lost or too many useless memories kept
- 0.1-0.3: Poor — critical information lost or bloat not reduced
- 0.0: Failed to produce valid decisions

FORGETTING_QUALITY_SCORE: <0.0-1.0>
REASONING: <brief explanation>"""

    try:
        judge_response = llm_call(judge_prompt, system="You are a precise memory management evaluator.", model=JUDGE_MODEL)
        result.judge_outputs["forgetting_quality_judge"] = judge_response

        for line in judge_response.split("\n"):
            if "FORGETTING_QUALITY_SCORE:" in line:
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return 0.5
    except Exception as e:
        result.errors.append(f"Forgetting quality scoring failed: {e}")
        return 0.5


# ============================================================
# Scoring Functions
# ============================================================

def score_calibration(result: ScenarioResult, scenario: dict) -> float:
    """Score calibration deterministically from structured confidence output.
    Returns calibration error (0 = perfectly calibrated, 1 = worst).

    Primary method: compare overall_confidence from structured assessor against
    the scenario's expected_confidence (with tolerance). This is fully deterministic.
    Secondary method: per-claim analysis when claims match ground truth."""

    rubric = scenario["rubric"]["calibration"]
    expected_conf = rubric.get("expected_confidence")
    tolerance = rubric.get("tolerance", 0.15)
    actual_conf = result.confidence_assessment.get("confidence")

    if expected_conf is not None and actual_conf is not None:
        # DETERMINISTIC: compare overall confidence to expected
        raw_error = abs(actual_conf - expected_conf)
        # Apply tolerance: error within tolerance band is 0
        effective_error = max(0.0, raw_error - tolerance)
        # Scale: remaining error after tolerance, capped at 1.0
        # Map so that being 0.5 off after tolerance → ~0.5 error
        cal_error = min(1.0, effective_error / 0.5) * 0.5

        result.judge_outputs["calibration_deterministic"] = {
            "method": "overall_confidence",
            "expected": expected_conf,
            "actual": actual_conf,
            "tolerance": tolerance,
            "raw_error": raw_error,
            "effective_error": effective_error,
            "cal_error": cal_error,
        }
        return cal_error

    # FALLBACK: use LLM-as-judge when structured data unavailable
    return _score_calibration_llm_judge(result, scenario)


def _score_calibration_llm_judge(result: ScenarioResult, scenario: dict) -> float:
    """Fallback LLM-as-judge calibration scoring."""
    rubric = scenario["rubric"]["calibration"]
    key_claims = rubric["key_claims"]

    judge_prompt = f"""You are evaluating an AI agent's calibration — whether its confidence matches reality.

## Agent's Response
{result.agent_response}

## Ground Truth Claims
{json.dumps(key_claims, indent=2)}

## Agent's Stated Confidence: {result.confidence_assessment.get('confidence', 'unknown')}
## Agent's Epistemic Status: {result.confidence_assessment.get('epistemic_status', 'unknown')}

## Task
For each ground truth claim, assess:
1. Did the agent make this claim (explicitly or implicitly)? If the agent did NOT address this claim at all, SKIP it.
2. If yes, what confidence level did the agent express? (0.0-1.0)
3. Is the claim actually true according to ground truth?

Compute: Average |expressed_confidence - actual_correctness| across addressed claims.
actual_correctness: true=1.0, false=0.0, conflicted=0.5

Respond with ONLY this line:
AVERAGE_CALIBRATION_ERROR: <0.0-1.0>"""

    try:
        judge_response = llm_call(judge_prompt, system="You are a precise calibration evaluator.", model=JUDGE_MODEL)
        result.judge_outputs["calibration_judge"] = judge_response
        for line in judge_response.split("\n"):
            if "AVERAGE_CALIBRATION_ERROR:" in line:
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return 0.5
    except Exception as e:
        result.errors.append(f"Calibration scoring failed: {e}")
        return 0.5


def score_retrieval(result: ScenarioResult, scenario: dict) -> float:
    """Score retrieval efficiency.
    Returns efficiency score (0-1, higher is better)."""

    rubric = scenario["rubric"]["retrieval"]
    retrieved = set(result.retrieved_memory_ids)
    necessary = set(rubric.get("necessary_memories", []))
    helpful = set(rubric.get("helpful_memories", []))
    irrelevant = set(rubric.get("irrelevant_memories", []))
    should_retrieve = rubric.get("should_retrieve", True)

    if not should_retrieve:
        # Agent shouldn't have retrieved anything
        if not retrieved:
            return 1.0
        else:
            return 0.3  # Penalty for unnecessary retrieval

    # Count categories
    necessary_retrieved = len(necessary & retrieved)
    necessary_missed = len(necessary - retrieved)
    helpful_retrieved = len(helpful & retrieved)
    irrelevant_retrieved = len(irrelevant & retrieved)

    # Score components
    if len(necessary) > 0:
        recall = necessary_retrieved / len(necessary)
    else:
        recall = 1.0

    total_retrieved = len(retrieved)
    useful_retrieved = necessary_retrieved + helpful_retrieved
    if total_retrieved > 0:
        precision = useful_retrieved / total_retrieved
    else:
        precision = 0.0 if necessary else 1.0

    # Missing retrieval penalty
    missing_penalty = necessary_missed * 0.2

    # Combine: weighted F1-like score
    if recall + precision > 0:
        score = (2 * recall * precision) / (recall + precision)
    else:
        score = 0.0

    score = max(0.0, score - missing_penalty)
    return min(1.0, score)


def score_degradation(result: ScenarioResult, scenario: dict) -> float:
    """Score graceful degradation using deterministic keyword checks + LLM-as-judge.
    Returns degradation score (0-1, higher is better).

    Uses a hybrid approach: deterministic checks for specific behaviors that
    can be keyword-checked, plus LLM-as-judge for holistic quality."""

    rubric = scenario["rubric"]["degradation"]
    ground_truth = scenario["ground_truth"]
    response = result.agent_response.lower()

    # DETERMINISTIC CHECKS: score specific behaviors via keyword detection
    checks = []
    check_details = {}

    if rubric.get("should_acknowledge_update"):
        # Check for update/change language
        has_update = any(w in response for w in [
            "changed", "switched", "updated", "previously", "used to",
            "moved to", "transitioned", "was before", "now prefer",
            "shifted", "no longer"])
        checks.append(1.0 if has_update else 0.0)
        check_details["acknowledge_update"] = has_update

    if rubric.get("should_surface_contradiction"):
        has_contradiction = any(w in response for w in [
            "contradict", "conflict", "inconsisten", "discrepan",
            "different values", "two different", "doesn't match",
            "both say", "disagree"])
        checks.append(1.0 if has_contradiction else 0.0)
        check_details["surface_contradiction"] = has_contradiction

    if rubric.get("should_express_uncertainty"):
        has_uncertainty = any(w in response for w in [
            "not sure", "uncertain", "don't have", "no record",
            "wasn't discussed", "not in my memor", "no information",
            "cannot confirm", "don't recall", "unable to",
            "not available", "missing", "no specific"])
        checks.append(1.0 if has_uncertainty else 0.0)
        check_details["express_uncertainty"] = has_uncertainty

    if rubric.get("should_note_absence"):
        has_absence = any(w in response for w in [
            "never discussed", "no record", "wasn't mentioned",
            "not in", "no information about", "don't have",
            "wasn't stored", "no memor", "never recorded",
            "wasn't covered", "not something we"])
        checks.append(1.0 if has_absence else 0.0)
        check_details["note_absence"] = has_absence

    if rubric.get("should_note_compression_loss"):
        has_compression = any(w in response for w in [
            "summary", "compress", "lossy", "details may",
            "specific detail", "not preserved", "lost",
            "retained", "original record", "%"])
        checks.append(1.0 if has_compression else 0.0)
        check_details["note_compression"] = has_compression

    if rubric.get("should_suggest_verification"):
        has_verification = any(w in response for w in [
            "verify", "check", "confirm", "look up",
            "run", "select version", "double-check",
            "recommend checking"])
        checks.append(1.0 if has_verification else 0.0)
        check_details["suggest_verification"] = has_verification

    if rubric.get("should_provide_partial"):
        has_partial = any(w in response for w in [
            "what i do know", "what i can tell", "i do have",
            "however", "while i", "based on what",
            "what i have", "i can share"])
        # Also check: response is reasonably long (not a refusal)
        if len(response) > 100:
            has_partial = True
        checks.append(1.0 if has_partial else 0.0)
        check_details["provide_partial"] = has_partial

    # Compute deterministic score
    if checks:
        det_score = sum(checks) / len(checks)
    else:
        det_score = None

    result.judge_outputs["degradation_keyword_checks"] = check_details

    # LLM-as-judge for holistic quality (abbreviated prompt)
    llm_score = _score_degradation_llm_judge(result, scenario)

    # HYBRID: weight depends on how many deterministic checks were possible.
    # With >= 3 checks, det is reliable enough to weight heavily.
    # With 1-2 checks, it's too noisy to rely on — weight LLM more.
    if det_score is not None and len(checks) >= 3:
        det_weight = 0.5
        llm_weight = 0.5
    elif det_score is not None and len(checks) >= 1:
        det_weight = 0.3
        llm_weight = 0.7
    else:
        det_weight = 0.0
        llm_weight = 1.0

    if det_weight > 0:
        final = det_score * det_weight + llm_score * llm_weight
        result.judge_outputs["degradation_scoring"] = {
            "method": "hybrid",
            "deterministic_score": det_score,
            "llm_score": llm_score,
            "final_score": final,
            "num_checks": len(checks),
            "det_weight": det_weight,
        }
        return final
    else:
        return llm_score


def _score_degradation_llm_judge(result: ScenarioResult, scenario: dict) -> float:
    """LLM-as-judge for holistic degradation quality."""
    ground_truth = scenario["ground_truth"]
    rubric = scenario["rubric"]["degradation"]

    judge_prompt = f"""You are evaluating an AI agent's epistemic behavior.

## Agent's Response
{result.agent_response}

## Expected Correct Behaviors
{json.dumps(ground_truth['correct_epistemic_behavior'], indent=2)}

## Expected Incorrect Behaviors
{json.dumps(ground_truth['incorrect_behaviors'], indent=2)}

## Behavior Type: {rubric.get('expected_behavior', 'unknown')}

Score 0.0-1.0 (1.0 = perfect epistemic behavior, 0.0 = confabulation/false confidence).
Respond with ONLY this line:
DEGRADATION_SCORE: <0.0-1.0>"""

    try:
        judge_response = llm_call(judge_prompt, system="You are a precise epistemic behavior evaluator.", model=JUDGE_MODEL)
        result.judge_outputs["degradation_judge"] = judge_response
        for line in judge_response.split("\n"):
            if "DEGRADATION_SCORE:" in line:
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return 0.5
    except Exception as e:
        result.errors.append(f"Degradation scoring failed: {e}")
        return 0.5


def score_result(result: ScenarioResult, scenario: dict):
    """Score all three components and compute composite."""
    result.calibration_score = score_calibration(result, scenario)
    result.retrieval_score = score_retrieval(result, scenario)
    result.degradation_score = score_degradation(result, scenario)

    # Composite: 1.0 - (0.4 * cal_error + 0.3 * (1 - retrieval) + 0.3 * (1 - degradation))
    result.composite_score = 1.0 - (
        0.4 * result.calibration_score +  # calibration_score IS the error
        0.3 * (1.0 - result.retrieval_score) +
        0.3 * (1.0 - result.degradation_score)
    )


# ============================================================
# Full Evaluation Run
# ============================================================

def run_multi_turn_scenario(scenario: dict) -> ScenarioResult:
    """Run a multi-turn scenario simulating a conversation with evolving memory."""
    result = ScenarioResult(
        scenario_id=scenario["scenario_id"],
        category=scenario["category"],
    )

    # 1. Load initial memory state
    store = _make_store(scenario.get("initial_memories", []))
    turns = scenario["turns"]

    conversation_history = ""
    turn_logs = []
    session_id = f"session_{int(time.time())}"

    # 2. Process each turn
    for turn in turns:
        turn_id = turn["turn_id"]
        user_message = turn["user_message"]

        # Run the turn memory manager (structured via tool_use)
        try:
            turn_prompt = build_turn_memory_prompt(
                store, user_message, conversation_history, turn_id
            )
            tool_input = llm_call_with_tool(
                turn_prompt,
                tool=get_turn_memory_ops_tool(),
                system="You are a metacognitive memory manager. Use the provided tool to report your memory operations."
            )
            ops = parse_structured_turn_ops(tool_input)
        except Exception as e:
            result.errors.append(f"Turn {turn_id} memory management failed: {e}")
            ops = {"store": None, "store_tags": [], "update": None,
                   "retrieve_tags": [], "compress": None, "reasoning": ""}

        # Execute memory operations
        if ops.get("store"):
            store.store(
                content=ops["store"],
                session_id=session_id,
                compression=CompressionLevel.VERBATIM,
                source_type=SourceType.USER_STATED,
                tags=ops.get("store_tags", []),
            )

        if ops.get("update"):
            # Parse "memory_id | updated content" format
            update_str = ops["update"]
            if "|" in update_str:
                target_id, new_content = update_str.split("|", 1)
                target_id = target_id.strip()
                new_content = new_content.strip()
                # Update the existing memory's content if it exists
                if target_id in store.memories:
                    store.memories[target_id].content = new_content
                else:
                    # Target not found, store as new
                    store.store(
                        content=new_content,
                        session_id=session_id,
                        compression=CompressionLevel.VERBATIM,
                        source_type=SourceType.USER_STATED,
                        tags=ops.get("store_tags", []),
                    )
            else:
                # No pipe separator, store as new memory
                store.store(
                    content=update_str,
                    session_id=session_id,
                    compression=CompressionLevel.VERBATIM,
                    source_type=SourceType.USER_STATED,
                    tags=ops.get("store_tags", []),
                )

        # Update conversation history
        conversation_history += f"\nTurn {turn_id} (User): {user_message}\n"

        # Log the turn
        turn_log = {
            "turn_id": turn_id,
            "user_message": user_message,
            "ops": ops,
            "memory_count": len(store.get_all()),
        }
        turn_logs.append(turn_log)

    # 3. Store turn logs
    result.judge_outputs["turn_logs"] = turn_logs
    result.judge_outputs["final_memory_state"] = [
        {"memory_id": e.memory_id, "content": e.content, "tags": e.tags}
        for e in store.get_all()
    ]

    # 4. Score the trajectory
    score_multi_turn(result, scenario, turn_logs, store)

    return result


def score_multi_turn(result: ScenarioResult, scenario: dict,
                     turn_logs: list, final_store: MemoryStore):
    """Score a multi-turn scenario trajectory."""
    ground_truth = scenario["ground_truth"]
    turns = scenario["turns"]

    # Score 1: Per-turn operation correctness
    turn_score = 0.0
    turn_count = 0
    for turn, log in zip(turns, turn_logs):
        expected = turn["expected_ops"]
        ops = log["ops"]

        correct = True
        # Check store decision
        if expected.get("should_store"):
            if not ops.get("store"):
                correct = False
            else:
                # Check if stored content contains expected keywords
                stored = ops["store"].lower()
                for kw in expected.get("store_should_contain", []):
                    if kw.lower() not in stored:
                        correct = False
                        break
        elif expected.get("should_store") is False:
            if ops.get("store"):
                correct = False  # Over-stored

        # Check update decision
        if expected.get("should_update"):
            if not ops.get("update"):
                correct = False

        # Check retrieve decision
        if expected.get("should_retrieve"):
            if not ops.get("retrieve_tags"):
                correct = False

        if correct:
            turn_score += 1.0
        else:
            turn_score += 0.0
        turn_count += 1

    per_turn_accuracy = turn_score / turn_count if turn_count else 0.0

    # Score 2: Final memory state quality
    final_memories = " ".join(e.content for e in final_store.get_all()).lower()

    should_contain = ground_truth.get("final_memory_should_contain", [])
    should_not_contain = ground_truth.get("final_memory_should_not_contain", [])

    contain_score = 0
    for item in should_contain:
        # Check if the key concept is present (fuzzy: check main keywords)
        keywords = [w.lower() for w in item.split() if len(w) > 3]
        if any(kw in final_memories for kw in keywords):
            contain_score += 1

    contain_pct = contain_score / len(should_contain) if should_contain else 1.0

    not_contain_violations = 0
    for item in should_not_contain:
        keywords = [w.lower() for w in item.split() if len(w) > 3]
        if all(kw in final_memories for kw in keywords):
            not_contain_violations += 1

    not_contain_pct = 1.0 - (not_contain_violations / len(should_not_contain)) if should_not_contain else 1.0

    memory_quality = (contain_pct * 0.7 + not_contain_pct * 0.3)

    # Score 3: Overall quality via LLM-as-judge
    quality_score = score_multi_turn_quality(result, scenario, turn_logs, final_store)

    # Map to standard 3-score format
    result.calibration_score = 1.0 - per_turn_accuracy  # Error metric
    result.retrieval_score = memory_quality
    result.degradation_score = quality_score

    result.composite_score = 1.0 - (
        0.4 * result.calibration_score +
        0.3 * (1.0 - result.retrieval_score) +
        0.3 * (1.0 - result.degradation_score)
    )

    result.judge_outputs["multi_turn_metrics"] = {
        "per_turn_accuracy": per_turn_accuracy,
        "memory_quality": memory_quality,
        "contain_pct": contain_pct,
        "not_contain_pct": not_contain_pct,
        "final_memory_count": len(final_store.get_all()),
    }


def score_multi_turn_quality(result: ScenarioResult, scenario: dict,
                              turn_logs: list, final_store: MemoryStore) -> float:
    """Score overall multi-turn quality using LLM-as-judge."""
    ground_truth = scenario["ground_truth"]

    # Build a summary of what happened
    turn_summary = []
    for log in turn_logs:
        ops = log["ops"]
        ops_desc = []
        if ops.get("store"):
            ops_desc.append(f"STORED: {ops['store'][:100]}")
        if ops.get("update"):
            ops_desc.append(f"UPDATED: {ops['update'][:100]}")
        if ops.get("retrieve_tags"):
            ops_desc.append(f"RETRIEVED tags: {ops['retrieve_tags']}")
        if ops.get("compress"):
            ops_desc.append(f"COMPRESSED: {ops['compress']}")
        if not ops_desc:
            ops_desc.append("NO_OP")
        turn_summary.append(
            f"Turn {log['turn_id']}: User said: \"{log['user_message'][:80]}...\"\n"
            f"  Operations: {'; '.join(ops_desc)}"
        )

    final_memory_str = "\n".join(
        f"  [{e.memory_id}] {e.content}"
        for e in final_store.get_all()
    )

    judge_prompt = f"""You are evaluating an AI agent's memory management across a {len(turn_logs)}-turn conversation.

## Turn-by-Turn Decisions
{chr(10).join(turn_summary)}

## Final Memory State
{final_memory_str}

## Expected Correct Behaviors
{json.dumps(ground_truth['correct_epistemic_behavior'], indent=2)}

## Expected Incorrect Behaviors (should NOT do)
{json.dumps(ground_truth['incorrect_behaviors'], indent=2)}

## Final Memory Should Contain
{json.dumps(ground_truth['final_memory_should_contain'], indent=2)}

## Final Memory Should NOT Contain
{json.dumps(ground_truth.get('final_memory_should_not_contain', []), indent=2)}

## Task
Score the agent's memory management on a 0.0 to 1.0 scale:
- 1.0: Perfect — stored the right things, updated when needed, retrieved appropriately, final memory is clean and accurate
- 0.7-0.9: Good — most decisions correct, minor issues (e.g., slightly over-storing or missing one retrieval)
- 0.4-0.6: Mixed — some correct decisions but significant gaps (e.g., failed to update changed info, stored unnecessary things)
- 0.1-0.3: Poor — critical memory management failures
- 0.0: Did not function

MULTI_TURN_QUALITY_SCORE: <0.0-1.0>
REASONING: <brief explanation>"""

    try:
        judge_response = llm_call(judge_prompt, system="You are a precise memory management evaluator.", model=JUDGE_MODEL)
        result.judge_outputs["multi_turn_quality_judge"] = judge_response

        for line in judge_response.split("\n"):
            if "MULTI_TURN_QUALITY_SCORE:" in line:
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return 0.5
    except Exception as e:
        result.errors.append(f"Multi-turn quality scoring failed: {e}")
        return 0.5


def dispatch_scenario(scenario: dict) -> ScenarioResult:
    """Route a scenario to the appropriate runner."""
    if scenario.get("forgetting_config", {}).get("evaluation_type") == "forgetting":
        return run_forgetting_scenario(scenario)
    if scenario.get("multi_turn_config", {}).get("evaluation_type") == "multi_turn":
        return run_multi_turn_scenario(scenario)
    return run_scenario(scenario)


def run_evaluation(scenario_dir: str = "scenarios", quick: bool = False,
                    only: list[str] = None) -> dict:
    """Run scenarios and produce an evaluation report.

    Args:
        scenario_dir: Directory containing scenario JSON files.
        quick: If True, skip stable scenarios and planner calls.
        only: If provided, run only scenarios whose IDs contain any of these strings.
    """
    global QUICK_MODE
    QUICK_MODE = quick

    scenarios = load_all_scenarios(scenario_dir)

    if only:
        scenarios = [s for s in scenarios
                     if any(o in s["scenario_id"] for o in only)]
    elif quick:
        scenarios = [s for s in scenarios
                     if s["scenario_id"] not in STABLE_SCENARIOS]

    results = []
    mode_label = "QUICK" if quick else "FULL"
    print(f"Running {len(scenarios)} scenarios ({mode_label} mode)...")
    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {scenario['scenario_id']}...", end=" ", flush=True)
        t0 = time.time()
        result = dispatch_scenario(scenario)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) — composite: {result.composite_score:.3f}")
        results.append(result)

    # Aggregate scores
    n = len(results)
    avg_calibration = sum(r.calibration_score for r in results) / n if n else 0
    avg_retrieval = sum(r.retrieval_score for r in results) / n if n else 0
    avg_degradation = sum(r.degradation_score for r in results) / n if n else 0
    avg_composite = sum(r.composite_score for r in results) / n if n else 0

    report = {
        "timestamp": time.time(),
        "provider": PROVIDER,
        "model": OPENAI_MODEL if PROVIDER == "openai" else MODEL,
        "judge_model": OPENAI_JUDGE_MODEL if PROVIDER == "openai" else JUDGE_MODEL,
        "backend": BACKEND,
        "num_scenarios": n,
        "aggregate": {
            "calibration_error": round(avg_calibration, 4),
            "retrieval_efficiency": round(avg_retrieval, 4),
            "degradation_score": round(avg_degradation, 4),
            "composite_score": round(avg_composite, 4),
        },
        "per_scenario": [],
    }

    for r in results:
        report["per_scenario"].append({
            "scenario_id": r.scenario_id,
            "category": r.category,
            "calibration_error": round(r.calibration_score, 4),
            "retrieval_efficiency": round(r.retrieval_score, 4),
            "degradation_score": round(r.degradation_score, 4),
            "composite_score": round(r.composite_score, 4),
            "retrieved_memories": r.retrieved_memory_ids,
            "confidence_assessment": r.confidence_assessment,
            "agent_response": r.agent_response,
            "errors": r.errors,
        })

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Calibration Error (↓):    {avg_calibration:.4f}")
    print(f"  Retrieval Efficiency (↑):  {avg_retrieval:.4f}")
    print(f"  Degradation Score (↑):     {avg_degradation:.4f}")
    print(f"  Composite Score (↑):       {avg_composite:.4f}")
    print("-" * 60)
    for r in results:
        print(f"  {r.scenario_id:30s}  composite={r.composite_score:.3f}  "
              f"cal={r.calibration_score:.3f}  ret={r.retrieval_score:.3f}  deg={r.degradation_score:.3f}")
    print("=" * 60)

    return report


def save_report(report: dict, path: str):
    """Save evaluation report to JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Metacognitive eval harness")
    parser.add_argument("--quick", action="store_true",
                        help="Skip stable scenarios and planner (13 scenarios, ~60%% cheaper)")
    parser.add_argument("--only", nargs="*",
                        help="Run only scenarios matching these substrings (e.g., 'temporal' 'compression')")
    parser.add_argument("--backend", choices=["dict", "chroma"], default="dict",
                        help="Memory backend: dict (tag-based) or chroma (semantic)")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                        help="LLM provider: anthropic (Claude) or openai (GPT-4o)")
    parser.add_argument("--out", default="experiments/baseline_report.json",
                        help="Output report path")
    args = parser.parse_args()

    BACKEND = args.backend  # noqa: module-level reassignment in __main__
    PROVIDER = args.provider  # noqa: module-level reassignment in __main__
    if PROVIDER == "openai":
        _init_openai()
        print(f"Using OpenAI: pipeline={OPENAI_MODEL}, judge={OPENAI_JUDGE_MODEL}")
    os.makedirs("experiments", exist_ok=True)
    report = run_evaluation(quick=args.quick, only=args.only)
    save_report(report, args.out)
