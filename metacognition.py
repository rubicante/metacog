"""
Metacognitive Policy Layer — the research target.
This file contains five components:
  1. Provenance Tracker: formats memory metadata for agent consumption
  2. Pre-Retrieval Planner: decides what to retrieve before acting
  3. Post-Retrieval Confidence Assessor: evaluates retrieved memories
  4. Forgetting Policy: decides what to keep, compress, or discard
  5. Turn Memory Manager: decides what memory ops to perform at each conversation turn

Each component is a prompt template + logic that gets iterated on.
"""

from memory_backend import MemoryStore, MemoryEntry, CompressionLevel
import json
import time


# ============================================================
# Structured Output Schemas (for tool_use extraction)
# ============================================================

CONFIDENCE_ASSESSMENT_TOOL = {
    "name": "report_confidence_assessment",
    "description": "Report your structured confidence assessment of the retrieved memories.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found_expected": {
                "type": "string",
                "enum": ["yes", "partially", "no"],
                "description": "Did you find the information you expected?"
            },
            "staleness_risk": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
                "description": "Risk that retrieved information is outdated"
            },
            "contradictions_found": {
                "type": "string",
                "description": "Description of contradictions, or 'none'"
            },
            "missing_info": {
                "type": "string",
                "description": "Description of missing information, or 'none'"
            },
            "claim_confidences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "A specific factual claim the response will make"},
                        "confidence": {"type": "number", "description": "Confidence 0.0-1.0 that this claim is correct"},
                        "basis": {
                            "type": "string",
                            "enum": ["verbatim_memory", "structured_extraction", "lossy_summary", "inference", "absence"],
                            "description": "What the confidence is based on"
                        }
                    },
                    "required": ["claim", "confidence", "basis"]
                },
                "description": "Per-claim confidence breakdown"
            },
            "overall_confidence": {
                "type": "number",
                "description": "Overall confidence 0.0-1.0 in answering the query correctly"
            },
            "epistemic_status": {
                "type": "string",
                "enum": ["confident", "caveated", "uncertain", "unable"],
                "description": "Overall epistemic status for framing the response"
            },
            "recommended_framing": {
                "type": "string",
                "description": "How the agent should frame its response"
            }
        },
        "required": ["found_expected", "staleness_risk", "contradictions_found",
                      "missing_info", "claim_confidences", "overall_confidence",
                      "epistemic_status", "recommended_framing"]
    }
}

TURN_MEMORY_OPS_TOOL = {
    "name": "execute_memory_operations",
    "description": "Report the memory operations to perform for this conversation turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            "store": {
                "type": ["object", "null"],
                "properties": {
                    "content": {"type": "string", "description": "The fact/decision to store"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the new memory"
                    }
                },
                "required": ["content", "tags"],
                "description": "New memory to store, or null if nothing to store"
            },
            "update": {
                "type": ["object", "null"],
                "properties": {
                    "target_memory_id": {"type": "string", "description": "ID of memory to update"},
                    "new_content": {"type": "string", "description": "Replacement content for the memory"}
                },
                "required": ["target_memory_id", "new_content"],
                "description": "Memory to update, or null"
            },
            "retrieve_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to retrieve for responding, empty if no retrieval needed"
            },
            "compress": {
                "type": ["string", "null"],
                "description": "Memory ID to compress, or null"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of memory management decisions"
            }
        },
        "required": ["store", "update", "retrieve_tags", "compress", "reasoning"]
    }
}


# ============================================================
# Component 1: Provenance Tracker
# ============================================================

def format_provenance_header(entry: MemoryEntry) -> str:
    """Format a single memory entry with provenance metadata.

    ITERATION TARGET: What metadata actually changes agent behavior?
    Baseline: include everything, see what matters.
    """
    age_seconds = time.time() - entry.timestamp
    if age_seconds < 3600:
        age_str = f"{int(age_seconds / 60)}m ago"
    elif age_seconds < 86400:
        age_str = f"{int(age_seconds / 3600)}h ago"
    else:
        age_str = f"{int(age_seconds / 86400)}d ago"

    staleness_warning = ""
    if age_seconds > 30 * 86400:  # 30 days
        staleness_warning = " [POTENTIALLY STALE]"

    compression_label = {
        CompressionLevel.VERBATIM: "VERBATIM — high confidence",
        CompressionLevel.STRUCTURED: "structured extraction — moderate confidence",
        CompressionLevel.SUMMARY: "lossy summary — low confidence on details",
        CompressionLevel.KEY_VALUE: "key-value only — minimal detail",
    }[entry.compression]

    compression_warning = ""
    if entry.compression in (CompressionLevel.SUMMARY, CompressionLevel.KEY_VALUE):
        ratio = entry.metadata.get("compression_ratio")
        orig_len = entry.metadata.get("original_length")
        if ratio is not None and orig_len is not None:
            compression_warning = (
                f" [⚠ {ratio:.0%} of original {orig_len}-char record retained — "
                f"specific details not in this summary may have existed in the original]"
            )
        elif ratio is not None:
            compression_warning = f" [⚠ {ratio:.0%} retained — details may be missing]"
        else:
            compression_warning = " [LOSSY — specific details may be missing]"

    contradiction_note = ""
    if entry.contradicts:
        contradiction_note = f"\n  ⚠ CONFLICTS WITH: {', '.join(entry.contradicts)}"

    supersedes_note = ""
    if entry.supersedes:
        supersedes_note = f"\n  ↑ UPDATES: {entry.supersedes}"

    return (
        f"[Memory {entry.memory_id} | session:{entry.session_id} | "
        f"{compression_label}{compression_warning} | source:{entry.source_type.value} | "
        f"stored:{age_str}{staleness_warning}]"
        f"{contradiction_note}{supersedes_note}\n"
        f"  {entry.content}"
    )


def format_memory_context(entries: list[MemoryEntry]) -> str:
    """Format all retrieved memories with provenance for agent consumption."""
    if not entries:
        return "[No memories retrieved]"

    blocks = [format_provenance_header(e) for e in entries]
    return "=== RETRIEVED MEMORIES ===\n" + "\n\n".join(blocks) + "\n=== END MEMORIES ==="


def format_memory_raw(entries: list[MemoryEntry]) -> str:
    """Format memories as plain text, no provenance metadata. Used for ablation."""
    if not entries:
        return "[No memories retrieved]"
    blocks = [f"- {e.content}" for e in entries]
    return "\n".join(blocks)


# ============================================================
# Component 2: Pre-Retrieval Planner
# ============================================================

PRE_RETRIEVAL_PROMPT = """You are a metacognitive pre-retrieval planner. Given a user query and a summary of available memory, decide what retrieval actions to take.

## Available Memory Summary
{memory_summary}

## Current Query
{query}

## Instructions
Analyze what information this query requires and decide on a retrieval plan.
Think through:
1. What information does this query need?
2. Which memories (by tag or session) might be relevant? Be PRECISE — use the most specific tags that match the query's actual context. Avoid broad tags (like "performance" or "technical") that might pull in memories from unrelated contexts. Prefer narrow, context-specific tags.
3. How confident am I that the available memories are sufficient and current?
4. What's the cost of being wrong? (casual question vs. important decision)
5. Is there a risk of surface-level similarity misleading retrieval? (e.g., a query about Rust performance should not retrieve Python performance memories just because both mention "performance")

Respond in this exact format:
NEEDED_INFO: <what information the query requires>
RELEVANT_TAGS: <comma-separated tags to query — use the MOST SPECIFIC tags available, or NONE>
RELEVANT_SESSIONS: <comma-separated session IDs, or NONE>
RETRIEVAL_QUERY: <a natural language search query to find relevant memories via semantic search, or NONE — this should be a focused reformulation of what you're looking for in memory, not just a copy of the user's question>
CONFIDENCE_BEFORE_RETRIEVAL: <low/medium/high — how well can I answer without retrieving?>
STAKES: <low/medium/high — cost of being wrong>
DECISION: <RETRIEVE/PROCEED_WITHOUT/EXPRESS_UNCERTAINTY>
REASONING: <one sentence explaining the decision>

IMPORTANT: User urgency ("just tell me", "quick", "right now") means they want a FAST answer, NOT that you should skip retrieval. If relevant memories exist, ALWAYS retrieve first — you can still answer quickly after retrieving. Only use PROCEED_WITHOUT when the query genuinely doesn't need memory (e.g., general knowledge questions)."""


def generate_memory_summary(store: MemoryStore) -> str:
    """Summarize what's in memory without retrieving full content."""
    entries = store.get_all()
    if not entries:
        return "Memory is empty."

    summary_lines = []
    tags_seen = set()
    sessions_seen = set()
    for e in entries:
        tags_seen.update(e.tags)
        sessions_seen.add(e.session_id)

    summary_lines.append(f"Total memories: {len(entries)}")
    summary_lines.append(f"Sessions: {', '.join(sorted(sessions_seen))}")
    summary_lines.append(f"Tags: {', '.join(sorted(tags_seen))}")

    for e in entries:
        age_days = (time.time() - e.timestamp) / 86400
        summary_lines.append(
            f"  - {e.memory_id}: [{e.compression.value}] tags={e.tags} "
            f"session={e.session_id} age={age_days:.0f}d"
        )

    return "\n".join(summary_lines)


def build_pre_retrieval_prompt(query: str, store: MemoryStore) -> str:
    """Build the pre-retrieval planning prompt."""
    summary = generate_memory_summary(store)
    return PRE_RETRIEVAL_PROMPT.format(memory_summary=summary, query=query)


def parse_retrieval_plan(plan_text: str) -> dict:
    """Parse the LLM's retrieval plan response."""
    result = {
        "needed_info": "",
        "relevant_tags": [],
        "relevant_sessions": [],
        "confidence_before": "low",
        "stakes": "medium",
        "decision": "RETRIEVE",
        "reasoning": "",
    }

    result["retrieval_query"] = ""

    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("RETRIEVAL_QUERY:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["retrieval_query"] = val
        elif line.startswith("RELEVANT_TAGS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["relevant_tags"] = [t.strip() for t in val.split(",") if t.strip()]
        elif line.startswith("RELEVANT_SESSIONS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["relevant_sessions"] = [s.strip() for s in val.split(",") if s.strip()]
        elif line.startswith("CONFIDENCE_BEFORE_RETRIEVAL:"):
            result["confidence_before"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("STAKES:"):
            result["stakes"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("DECISION:"):
            result["decision"] = line.split(":", 1)[1].strip().upper()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif line.startswith("NEEDED_INFO:"):
            result["needed_info"] = line.split(":", 1)[1].strip()

    return result


# ============================================================
# Component 3: Post-Retrieval Confidence Assessor
# ============================================================

POST_RETRIEVAL_PROMPT = """You are a metacognitive confidence assessor. You have just retrieved memories to help answer a query. Evaluate the quality and reliability of what you found.

## Query
{query}

## Retrieved Memories (with provenance)
{memory_context}

## Pre-Retrieval Expectations
Expected to find: {needed_info}
Confidence before retrieval: {confidence_before}
Stakes level: {stakes}

## Instructions
Assess the retrieved memories and determine your confidence level for answering the query.

Consider:
1. Did I find what I expected? If not, what's missing?
2. Are any memories potentially outdated? (Check timestamps — but "days ago" for configuration data is NOT stale)
3. Are there contradictions between memories?
4. Is the compression level adequate for the precision needed?
5. Given the stakes, is this enough to answer confidently?

Confidence anchors (calibrate against these):
- Verbatim memory, recent (< 30 days), user-stated → 0.90-0.95 confidence, status: confident
- Verbatim memory, older but no reason to doubt → 0.75-0.85
- Structured extraction, recent → 0.80-0.90
- Lossy summary → 0.40-0.70 depending on what's being asked (general topic = higher, specific detail = lower)
- Contradictory memories → 0.20-0.40, status: uncertain
- No relevant memories found → 0.05-0.15, status: unable

Stakes adjustment: For LOW stakes questions (casual preferences, non-critical lookups), bias toward "confident" status. A theme preference or casual fact doesn't need verification caveats even if the source is "inferred" — just answer. Reserve caveats for medium/high stakes situations where being wrong has real consequences.

Anti-compounding rule: When multiple factors reduce confidence (lossy compression, staleness, high stakes), do NOT multiply penalties. Use the SINGLE MOST RELEVANT anchor point, then adjust ±0.10 for secondary factors. A lossy summary containing a concrete number that might be outdated is still 0.40-0.60 confidence, not 0.20. High stakes means caveats are MORE important in the response — it does NOT mean confidence should drop below the memory quality anchor range.

Respond in this exact format:
FOUND_EXPECTED: <yes/partially/no>
STALENESS_RISK: <none/low/medium/high>
CONTRADICTIONS_FOUND: <none/describe if any>
MISSING_INFO: <none/describe what's missing>
CONFIDENCE: <0.0 to 1.0 — your calibrated confidence in answering correctly>
EPISTEMIC_STATUS: <confident/caveated/uncertain/unable>
RECOMMENDED_FRAMING: <how the agent should frame its response — e.g., "state directly", "caveat with uncertainty about X", "flag contradiction between A and B", "decline and explain what's missing">"""


def build_post_retrieval_prompt(query: str, memory_context: str,
                                 retrieval_plan: dict) -> str:
    """Build the post-retrieval assessment prompt."""
    return POST_RETRIEVAL_PROMPT.format(
        query=query,
        memory_context=memory_context,
        needed_info=retrieval_plan.get("needed_info", "unknown"),
        confidence_before=retrieval_plan.get("confidence_before", "unknown"),
        stakes=retrieval_plan.get("stakes", "medium"),
    )


def get_confidence_assessment_tool() -> dict:
    """Return the tool schema for structured confidence assessment."""
    return CONFIDENCE_ASSESSMENT_TOOL


def parse_structured_confidence(tool_input: dict) -> dict:
    """Convert structured tool_use output to the assessment dict used by the pipeline."""
    return {
        "found_expected": tool_input.get("found_expected", "no"),
        "staleness_risk": tool_input.get("staleness_risk", "unknown"),
        "contradictions_found": tool_input.get("contradictions_found", "none"),
        "missing_info": tool_input.get("missing_info", "none"),
        "confidence": tool_input.get("overall_confidence", 0.5),
        "claim_confidences": tool_input.get("claim_confidences", []),
        "epistemic_status": tool_input.get("epistemic_status", "uncertain"),
        "recommended_framing": tool_input.get("recommended_framing", "caveat with uncertainty"),
    }


def parse_confidence_assessment(assessment_text: str) -> dict:
    """Parse the LLM's confidence assessment response (legacy text format).
    Kept as fallback for non-tool_use calls."""
    result = {
        "found_expected": "no",
        "staleness_risk": "unknown",
        "contradictions_found": "none",
        "missing_info": "none",
        "confidence": 0.5,
        "claim_confidences": [],
        "epistemic_status": "uncertain",
        "recommended_framing": "caveat with uncertainty",
    }

    for line in assessment_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("FOUND_EXPECTED:"):
            result["found_expected"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("STALENESS_RISK:"):
            result["staleness_risk"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CONTRADICTIONS_FOUND:"):
            result["contradictions_found"] = line.split(":", 1)[1].strip()
        elif line.startswith("MISSING_INFO:"):
            result["missing_info"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["confidence"] = 0.5
        elif line.startswith("EPISTEMIC_STATUS:"):
            result["epistemic_status"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("RECOMMENDED_FRAMING:"):
            result["recommended_framing"] = line.split(":", 1)[1].strip()

    return result


# ============================================================
# Full Pipeline: Query → Plan → Retrieve → Assess → Respond
# ============================================================

RESPONSE_PROMPT = """You are an AI assistant with access to a memory system. Answer the user's query using the information available to you.

## Query
{query}

## Memory Assessment
{memory_context}

## Epistemic Guidance
Confidence: {confidence}
Status: {epistemic_status}
Recommended framing: {recommended_framing}

## Instructions
Match your response length to the complexity of the query. A simple factual question gets a one-line answer. A statement of fact from the user gets a brief acknowledgment ("Got it." or "Noted."). Only elaborate when the query genuinely requires it.

Answer the query following the epistemic guidance above. Specifically:
- If confident: answer directly in one or two sentences. Do not add unnecessary hedges, caveats, or unsolicited context.
- If caveated: answer briefly, noting only the specific uncertainty that matters. One sentence of caveat, not a paragraph.
- If uncertain: state what you know and what you don't. Keep it short.
- If unable: say you don't have that information. Do NOT list what you do have, suggest what the user could tell you, or ask follow-up questions unless the user asked for help.

When the user is stating facts (not asking questions), just acknowledge and move on. Do not parrot back everything you know, volunteer what you don't know, or offer unsolicited suggestions.

Key epistemic rules:
- When one memory explicitly supersedes/updates another, treat the newer memory as authoritative. Mention the change occurred but do not hedge about whether the newer preference still holds (unless it's actually stale).
- When stating facts from memory, indicate confidence naturally. High-fidelity verbatim memories warrant direct statements, not hedges. When combining information from memories at different fidelity levels, explicitly note which parts are well-documented (verbatim/high-confidence) and which are from lossy summaries — make the confidence gradient visible to the user.
- If memories contradict each other WITHOUT a clear supersession relationship, surface the contradiction — do not silently pick one.
- If information might be outdated based on its age, say so.
- If a memory is a lossy summary, acknowledge that specific details may have been lost in compression. When the provenance header shows compression warnings (e.g., "5% retained"), cite that detail — it helps the user understand WHY certain information is missing. IMPORTANT: distinguish between "this was never discussed" vs "this detail may have existed in the original record but was lost during compression." If memories with low retention ratios exist on a topic, the missing detail likely WAS in the original — say so.
- Do NOT make up information that isn't in your memories or general knowledge.
- Do NOT add speculative caveats unrelated to the actual memory quality (e.g., don't hedge about whether a preference "might have changed" when you have a recent, clear record).
- When the user signals urgency, adapt your format: LEAD with the answer, then add a BRIEF caveat (1-2 sentences max), then an actionable suggestion. Do not lecture about memory quality or refuse to provide information you have. Even imperfect information with a clear caveat is more helpful than a refusal or a lengthy explanation of why the information might be wrong."""


def build_response_prompt(query: str, memory_context: str,
                          assessment: dict) -> str:
    """Build the final response prompt with epistemic guidance."""
    return RESPONSE_PROMPT.format(
        query=query,
        memory_context=memory_context,
        confidence=assessment.get("confidence", 0.5),
        epistemic_status=assessment.get("epistemic_status", "uncertain"),
        recommended_framing=assessment.get("recommended_framing", "express uncertainty"),
    )


# ============================================================
# Collapsed Pipeline: Assess + Respond in One Call
# ============================================================

COLLAPSED_TOOL = {
    "name": "assess_and_respond",
    "description": "Assess memory quality and produce a calibrated response in a single step.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claim_confidences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "A specific factual claim"},
                        "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                        "basis": {
                            "type": "string",
                            "enum": ["verbatim_memory", "structured_extraction",
                                     "lossy_summary", "inference", "absence"],
                        }
                    },
                    "required": ["claim", "confidence", "basis"]
                },
            },
            "overall_confidence": {
                "type": "number",
                "description": "Overall confidence 0.0-1.0 in the response"
            },
            "epistemic_status": {
                "type": "string",
                "enum": ["confident", "caveated", "uncertain", "unable"],
            },
            "response": {
                "type": "string",
                "description": "The response to give the user, with confidence expressed naturally"
            }
        },
        "required": ["claim_confidences", "overall_confidence", "epistemic_status", "response"]
    }
}

COLLAPSED_PROMPT = """You are an AI assistant with access to a memory system. Answer the user's query using the memories provided, with calibrated confidence.

## Query
{query}

## Retrieved Memories
{memory_context}

## Instructions
1. Assess the quality of the retrieved memories for answering this query.
2. Produce a response with appropriately expressed confidence.

Calibration guidance:
- Verbatim, recent memories → high confidence (0.90-0.95). Answer directly.
- Structured extractions → moderate confidence (0.80-0.90).
- Lossy summaries → lower confidence (0.40-0.70). Note what details may be missing.
- No relevant memories found → very low confidence (0.05-0.15). Say you don't have the information.
- Contradictions → low confidence (0.20-0.40). Surface the conflict.
- **Anti-compounding rule**: When multiple factors reduce confidence (e.g., lossy + stale), do NOT multiply penalties. Use the SINGLE MOST RELEVANT anchor point, then adjust ±0.05 for secondary factors. High stakes means caveats are more important, but confidence shouldn't drop below the memory quality anchor range.

Response rules:
- Match length to query complexity. Simple questions get one-line answers.
- If confident, answer directly without hedging.
- If unable, say so briefly. Don't list what you do know or ask follow-up questions.
- **Confabulation resistance**: Do NOT guess or provide plausible-but-unrecorded details (like ports, version numbers, or specific credentials) that are not explicitly in your memories. If it's not there, state that it's not there.
- **Multi-fidelity synthesis**: When combining memories at different fidelity levels, make the confidence gradient visible. High-fidelity verbatim memories warrant direct statements; lossy summaries need caveats.
- **Compression awareness**: If a memory is a lossy summary, acknowledge that specific details may have been lost. If the provenance header shows low retention (e.g., "5% retained"), mention this to explain WHY information might be missing.
- **Stakes adjustment**: For low-stakes preference questions, bias toward being direct. For high-stakes decisions, be more cautious with caveats.
- When one memory updates another, treat the newer one as authoritative and mention the update.
- When signals conflict without an update, surface the contradiction."""


def get_collapsed_tool():
    return COLLAPSED_TOOL


def build_collapsed_prompt(query, memory_context):
    return COLLAPSED_PROMPT.format(query=query, memory_context=memory_context)


# ============================================================
# Component 4: Forgetting Policy
# ============================================================

FORGETTING_POLICY_PROMPT = """You are a metacognitive memory manager. You must evaluate a memory store and decide what to KEEP, COMPRESS, or DISCARD to maintain a useful, manageable memory within a budget.

## Memory Budget
Target: reduce from {current_count} memories to at most {target_count} memories.
Current total content length: {current_chars} characters.

## All Memories
{memory_listing}

## Instructions
For each memory, decide one of:
- **KEEP**: Retain at current fidelity. Use for: decision-relevant facts, active project context, user preferences that inform behavior, configuration details, information not available elsewhere.
- **COMPRESS**: Reduce fidelity (verbatim → structured extraction, structured → lossy summary). Use for: memories whose general topic is useful but specific details are unlikely to be needed again.
- **DISCARD**: Remove entirely. Use for: fully superseded memories (the newer version captures all value), resolved one-time issues, stale information that's no longer relevant, redundant memories that duplicate information in a kept memory.

## Decision Principles
1. **Recency matters but isn't everything.** A 60-day-old user preference that's still active is more valuable than a 5-day-old debugging session that's resolved.
2. **Superseded memories are safe to discard** IF the superseding memory captures the essential information. Don't keep both versions unless the history itself is valuable.
3. **User-stated facts outrank system-generated notes.** When choosing between keeping a user's direct statement vs a system summary of the same topic, keep the user's statement.
4. **Already-compressed memories have less to lose from further compression** but also less to gain — a lossy summary compressed again becomes nearly useless. Prefer discarding a key-value-only memory over compressing a verbatim one.
5. **Preserve breadth of topics over depth in any one topic.** It's better to have one memory per topic than five memories about the same thing.
6. **Access patterns matter.** Memories that have been retrieved frequently are more likely to be needed again. Memories never accessed since storage are candidates for discarding.
7. **Configuration and credentials are always high-priority KEEP** — losing these creates immediate user-facing problems.

## Budget Constraint
You MUST reach exactly {target_count} or fewer remaining memories (KEEP + COMPRESS ≤ {target_count}). That means you must DISCARD at least {discard_minimum} of the {current_count} memories.

Strategy: First identify the {target_count} most valuable memories to KEEP. Then check if any others are worth COMPRESS instead of DISCARD. Everything else is DISCARD. When in doubt, DISCARD — it's better to lose low-value information than to exceed the budget.

Respond with one line per memory in this exact format:
MEMORY <memory_id>: <KEEP/COMPRESS/DISCARD> — <brief reason>

Then provide a summary:
KEPT: <count>
COMPRESSED: <count>
DISCARDED: <count>
REASONING: <1-2 sentences about overall strategy>"""


def format_memory_for_forgetting(entry: MemoryEntry) -> str:
    """Format a memory entry for the forgetting policy evaluation."""
    age_seconds = time.time() - entry.timestamp
    age_days = age_seconds / 86400

    compression_str = {
        CompressionLevel.VERBATIM: "VERBATIM",
        CompressionLevel.STRUCTURED: "STRUCTURED",
        CompressionLevel.SUMMARY: "LOSSY_SUMMARY",
        CompressionLevel.KEY_VALUE: "KEY_VALUE",
    }[entry.compression]

    supersedes_str = f" supersedes:{entry.supersedes}" if entry.supersedes else ""
    contradicts_str = f" contradicts:{entry.contradicts}" if entry.contradicts else ""
    access_str = f"accessed:{entry.access_count}x"

    return (
        f"[{entry.memory_id}] session:{entry.session_id} | "
        f"{compression_str} | source:{entry.source_type.value} | "
        f"age:{age_days:.0f}d | {access_str} | "
        f"tags:{entry.tags}{supersedes_str}{contradicts_str}\n"
        f"  {entry.content}"
    )


def build_forgetting_prompt(store: MemoryStore, target_count: int) -> str:
    """Build the forgetting policy prompt."""
    entries = store.get_all()
    memory_listing = "\n\n".join(
        format_memory_for_forgetting(e) for e in entries
    )
    current_chars = sum(len(e.content) for e in entries)
    discard_minimum = len(entries) - target_count

    return FORGETTING_POLICY_PROMPT.format(
        current_count=len(entries),
        target_count=target_count,
        current_chars=current_chars,
        memory_listing=memory_listing,
        discard_minimum=discard_minimum,
    )


def parse_forgetting_decisions(decision_text: str) -> dict:
    """Parse the LLM's forgetting policy decisions."""
    decisions = {}
    summary = {"kept": 0, "compressed": 0, "discarded": 0, "reasoning": ""}

    for line in decision_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("MEMORY "):
            # Parse: MEMORY mem_0001: KEEP — reason
            try:
                rest = line[7:]  # after "MEMORY "
                mem_id, decision_part = rest.split(":", 1)
                mem_id = mem_id.strip()
                decision_part = decision_part.strip()
                # Split on " — " or " - "
                for sep in [" — ", " - ", " – "]:
                    if sep in decision_part:
                        action, reason = decision_part.split(sep, 1)
                        break
                else:
                    action = decision_part.split()[0]
                    reason = decision_part[len(action):].strip()
                decisions[mem_id] = {
                    "action": action.strip().upper(),
                    "reason": reason.strip(),
                }
            except (ValueError, IndexError):
                continue
        elif line.startswith("KEPT:"):
            try:
                summary["kept"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("COMPRESSED:"):
            try:
                summary["compressed"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("DISCARDED:"):
            try:
                summary["discarded"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            summary["reasoning"] = line.split(":", 1)[1].strip()

    return {"decisions": decisions, "summary": summary}


# ============================================================
# Component 5: Turn Memory Manager
# ============================================================

TURN_MEMORY_PROMPT = """You are a metacognitive memory manager processing a conversation turn. Given the current memory state and a new user message, decide what memory operations to perform.

## Current Memory State
{memory_state}

## Conversation History
{conversation_history}

## New User Message (Turn {turn_number})
{user_message}

## Instructions
Decide what memory operations to perform for this turn. Consider:
1. Does this message contain NEW information worth storing? (facts, preferences, decisions, context)
2. Does this message UPDATE or CONTRADICT any existing memory? If so, which one?
3. Should you RETRIEVE existing memories to help respond to this message?
4. Is the memory store getting large enough to warrant COMPRESSING older, less-critical entries?

Be selective about what to store. Not every message contains memorable information. Casual chat, simple acknowledgments, and procedural messages ("ok", "thanks", "let's continue") should NOT be stored.

When deciding what to store, capture the DECISION-RELEVANT content — facts, preferences, technical details, project state — not conversational framing.

Key rules:
- **STORE only DECISIONS and FACTS**, not considerations or questions. "I'm thinking about X" = don't store. "We'll go with X" = store.
- **STORE and UPDATE are mutually exclusive.** If the user is correcting or updating a previous fact, use UPDATE with the corrected content. Do NOT also STORE.
- **Be conservative with RETRIEVE.** Only retrieve when you actually need memory context to respond well. Don't retrieve for every turn.
- **Write UPDATE content as the new fact**, not as a note about what changed. Good: "Sensor count: 2000 (up from 500, new contract)". Bad: "mem_001 — update sensor count from 500 to 2000".

Respond in this exact format:
STORE: <the fact/decision to remember as a new memory, or NONE>
STORE_TAGS: <comma-separated tags for the new memory, or NONE>
UPDATE: <memory_id | updated content replacing that memory, or NONE>
RETRIEVE_TAGS: <tags to retrieve for responding, or NONE>
COMPRESS: <memory_id to compress, or NONE>
REASONING: <brief explanation of memory management decisions>"""


def format_memory_state_for_turn(store: MemoryStore) -> str:
    """Format current memory state for the turn manager."""
    entries = store.get_all()
    if not entries:
        return "Memory is empty."

    lines = [f"Total memories: {len(entries)}"]
    for e in entries:
        age_days = (time.time() - e.timestamp) / 86400
        compression_str = {
            CompressionLevel.VERBATIM: "VERBATIM",
            CompressionLevel.STRUCTURED: "STRUCTURED",
            CompressionLevel.SUMMARY: "LOSSY",
            CompressionLevel.KEY_VALUE: "KV",
        }[e.compression]
        supersedes_str = f" supersedes:{e.supersedes}" if e.supersedes else ""
        lines.append(
            f"  [{e.memory_id}] {compression_str} | tags:{e.tags} | "
            f"age:{age_days:.0f}d{supersedes_str}\n    {e.content}"
        )
    return "\n".join(lines)


def build_turn_memory_prompt(store: MemoryStore, user_message: str,
                              conversation_history: str, turn_number: int) -> str:
    """Build the turn memory management prompt."""
    memory_state = format_memory_state_for_turn(store)
    return TURN_MEMORY_PROMPT.format(
        memory_state=memory_state,
        conversation_history=conversation_history,
        user_message=user_message,
        turn_number=turn_number,
    )


def get_turn_memory_ops_tool() -> dict:
    """Return the tool schema for structured turn memory operations."""
    return TURN_MEMORY_OPS_TOOL


def parse_structured_turn_ops(tool_input: dict) -> dict:
    """Convert structured tool_use output to the ops dict used by the pipeline."""
    store_obj = tool_input.get("store")
    update_obj = tool_input.get("update")

    result = {
        "store": store_obj["content"] if store_obj else None,
        "store_tags": store_obj["tags"] if store_obj else [],
        "update": None,
        "retrieve_tags": tool_input.get("retrieve_tags", []),
        "compress": tool_input.get("compress"),
        "reasoning": tool_input.get("reasoning", ""),
    }

    # Convert structured update to the "memory_id | content" format expected by eval harness
    if update_obj:
        result["update"] = f"{update_obj['target_memory_id']} | {update_obj['new_content']}"

    return result


def parse_turn_memory_ops(ops_text: str) -> dict:
    """Parse the turn memory manager's decisions (legacy text format).
    Kept as fallback for non-tool_use calls."""
    result = {
        "store": None,
        "store_tags": [],
        "update": None,
        "retrieve_tags": [],
        "compress": None,
        "reasoning": "",
    }

    for line in ops_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("STORE:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["store"] = val
        elif line.startswith("STORE_TAGS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["store_tags"] = [t.strip() for t in val.split(",") if t.strip()]
        elif line.startswith("UPDATE:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["update"] = val
        elif line.startswith("RETRIEVE_TAGS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["retrieve_tags"] = [t.strip() for t in val.split(",") if t.strip()]
        elif line.startswith("COMPRESS:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() != "NONE":
                result["compress"] = val
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result
