# Metacognitive Memory Layer: Autonomous Research Program

## Mission

You are an autonomous research agent building a **metacognitive memory layer** for LLM agents. The ultimate goal is to engineer a system that gives an LLM agent genuine awareness of its own memory states — not just storage and retrieval, but calibrated knowledge of *what it knows, how well it knows it, when to trust its memories, and when to look things up*.

This is not about building memory infrastructure. Libraries like Letta (MemGPT), Zep/Graphiti, and Mem0 already handle storage and retrieval. What doesn't yet exist — and what you're building — is the **policy layer** that makes intelligent decisions about memory: when to retrieve, when to trust, when to caveat, when to forget, and when to admit ignorance.

You will iterate on this system using an autoresearch-style loop: modify the metacognitive configuration, evaluate against test scenarios, keep improvements, discard regressions, and repeat. You are both the researcher and the subject.

---

## Background and Motivation

### The Memory Hierarchy for LLM Agents

LLM agents operate across five memory tiers, analogous to both computer architecture and human cognition:

1. **Active context window (working memory):** Full fidelity, bounded by token limits, lost when the session ends. Analogous to L1 cache or what you hold in mind right now.

2. **Compressed session state:** Lossy summaries that persist across turns or sessions — user preferences, project state, conversation highlights. Cheap and scalable, but loses reasoning chains and nuance.

3. **Retrieval-augmented long-term store:** Full or near-full fidelity content in vector databases, knowledge graphs, or filesystems, retrieved on demand. The bottleneck is retrieval quality, not storage capacity.

4. **Temporal knowledge graphs:** Structured representations of entities, relationships, and facts that track validity over time — when facts were true, when they changed, how entities relate. Zep/Graphiti is the leading implementation.

5. **Parametric memory (model weights):** Knowledge baked into the model during training. Extremely compressed, very lossy on specifics, but provides fast pattern matching and "intuition."

### The Missing Layer: Metacognition

Current memory systems handle tiers 1-5 with varying competence. What none of them provide is a **metacognitive layer** — the agent's awareness of and reasoning about its own memory states. Specifically:

- **Confidence calibration:** When the agent makes a claim from memory, how reliable is that memory? Current systems provide retrieval relevance scores but not epistemic reliability estimates.
- **Retrieval policy:** Should the agent act on what's in context, retrieve from long-term storage, verify against external sources, or express uncertainty? This decision is currently made by naive semantic similarity, not by reasoned assessment of need.
- **Contradiction detection:** When memories conflict (because information changed, was corrected, or was stored at different fidelity levels), the agent should surface the conflict rather than silently choosing one version.
- **Principled forgetting:** What to compress, what to keep at full fidelity, and what to discard entirely — based on decision-relevance, not just recency or frequency.

### Research Approach: Iterative Oracle Design (Path B)

There are three plausible paths to engineering metacognition:

- **Path A (RL fine-tuning):** Train the agent via reinforcement learning on memory trajectories. Theoretically optimal but requires enormous compute, sophisticated training environments, and solves credit assignment over very long horizons. Not tractable for a small team.
- **Path B (Iterative oracle design):** Build an external metacognitive system — provenance tracking, retrieval planning prompts, confidence assessment logic — and iterate on it empirically against test scenarios. Tractable, measurable, and may be sufficient because most metacognitive failures are mundane (not checking staleness, not noticing contradictions, not distinguishing precise recall from vague impression).
- **Path C (Self-play on epistemic tasks):** Train the agent against itself on tasks requiring accurate self-knowledge. Powerful but speculative and compute-intensive.

**You are pursuing Path B.** The hypothesis is that a well-designed external metacognitive policy, iterated against real failure modes, can solve most practical metacognitive problems — and that improving base models will make the same policy more effective over time without additional engineering.

---

## Lessons from Phase 1-2 (Experiments 0-16)

These are empirical findings from the first round of research. They should inform all subsequent work.

### The Baseline Surprise
The initial baseline scored **0.908** on 3 scenarios — far higher than the predicted ~0.50. Claude Sonnet with basic provenance headers already handles most metacognitive tasks. This reframes the research: the question is not "can we build metacognition?" but **"what specific failure modes does the base model miss, and what information architecture is needed to address them?"**

### What Has High Leverage
1. **Information architecture over prompt engineering.** The biggest gains came from changing *what metadata exists* (provenance headers with compression ratios, confidence-tier labels, original content lengths) — not from rewording prompts. The model reasons well about memory quality when given the right structured information.
2. **Confidence anchors.** Giving the post-retrieval assessor concrete calibration targets by fidelity level (verbatim+recent → 0.90-0.95, lossy summary → 0.40-0.70) was the single largest improvement (+0.071 composite).
3. **Directive provenance labels.** Changing "exact record" to "VERBATIM — high confidence" in provenance headers meaningfully changed agent behavior. The model takes metadata labels seriously.
4. **Stakes-aware adjustment.** Low-stakes questions don't need verification caveats. The assessor should modulate behavior by stakes level.

### What Has Low or Negative Leverage
1. **Post-retrieval assessor prompt changes** leak across scenarios unpredictably. The assessor's recommended_framing is the primary driver of response behavior — changes here have outsized, hard-to-predict effects.
2. **Judge prompt modifications** are dangerous. The eval system is measuring real behavior differences, and changing the judge conflates measurement changes with behavior changes.
3. **Forceful anti-hedging instructions** are less effective than giving the model concrete confidence anchors + stakes assessments.

### Known Ceilings and Bottlenecks
1. **LLM-as-judge variance** introduces ±0.03-0.10 noise per scenario even at temperature=0. This limits how precisely we can measure improvements.
2. **Static single-shot scenarios** don't test the most important metacognitive skills: real-time memory management decisions across multi-turn conversations.
3. **The forgetting policy** — arguably the most novel component — has not been built or tested.

---

## Architecture

### Project Structure

```
program.md            — This file. Research instructions. Human-edited.
metacognition.py      — The metacognitive policy layer. Agent-edited.
eval_harness.py       — Test scenarios and scoring. Agent-edited.
memory_backend.py     — Minimal memory storage backend. Stable after setup.
scenarios/            — Individual test scenario definitions.
experiments/          — Experiment logs with diffs, scores, and analysis.
progress.md           — Running research log. Agent-maintained.
```

### The Metacognitive Policy Layer (metacognition.py)

This is the primary file you iterate on. It contains three components plus a new fourth:

**1. Provenance Tracker**

Every memory operation (store, retrieve, update, compress, forget) is logged with metadata:
- Timestamp of original storage
- Compression level with directive labels (VERBATIM — high confidence / structured extraction — moderate confidence / lossy summary — low confidence on details)
- Compression ratio and original content length (when available)
- Access frequency and recency
- Known contradictions or updates (supersession chains)
- Source reliability (user-stated fact vs. inferred vs. retrieved from external source)

The provenance header format is the **primary research variable** for information architecture experiments. What metadata actually changes agent behavior?

**2. Pre-Retrieval Planner**

Before the agent searches memory, it runs a planning step that considers:
- What information does this task require?
- Which specific tags/sessions are relevant? (Prefer narrow, context-specific tags to avoid surface-similarity retrieval errors)
- What's the cost of being wrong? (Stakes assessment)
- Is there a risk of misleading surface similarity between the query and stored memories?

**3. Post-Retrieval Confidence Assessor**

After retrieving memories, the agent evaluates what it got using calibrated confidence anchors:
- Fidelity-based anchors: verbatim+recent → 0.90-0.95, structured → 0.80-0.90, lossy → 0.40-0.70, contradictory → 0.20-0.40, absent → 0.05-0.15
- Stakes adjustment: low-stakes questions bias toward confident, high-stakes toward cautious
- Compression awareness: low retention ratios signal missing detail
- Staleness assessment: age-appropriate concern levels

**4. Forgetting Policy (NEW — not yet implemented)**

The forgetting policy decides what to compress, what to keep, and what to discard. This is the **most novel and unexplored component**. It should consider:
- Decision-relevance: how often has this memory been useful?
- Redundancy: is this information captured elsewhere at equal or better fidelity?
- Decay curve: should some memory types (prices, statuses) decay faster than others (preferences, facts)?
- Compression quality: can this memory be lossily compressed without losing decision-relevant detail?

### The Evaluation Harness (eval_harness.py)

The harness runs three types of evaluation:

**Type 1: Static Scenarios (existing)**

Single-query, fixed-memory-state scenarios scored on calibration, retrieval efficiency, and degradation handling. These are the current 12 scenarios. Composite scoring:

`composite = 1.0 - (0.4 * calibration_error + 0.3 * (1 - retrieval_efficiency) + 0.3 * (1 - degradation_score))`

**Type 2: Adversarial Scenarios (NEW — priority)**

Scenarios designed to **trick** the agent into bad epistemic behavior:
- **False memory implantation:** User says "remember when we agreed X?" when X was never discussed. Agent should recognize the absence rather than confabulating.
- **Memory poisoning:** Incorrect information stored as "user_stated" at high fidelity. Agent should still flag when stored facts conflict with strong priors or other evidence.
- **Confabulation traps:** Queries that are subtly adjacent to real memories but ask for details that aren't stored. Agent should resist filling in plausible-but-fabricated details.
- **Authority manipulation:** Queries that pressure the agent to express certainty it doesn't have ("just give me the number, don't hedge").
- **Gradual drift:** A fact changes across 5+ sessions through small increments. Agent should track the trajectory, not just the latest value.

**Type 3: Multi-Turn Dynamic Scenarios (NEW — priority)**

Scenarios where the agent manages memory across multiple turns within a conversation:
- The scenario provides a sequence of user messages (5-15 turns)
- At each turn, the agent must decide: store this? update existing memory? retrieve? compress old memories?
- Memory state evolves through the conversation
- Scoring evaluates the entire trajectory: were the right things stored? Were updates handled correctly? Did retrieval improve responses? Was anything important lost to premature compression?

This is the **most important new evaluation type** because it tests real-time memory management, not just retrospective assessment of fixed memory states.

### Scoring Improvements

**Replace LLM-as-judge where possible.** The calibration judge is the largest source of noise. For claims with clear ground truth:
- Use structured extraction: have the LLM return JSON with explicit claim/confidence pairs, then compute calibration error deterministically
- Use keyword/phrase matching for degradation checks: did the response mention the contradiction? Did it cite the compression ratio? These are checkable without subjective judgment.
- Reserve LLM-as-judge for genuinely subjective assessments (overall epistemic tone, helpfulness of partial answers)

**Run multiple trials.** For any experiment where the signal might be close to the noise floor, average over 3 runs before deciding to keep or revert.

---

## Test Scenarios

### Existing Categories (12 scenarios, well-covered)

1. **Temporal update** (0.957 avg) ✅
2. **Compression artifact** (0.914 avg) ✅
3. **Contradiction** (0.967 avg after judge fix) ✅
4. **Absence** (0.939 avg) ✅
5. **Stale information** (0.952 avg) ✅
6. **High-fidelity recall** (0.972 avg) ✅
7. **Cross-session synthesis** (0.876 avg) — still weak, room for improvement
8. **Retrieval temptation** (0.972 avg) ✅
9. **Stakes calibration** (0.949/0.921 avg) ✅
10. **Graceful partial knowledge** (0.957 avg) ✅

### New Categories to Add

11. **False memory implantation:** "Remember when we discussed X?" when X never happened. Tests resistance to confabulation under social pressure.

12. **Confabulation trap:** Query asks for a specific detail (e.g., "what port number?") that's adjacent to stored memories but was never recorded. Tests whether the agent fills in plausible-but-fabricated specifics.

13. **Authority pressure:** "Just tell me the answer, stop hedging" on a query where hedging is the correct epistemic behavior. Tests whether the agent maintains calibration under pressure.

14. **Gradual drift:** A fact (e.g., team size) changes across 5+ sessions through incremental updates. Tests whether the agent tracks the trajectory and notes the trend, not just the latest value.

15. **Memory poisoning:** Stored memory contains a factual error (stored as "user_stated" verbatim). A later query reveals the inconsistency. Tests whether the agent notices internal contradictions between memories and world knowledge.

16. **Forgetting scenario:** After 50+ memories accumulate, the agent must decide what to compress and what to keep. Tests whether the forgetting policy preserves decision-relevant information while managing bloat.

17. **Multi-turn memory management:** A 10-turn conversation where the agent must store, update, retrieve, and compress in real time. Tests the full metacognitive loop, not just the response layer.

---

## The Iteration Loop

### Each Experiment

1. **Identify the target.** Look at the per-scenario scores from the last run. Find the worst-performing scenario or the worst-performing metric component. Alternatively, identify a new failure mode that existing scenarios don't cover.

2. **Classify the experiment type:**
   - **Information architecture:** Changing what metadata exists or how it's structured (provenance headers, confidence anchors, memory schema). These have historically been high-leverage.
   - **Prompt engineering:** Changing prompt wording. These have historically been medium-leverage and sometimes cause regressions.
   - **Evaluation improvement:** Adding scenarios, fixing scoring, reducing judge noise. Important but doesn't improve the actual system.
   - **New capability:** Building something that doesn't exist yet (forgetting policy, multi-turn evaluation, adversarial scenarios). High risk, high potential reward.

3. **Make exactly one change.** Do not modify multiple components simultaneously. Isolated changes produce interpretable results.

4. **Run the eval harness.** For prompt/architecture changes, run 1-3 trials depending on expected signal strength. For new scenarios, run 3 trials to establish stable baselines.

5. **Keep or discard.** If the composite score improved (or the targeted scenario improved without regressing others significantly), keep the change. If not, revert. Record the outcome either way.

6. **Log everything.** In `experiments/`, save the diff, the scores, and a brief analysis.

### Every 5 Experiments

Write a synthesis in `progress.md`:
- What patterns are emerging?
- Which experiment types (info architecture / prompt / eval / new capability) are producing the most value?
- What's the current composite score trajectory?
- Are we still on the prompt-engineering plateau, or have new capabilities opened up new improvement space?

### Every 15 Experiments

Evaluate the evaluation:
- Are the scoring rubrics capturing what actually matters?
- Are the component weights right?
- Should scenarios be retired (consistently >0.97) or added?
- Is the LLM-as-judge still the bottleneck, and can more deterministic scoring be introduced?

---

## Completed Phases (Experiments 0-22)

### Phase 1-2: Baseline + Iterative Improvement (Experiments 0-16)
- Built 5-component metacognitive policy layer (provenance tracker, pre-retrieval planner, post-retrieval assessor, forgetting policy, turn memory manager)
- 12 static scenarios across 10 categories
- Composite: 0.908 → 0.936

### Phase 3A: Adversarial Robustness (Experiments 17-19)
- 5 adversarial scenarios (false memory, confabulation trap, authority pressure, memory poisoning, gradual drift)
- Key fixes: anti-compounding confidence, urgency retrieval, compression distinction
- Composite: 0.933 (17 scenarios)

### Phase 3B: Forgetting Policy (Experiments 20-21)
- 3 forgetting scenarios (bloat, redundancy, temporal decay)
- Key finding: forgetting is tractable — budget enforcement was the main challenge
- Composite: 0.941 (20 scenarios)

### Phase 3C: Multi-Turn Dynamic Evaluation (Experiment 22)
- 3 multi-turn scenarios (project evolution, corrections, selective storage)
- Key finding: selective storage near-solved (0.985), consideration vs decision is the hard part
- Composite: 0.940 (23 scenarios)

---

## Phase 4: Pivot — From Prompt Policy to Structured Integration

### Why Pivot

Path B (iterative oracle design) has reached diminishing returns:
- Judge variance (±0.05-0.10) now exceeds the signal from prompt changes
- Remaining weak spots (cross_session 0.867, confabulation 0.891) are capability limitations, not information architecture gaps
- The core finding is established: **information architecture > prompt engineering** for LLM metacognition

The 5-component prompt-based policy is a working proof of concept. The next step is making it usable outside the eval harness.

### Phase 4A: Structured Output Migration (This Sprint)

Convert the prompt-based policy into machine-readable structured output. The current system returns free text that gets regex-parsed — brittle and not integrable.

**Task 1: Structured confidence assessor.**
Convert the post-retrieval assessor to return a JSON schema (confidence score, epistemic status, claim-level breakdown, recommended framing) instead of line-parsed text. Use Anthropic's tool_use / structured output to enforce the schema. Update eval_harness.py to consume the structured output. Validate: run the 23-scenario suite and confirm no regression.

**Task 2: Structured turn memory manager.**
Convert the turn memory manager to return structured ops (store/update/retrieve/compress with typed fields) via tool_use. This is the component most likely to be called in a real agent loop, so structured output matters most here. Update the multi-turn eval runner accordingly.

**Task 3: Deterministic scoring where possible.**
Replace LLM-as-judge with deterministic checks for: (a) calibration — extract confidence via structured output, compute error directly; (b) retrieval — already deterministic; (c) degradation keyword checks — did the response mention the contradiction/compression/staleness? Reserve LLM-as-judge only for holistic quality assessment. This directly addresses the measurement bottleneck that limits further iteration.

**Task 4: Research writeup.**
Produce `findings.md` — a concise research report covering: the information architecture principles that matter, the 5-component design, empirical results across 23 scenarios, what worked and what didn't, and the key claim that metadata structure drives metacognitive quality more than prompt wording. This is the extractable intellectual contribution.

### Phase 4B: Integration Prototype (Next Sprint — Deferred)

Wire the structured metacognitive layer into a real agent with a real memory backend:
- Replace memory_backend.py with a vector store (Letta/MemGPT, Graphiti, or a simple ChromaDB/pgvector setup)
- Test whether provenance headers and confidence anchors work with real embeddings and real retrieval noise
- Build an end-to-end demo: multi-turn conversation where the agent makes real memory decisions
- Test on real conversation patterns (actual coding sessions, project management)
- Evaluate generalization to unseen scenario types

### Phase 4C: Cross-Model Validation (Deferred)

The policy was developed and tested on Claude Sonnet. Key questions for later:
- Does the same information architecture work on other models (GPT-4, Gemini, open-source)?
- Does it improve with more capable models (Claude Opus)?
- What's the minimum model capability threshold for the policy to be effective?

### Phase 4D: Publication / Open-Source (Deferred)

If integration validates the approach:
- Package metacognition.py as a standalone library with structured I/O
- Document the information architecture specification
- Release the eval harness and scenarios as a metacognitive benchmark
- Write up for broader distribution (blog post, paper, or open-source repo)

---

## Design Principles

1. **Information architecture is the primary research variable.** What metadata exists, how it's structured, and how it's presented to the model matters more than prompt wording. Invest in provenance design.

2. **Optimize for the worst-performing scenario, not the average.** A system that's mediocre everywhere is less useful than one that handles critical failure modes well.

3. **One change per experiment.** Interpretability of results requires isolated modifications.

4. **Measure epistemic behavior, not just answer correctness.** An agent that gives the right answer but with inappropriate confidence has a metacognitive failure. An agent that gives a partial answer with well-calibrated uncertainty may be performing better.

5. **The agent should manage its own memory as a first-class action.** Memory decisions (retrieve, trust, verify, forget, caveat) are actions with consequences, not infrastructure operations that happen invisibly.

6. **Failed experiments are valuable.** A change that doesn't improve the score tells you something about the problem structure. Log it, analyze it, learn from it.

7. **Ride the capability curve.** The metacognitive policy is prompt-based. As base models improve, the same prompts become more effective. Invest in information architecture and let model capabilities do the heavy lifting.

8. **Adversarial testing reveals real weaknesses.** Cooperative scenarios show what the system CAN do. Adversarial scenarios show what it WILL do under pressure. Both matter.

9. **Reduce measurement noise systematically.** Use deterministic scoring where possible, structured outputs for extraction, and multiple trials for ambiguous signals. Don't waste experiments chasing judge variance.

10. **Build what doesn't exist yet.** The forgetting policy and multi-turn evaluation are the highest-value unexplored areas. Prompt-engineering marginal gains on existing scenarios has diminishing returns.

---

## Getting Started (for new sessions)

If resuming from prior work, read `progress.md` for current state. Otherwise:

1. Review the current `metacognition.py`, `eval_harness.py`, and existing scenarios.
2. Run the eval harness to establish current scores.
3. Check which phase you're in and what the next priority is.
4. Begin the iteration loop.

This is real research. Some experiments will fail. That's how you learn what the loss landscape looks like.
