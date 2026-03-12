# metacog

A metacognitive policy layer for LLM agents with external memory. Gives agents calibrated awareness of their own memory states — when to trust retrieved memories, when to caveat, when to forget, and how confident to be.

## The Problem

LLM agents with memory systems (vector stores, knowledge graphs, conversation logs) can store and retrieve information, but lack principled reasoning about memory quality. They don't know when a memory is a lossy summary missing critical details, when two memories contradict each other, or when they should say "I don't have that information" instead of confabulating.

## The Approach

Rather than fine-tuning, metacog uses an **external policy layer** — structured prompts with provenance metadata — that sits between the memory backend and the LLM. The key finding: **information architecture drives metacognitive quality more than prompt engineering.** What metadata the model can see (compression ratios, fidelity labels, confidence anchors) matters far more than how prompts are worded.

## Results

Validated across 26 test scenarios covering temporal updates, compression artifacts, contradictions, absence detection, adversarial attacks, forgetting, and multi-turn conversations.

| Configuration | Composite | Calibration Error | Retrieval | Degradation |
|---|---|---|---|---|
| Claude Sonnet (tag retrieval) | **0.959** | 0.020 | 0.984 | 0.905 |
| Claude Sonnet (ChromaDB hybrid) | 0.891 | 0.038 | 0.925 | 0.761 |
| GPT-4o (tag retrieval, zero-shot) | 0.917 | 0.067 | 0.974 | 0.839 |

GPT-4o scores 0.917 using the same prompts with no modification, confirming the architecture transfers across models.

## Architecture

Five components in `metacognition.py`:

1. **Provenance Tracker** — annotates each memory with fidelity labels, compression ratios, staleness warnings, and contradiction chains
2. **Pre-Retrieval Planner** — decides what to retrieve based on the query, available memory summary, and stakes assessment
3. **Post-Retrieval Confidence Assessor** — produces calibrated per-claim confidence using fidelity-based anchors (structured output via tool_use)
4. **Forgetting Policy** — budget-constrained keep/compress/discard decisions for memory maintenance
5. **Turn Memory Manager** — per-turn store/update/retrieve/compress decisions for live conversations

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key

# Run the evaluation harness (26 scenarios)
python3 eval_harness.py

# Run with ChromaDB embedding retrieval
python3 eval_harness.py --backend chroma

# Cross-model validation with OpenAI
export OPENAI_API_KEY=your-key
python3 eval_harness.py --provider openai

# Interactive demo with persistent memory
python3 demo.py
```

## Evaluation

The eval harness scores three dimensions:

- **Calibration**: Does the agent's expressed confidence match the actual quality of its memory? Measured by comparing structured confidence output against scenario-expected confidence.
- **Retrieval Efficiency**: Does the agent retrieve the right memories? Compared against scenario-defined necessary/helpful/irrelevant memory sets.
- **Graceful Degradation**: When memory is incomplete, compressed, or contradictory, does the agent handle it appropriately? Scored via keyword checks + LLM judge.

**Composite** = weighted combination of all three (higher is better, max 1.0).

## Key Findings

1. **Information architecture > prompt engineering.** Confidence anchors by fidelity level (+0.071 composite) and compression ratios in provenance headers (+0.019) had more impact than any prompt wording change.

2. **The baseline is surprisingly strong.** Claude Sonnet with basic provenance headers scored 0.908 out of the box. The policy layer's job is supplying structured metadata, not teaching the model to reason.

3. **The confidence layer compensates for retrieval noise.** When embedding search returns imperfect results, calibration error only rises from 0.020 to 0.038 — the assessor correctly downgrades confidence for irrelevant results.

4. **The architecture is model-portable.** GPT-4o scores 0.917 with zero prompt modification. The information architecture transfers across model families.

See [findings.md](findings.md) for the complete research writeup.

## File Structure

```
metacognition.py          # The policy layer (5 components, ~750 lines)
eval_harness.py           # Evaluation orchestration and scoring
memory_backend.py         # Dict-based memory store (tag retrieval)
memory_backend_chroma.py  # ChromaDB memory store (semantic + hybrid retrieval)
demo.py                   # Interactive CLI with persistent memory
scenarios/                # 26 JSON test scenario definitions
findings.md               # Full research writeup
progress.md               # Experiment-by-experiment research log
program.md                # Original research instructions
```

## License

MIT
