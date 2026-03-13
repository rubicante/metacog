# metacog: Metacognitive Memory Layer for LLM Agents

A policy layer that gives LLM agents **calibrated awareness** of their own memory. It sits between your memory backend (vector store, graph, or log) and your agent, providing principled reasoning about when to trust memories, how to express uncertainty, and when to forget.

## The Problem

LLM agents can store and retrieve information, but they lack **metacognition**. They don't know when a memory is a lossy summary missing critical details, when two memories contradict each other, or when they should admit ignorance instead of confabulating. Naive RAG often leads to "false confidence" in poor-quality retrievals.

## The Solution: A Metacognitive Policy Layer

`metacog` implements a validated information architecture that transforms raw memories into calibrated responses.

### Key Features
- **Fidelity-Based Calibration**: Uses "confidence anchors" to map memory types (verbatim, structured, lossy) to numeric confidence scores.
- **2-Step Orchestration**: Efficiently manages memory operations (STORE/UPDATE/RETRIEVE) and generates calibrated responses in just two LLM calls.
- **Confabulation Resistance**: Explicitly tuned to resist "plausible guessing" for details missing from memory.
- **Compression Awareness**: Recognizes when information was lost during summarization and cites retention ratios (e.g., "7% retained") to the user.
- **Hybrid Retrieval**: Combines semantic embeddings with logical tags to ensure both "vibes" and "facts" are captured.
- **Principled Forgetting**: Automatically prunes the memory store based on a budget, prioritizing decision-relevant facts over conversational noise.

---

## Results

Validated across 27 test scenarios covering temporal updates, compression artifacts, contradictions, adversarial traps, and multi-turn conversations.

| Configuration | Composite | Calibration Error | Retrieval | Degradation |
|---|---|---|---|---|
| **Claude Sonnet (Hybrid)** | **0.917** | **0.040** | **0.941** | **0.835** |
| Claude Sonnet (Tag-only) | 0.923 | 0.036 | 0.964 | 0.828 |
| GPT-4o (Tag-only) | 0.893 | 0.062 | 0.956 | 0.769 |

*The "Collapsed" architecture (Phase 6) achieves 0.917 composite while reducing LLM calls by 67% compared to multi-stage pipelines.*

---

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
```

### 2. Run the Interactive Demo
Chat with an agent that manages its own persistent memory:
```bash
python3 demo.py
```

### 3. Run the Evaluation Suite
Test the policy layer against 27 adversarial and logic-based scenarios:
```bash
python3 eval_harness.py --collapsed --backend chroma
```

---

## Architecture

The system uses a 2-step orchestration pattern:

1.  **Turn Manager**: Processes the user message to decide on state changes (STORE new facts, UPDATE corrections, RETRIEVE specific context).
2.  **Collapsed Assessor**: Evaluates retrieved memories and generates a response in a single call using the `assess_and_respond` tool, which enforces "epistemic accounting" through its JSON schema.

See [findings.md](findings.md) for the full research writeup and [USAGE.md](USAGE.md) for integration guides.

---

## Key Findings

1.  **Information Architecture > Prompt Engineering**: Metadata labels (like "VERBATIM") and confidence anchors had more impact (+0.071 composite) than any prompt wording change.
2.  **Tool-Use Drives Metacognition**: Forcing the model to produce claim-level confidences via a JSON schema is the most effective way to ensure calibration.
3.  **Metacognition Buffers Retrieval Noise**: The policy layer maintains low calibration error even when the underlying vector store returns imperfect results.

## License

MIT
