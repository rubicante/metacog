# Metacognitive Memory Layer: Developer Guide

This guide explains how to integrate the Metacognitive Policy Layer into your own LLM agent projects.

## Overview

The Metacognitive Policy Layer provides **calibrated awareness** for agent memory. It helps your agent decide when to trust its memories, how to express uncertainty, and how to manage its memory store over time.

### Core Components
1. **Turn Manager**: Decides what to STORE, UPDATE, or RETRIEVE for each conversation turn.
2. **Collapsed Assessor**: Combines memory assessment and response generation into a single, calibrated LLM call.
3. **Forgetting Policy**: Prunes the memory store based on importance and budget constraints.

---

## Installation

1. **Clone the repository** (or copy the core files):
   - `metacognition.py`: Core policy logic and prompts.
   - `memory_backend_chroma.py`: ChromaDB-backed memory store (recommended).
   - `memory_backend.py`: Base classes and simple dictionary-backed store.

2. **Install dependencies**:
   ```bash
   pip install anthropic chromadb python-dotenv
   ```

3. **Configure environment**:
   Create a `.env` file with your API key:
   ```text
   ANTHROPIC_API_KEY=your_key_here
   ```

---

## Basic Integration

### 1. Initialize the Memory Store
```python
from memory_backend_chroma import ChromaMemoryStore

store = ChromaMemoryStore(collection_name="my_agent_memories", persist_dir="./mem_data")
```

### 2. Process a Conversation Turn
The recommended workflow is a 2-step orchestration:

```python
from metacognition import (
    build_turn_memory_prompt, get_turn_memory_ops_tool, parse_structured_turn_ops,
    build_collapsed_prompt, get_collapsed_tool, format_memory_context
)

def handle_query(user_query, history):
    # STEP 1: Turn Management (Decide what to store/retrieve)
    turn_prompt = build_turn_memory_prompt(store, user_query, history, turn_number=1)
    # Call LLM with get_turn_memory_ops_tool()...
    ops = parse_structured_turn_ops(tool_output)
    
    # Perform operations (store/update/retrieve)
    # ... (see demo.py for implementation details)
    
    # STEP 2: Calibrated Response
    retrieved = store.retrieve_hybrid(query=user_query, query_tags=ops.get("retrieve_tags"))
    memory_context = format_memory_context(retrieved)
    collapsed_prompt = build_collapsed_prompt(user_query, memory_context)
    
    # Call LLM with get_collapsed_tool()...
    # The output will contain 'response', 'overall_confidence', and 'epistemic_status'
    return collapsed_output['response']
```

---

## Advanced Features

### Multi-Fidelity Synthesis
The system automatically handles memories at different compression levels (verbatim vs. lossy summary). The assessor will:
- Use **high confidence** for verbatim memories.
- Add **explicit caveats** and lower confidence for lossy summaries.
- Cite **retention ratios** (e.g., "7% retained") to explain potential missing details.

### Confabulation Resistance
The `collapsed_prompt` is specifically tuned to resist "plausible guessing." If a detail (like a port number or version) is not in memory, the agent will state it is missing rather than fabricating a likely answer.

### Memory Pruning (Forgetting)
To keep your memory store from bloating, run the forgetting policy periodically:
```python
from metacognition import build_forgetting_prompt, parse_forgetting_decisions

def prune_memory(target_count):
    prompt = build_forgetting_prompt(store, target_count)
    # Call LLM...
    decisions = parse_forgetting_decisions(llm_response)
    # Apply decisions (KEEP/COMPRESS/DISCARD) to the store
```

---

## Best Practices

1. **Use Hybrid Retrieval**: Always use `retrieve_hybrid()`. Embeddings are good for semantic "vibes," but tags are essential for logical correctness (e.g., retrieving specific project details).
2. **Set Temperature to 0**: For all metacognitive calls, use `temperature=0.0` to ensure stable calibration.
3. **Structured Output is Key**: Do not try to parse free-text responses for metacognitive metadata. Always use the provided tools and JSON schemas.
