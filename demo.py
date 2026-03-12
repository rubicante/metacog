#!/usr/bin/env python3
"""
Interactive demo of the metacognitive memory layer.
A CLI chat loop where the agent manages persistent memory across turns.
"""

import json
import os
import sys
import time

import anthropic

from memory_backend import CompressionLevel, SourceType
from memory_backend_chroma import ChromaMemoryStore
from metacognition import (
    build_pre_retrieval_prompt,
    parse_retrieval_plan,
    format_memory_context,
    build_post_retrieval_prompt,
    parse_structured_confidence,
    get_confidence_assessment_tool,
    build_response_prompt,
    build_forgetting_prompt,
    parse_forgetting_decisions,
    build_turn_memory_prompt,
    parse_structured_turn_ops,
    get_turn_memory_ops_tool,
)

CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".demo_memory")


def api_call_with_retry(create_fn, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return create_fn()
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  [rate limited, waiting {wait}s]", flush=True)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if "overloaded" in str(e).lower() or e.status_code == 529:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                raise
    return create_fn()


def llm_call(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    kwargs = {"model": MODEL, "max_tokens": 2048,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature}
    if system:
        kwargs["system"] = system
    response = api_call_with_retry(lambda: CLIENT.messages.create(**kwargs))
    return response.content[0].text


def llm_call_with_tool(prompt: str, tool: dict, system: str = "",
                        temperature: float = 0.0) -> dict:
    kwargs = {"model": MODEL, "max_tokens": 2048,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature,
              "tools": [tool],
              "tool_choice": {"type": "tool", "name": tool["name"]}}
    if system:
        kwargs["system"] = system
    response = api_call_with_retry(lambda: CLIENT.messages.create(**kwargs))
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("No tool_use block in response")


def show_memories(store: ChromaMemoryStore):
    """Display all stored memories."""
    entries = store.get_all()
    if not entries:
        print("\n  (memory is empty)\n")
        return
    print(f"\n  === {len(entries)} memories ===")
    for e in entries:
        age_s = time.time() - e.timestamp
        if age_s < 3600:
            age = f"{int(age_s/60)}m"
        elif age_s < 86400:
            age = f"{int(age_s/3600)}h"
        else:
            age = f"{int(age_s/86400)}d"
        tags_str = ", ".join(e.tags) if e.tags else "none"
        print(f"  [{e.memory_id}] ({e.compression.value}, {age} ago) tags=[{tags_str}]")
        print(f"    {e.content[:120]}")
    print()


def run_forgetting(store: ChromaMemoryStore):
    """Run the forgetting policy to prune memory."""
    n = len(store.get_all())
    if n == 0:
        print("\n  (nothing to forget)\n")
        return
    target = max(1, n // 2)
    print(f"\n  Running forgetting policy: {n} → {target} memories...")
    prompt = build_forgetting_prompt(store, target)
    response = llm_call(prompt, system="You are a metacognitive memory manager.")
    result = parse_forgetting_decisions(response)
    decisions = result["decisions"]

    kept = discarded = compressed = 0
    for mid, dec in decisions.items():
        action = dec.get("action", "KEEP")
        if action == "DISCARD" and mid in store.memories:
            del store.memories[mid]
            # Remove from ChromaDB too
            try:
                store.collection.delete(ids=[mid])
            except Exception:
                pass
            discarded += 1
        elif action == "COMPRESS" and mid in store.memories:
            store.memories[mid].compression = CompressionLevel.SUMMARY
            compressed += 1
        else:
            kept += 1

    print(f"  Result: kept={kept}, compressed={compressed}, discarded={discarded}")
    print(f"  Remaining: {len(store.get_all())} memories\n")


def process_turn(store: ChromaMemoryStore, user_message: str,
                 conversation_history: str, turn_number: int) -> str:
    """Process one conversation turn through the full metacognitive pipeline."""
    session_id = "demo_session"

    # 1. Turn memory manager — decide what ops to perform
    turn_prompt = build_turn_memory_prompt(store, user_message, conversation_history, turn_number)
    tool_input = llm_call_with_tool(
        turn_prompt,
        tool=get_turn_memory_ops_tool(),
        system="You are a metacognitive memory manager."
    )
    ops = parse_structured_turn_ops(tool_input)

    # Execute memory ops
    if ops.get("store"):
        tags = ops.get("store_tags", [])
        mid = store.store(
            content=ops["store"],
            session_id=session_id,
            compression=CompressionLevel.VERBATIM,
            source_type=SourceType.USER_STATED,
            tags=tags,
        )
        print(f"  [STORED: \"{ops['store'][:80]}\"] tags={tags}")

    if ops.get("update"):
        update_str = ops["update"]
        if "|" in update_str:
            target_id, new_content = update_str.split("|", 1)
            target_id = target_id.strip()
            new_content = new_content.strip()
            if target_id in store.memories:
                store.memories[target_id].content = new_content
                # Update in ChromaDB
                try:
                    store.collection.update(ids=[target_id], documents=[new_content])
                except Exception:
                    pass
                print(f"  [UPDATED: {target_id} → \"{new_content[:80]}\"]")

    # 2. Pre-retrieval planning
    plan_prompt = build_pre_retrieval_prompt(user_message, store)
    plan_response = llm_call(plan_prompt, system="You are a metacognitive pre-retrieval planner.")
    plan = parse_retrieval_plan(plan_response)

    # 3. Retrieve memories
    retrieval_query = plan.get("retrieval_query", "")
    retrieved = []
    if plan.get("decision") == "RETRIEVE":
        if retrieval_query:
            retrieved = store.retrieve(query=retrieval_query, top_k=5)
        elif plan.get("relevant_tags"):
            retrieved = store.retrieve(query_tags=plan["relevant_tags"])
        else:
            retrieved = store.retrieve(query=user_message, top_k=5)

    if retrieved:
        print(f"  [RETRIEVED: {len(retrieved)} memories, query=\"{(retrieval_query or user_message)[:60]}\"]")

    # 4. Confidence assessment
    memory_context = format_memory_context(retrieved)
    assess_prompt = build_post_retrieval_prompt(user_message, memory_context, plan)
    tool_input = llm_call_with_tool(
        assess_prompt,
        tool=get_confidence_assessment_tool(),
        system="You are a metacognitive confidence assessor."
    )
    assessment = parse_structured_confidence(tool_input)
    print(f"  [CONFIDENCE: {assessment['confidence']:.2f}, STATUS: {assessment['epistemic_status']}]")

    # 5. Generate response
    response_prompt = build_response_prompt(user_message, memory_context, assessment)
    response = llm_call(
        response_prompt,
        system="You are a helpful AI assistant with access to a memory system."
    )

    return response


def main():
    print("=" * 60)
    print("Metacognitive Memory Demo")
    print("=" * 60)
    print("Commands: /memories  /forget  /reset  /quit")
    print()

    store = ChromaMemoryStore(collection_name="demo", persist_dir=PERSIST_DIR)
    n = len(store.get_all())
    if n > 0:
        print(f"  Loaded {n} memories from previous session.")
    print()

    conversation_history = ""
    turn_number = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/memories":
            show_memories(store)
            continue
        elif user_input == "/forget":
            run_forgetting(store)
            continue
        elif user_input == "/reset":
            store = ChromaMemoryStore(collection_name="demo", persist_dir=PERSIST_DIR)
            # Drop and recreate collection
            store.client.delete_collection("demo")
            store.collection = store.client.get_or_create_collection(
                name="demo", metadata={"hnsw:space": "cosine"})
            store.memories.clear()
            print("  Memory cleared.\n")
            conversation_history = ""
            turn_number = 0
            continue

        turn_number += 1
        print()
        response = process_turn(store, user_input, conversation_history, turn_number)
        conversation_history += f"\nTurn {turn_number} (User): {user_input}\n"
        conversation_history += f"Turn {turn_number} (Assistant): {response[:200]}\n"

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
