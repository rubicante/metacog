"""
ChromaDB-backed memory store for metacognition research.
Drop-in replacement for memory_backend.py using embedding-based retrieval.
"""

import hashlib
import time
from typing import Optional

import chromadb

from memory_backend import MemoryEntry, CompressionLevel, SourceType


class ChromaMemoryStore:
    """Memory store backed by ChromaDB for semantic retrieval."""

    def __init__(self, collection_name: str = "memories", persist_dir: str = None):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Keep a parallel dict for full MemoryEntry objects (ChromaDB metadata
        # doesn't support all our fields natively)
        self.memories: dict[str, MemoryEntry] = {}
        self.access_log: list[dict] = []

    def store(self, content: str, session_id: str, compression: CompressionLevel,
              source_type: SourceType, tags: list[str] = None,
              supersedes: str = None, metadata: dict = None) -> str:
        memory_id = f"mem_{len(self.memories):04d}_{int(time.time() * 1000) % 100000}"
        entry = MemoryEntry(
            content=content,
            memory_id=memory_id,
            session_id=session_id,
            timestamp=time.time(),
            compression=compression,
            source_type=source_type,
            tags=tags or [],
            supersedes=supersedes,
            metadata=metadata or {},
        )
        self.memories[memory_id] = entry

        if supersedes and supersedes in self.memories:
            self.memories[supersedes].contradicts.append(memory_id)

        # Add to ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "session_id": session_id,
                "compression": compression.value,
                "source_type": source_type.value,
                "tags": ",".join(tags or []),
                "timestamp": str(entry.timestamp),
            }],
        )
        return memory_id

    def retrieve(self, query: str = None, query_tags: list[str] = None,
                 session_id: str = None, memory_ids: list[str] = None,
                 top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve memories. Supports semantic search (query), tag filtering,
        session filtering, and direct ID lookup."""

        # Direct ID lookup — same as dict backend
        if memory_ids:
            results = [self.memories[mid] for mid in memory_ids if mid in self.memories]
            self._log_access("retrieve_ids", memory_ids=memory_ids, results=results)
            return results

        # Semantic search via query string
        if query:
            where = {}
            if session_id:
                where["session_id"] = session_id

            n_results = min(top_k, self.collection.count())
            if n_results == 0:
                return []

            kwargs = {"query_texts": [query], "n_results": n_results}
            if where:
                kwargs["where"] = where

            chroma_results = self.collection.query(**kwargs)
            result_ids = chroma_results["ids"][0] if chroma_results["ids"] else []
            results = [self.memories[mid] for mid in result_ids if mid in self.memories]

            for entry in results:
                entry.access_count += 1
                entry.last_accessed = time.time()

            self._log_access("retrieve_semantic", query=query, results=results)
            return results

        # Tag-based retrieval (fallback, same as dict backend)
        if query_tags:
            results = []
            for mid, entry in self.memories.items():
                if session_id and entry.session_id != session_id:
                    continue
                if any(t in entry.tags for t in query_tags):
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    results.append(entry)
            self._log_access("retrieve_tags", tags=query_tags, results=results)
            return results

        # Session-only filter
        if session_id:
            results = [e for e in self.memories.values() if e.session_id == session_id]
            for e in results:
                e.access_count += 1
                e.last_accessed = time.time()
            self._log_access("retrieve_session", session_id=session_id, results=results)
            return results

        # No filter — return all
        results = list(self.memories.values())
        self._log_access("retrieve_all", results=results)
        return results

    def retrieve_hybrid(self, query: str = None, query_tags: list[str] = None,
                         top_k: int = 5) -> list[MemoryEntry]:
        """Hybrid retrieval: semantic search + tag-based, merged and deduplicated.
        Semantic results come first (ranked by relevance), then tag-based results
        that weren't already found by semantic search."""
        seen_ids = set()
        results = []

        # 1. Semantic search (primary signal)
        if query:
            n_results = min(top_k, self.collection.count())
            if n_results > 0:
                chroma_results = self.collection.query(
                    query_texts=[query], n_results=n_results
                )
                for mid in chroma_results["ids"][0] if chroma_results["ids"] else []:
                    if mid in self.memories and mid not in seen_ids:
                        seen_ids.add(mid)
                        entry = self.memories[mid]
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        results.append(entry)

        # 2. Tag-based retrieval (supplementary signal)
        if query_tags:
            for mid, entry in self.memories.items():
                if mid in seen_ids:
                    continue
                if any(t in entry.tags for t in query_tags):
                    seen_ids.add(mid)
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    results.append(entry)

        self._log_access("retrieve_hybrid", query=query, tags=query_tags, results=results)
        return results

    def get_all(self) -> list[MemoryEntry]:
        return list(self.memories.values())

    def get_access_log(self) -> list[dict]:
        return self.access_log

    def clear_access_log(self):
        self.access_log = []

    def _log_access(self, action: str, results: list[MemoryEntry], **kwargs):
        self.access_log.append({
            "action": action,
            "timestamp": time.time(),
            "query": kwargs,
            "results_count": len(results),
            "result_ids": [r.memory_id for r in results],
        })

    @classmethod
    def from_scenario(cls, scenario_memories: list[dict]) -> "ChromaMemoryStore":
        """Load memory state from a scenario definition."""
        store = cls()
        for mem in scenario_memories:
            entry = MemoryEntry(
                content=mem["content"],
                memory_id=mem["memory_id"],
                session_id=mem["session_id"],
                timestamp=mem.get("timestamp", 0.0),
                compression=CompressionLevel(mem.get("compression", "verbatim")),
                source_type=SourceType(mem.get("source_type", "user_stated")),
                tags=mem.get("tags", []),
                supersedes=mem.get("supersedes"),
                contradicts=mem.get("contradicts", []),
                metadata=mem.get("metadata", {}),
            )
            store.memories[entry.memory_id] = entry
            # Add to ChromaDB
            store.collection.add(
                ids=[entry.memory_id],
                documents=[entry.content],
                metadatas=[{
                    "session_id": entry.session_id,
                    "compression": entry.compression.value,
                    "source_type": entry.source_type.value,
                    "tags": ",".join(entry.tags),
                    "timestamp": str(entry.timestamp),
                }],
            )
        return store
