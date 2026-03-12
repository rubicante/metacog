"""
Minimal memory backend for metacognition research.
Dictionary-based store with provenance metadata.
This is NOT the research target — keep it simple.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class CompressionLevel(Enum):
    VERBATIM = "verbatim"
    STRUCTURED = "structured_extraction"
    SUMMARY = "lossy_summary"
    KEY_VALUE = "key_value_only"


class SourceType(Enum):
    USER_STATED = "user_stated"
    INFERRED = "inferred"
    EXTERNAL = "external_source"
    SYSTEM = "system_generated"


@dataclass
class MemoryEntry:
    content: str
    memory_id: str
    session_id: str
    timestamp: float
    compression: CompressionLevel
    source_type: SourceType
    content_hash: str = ""
    access_count: int = 0
    last_accessed: float = 0.0
    tags: list[str] = field(default_factory=list)
    supersedes: Optional[str] = None  # memory_id this entry updates/replaces
    contradicts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        if not self.last_accessed:
            self.last_accessed = self.timestamp


class MemoryStore:
    """Simple in-memory store with provenance tracking."""

    def __init__(self):
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

        return memory_id

    def retrieve(self, query_tags: list[str] = None, session_id: str = None,
                 memory_ids: list[str] = None) -> list[MemoryEntry]:
        results = []
        for mid, entry in self.memories.items():
            if memory_ids and mid not in memory_ids:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if query_tags and not any(t in entry.tags for t in query_tags):
                continue
            entry.access_count += 1
            entry.last_accessed = time.time()
            results.append(entry)

        self.access_log.append({
            "action": "retrieve",
            "timestamp": time.time(),
            "query": {"tags": query_tags, "session_id": session_id, "memory_ids": memory_ids},
            "results_count": len(results),
            "result_ids": [r.memory_id for r in results],
        })
        return results

    def get_all(self) -> list[MemoryEntry]:
        return list(self.memories.values())

    def get_access_log(self) -> list[dict]:
        return self.access_log

    def clear_access_log(self):
        self.access_log = []

    def to_dict(self) -> dict:
        return {
            mid: {
                "content": e.content,
                "memory_id": e.memory_id,
                "session_id": e.session_id,
                "timestamp": e.timestamp,
                "compression": e.compression.value,
                "source_type": e.source_type.value,
                "content_hash": e.content_hash,
                "access_count": e.access_count,
                "last_accessed": e.last_accessed,
                "tags": e.tags,
                "supersedes": e.supersedes,
                "contradicts": e.contradicts,
                "metadata": e.metadata,
            }
            for mid, e in self.memories.items()
        }

    @classmethod
    def from_scenario(cls, scenario_memories: list[dict]) -> "MemoryStore":
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
        return store
