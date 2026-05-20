"""Memory-etch provider implementations for the benchmark.

These implement ``MemoryProvider`` using memory-etch's EtchStore/EtchRetriever.
Use them to benchmark memory-etch against any other MemoryProvider.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from .runner import MemoryProvider
from .dataset import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart keyword extraction for FTS5 fallback
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    "what", "is", "does", "have", "any", "the", "a", "an", "in", "at",
    "on", "for", "to", "of", "and", "or", "was", "were", "are", "who",
    "which", "how", "where", "when", "why", "do", "did", "has", "had",
    "s", "t", "m",
    "be", "been", "being", "it", "its", "with", "as", "by", "that",
    "this", "from", "about", "than", "so", "if", "no", "not", "just",
    "up", "out", "also", "very", "too", "can", "will",
    "would", "could", "should", "may", "might", "shall",
})


def _extract_keywords(query: str) -> list[str]:
    import re
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class EtchBenchmarkProvider(MemoryProvider):
    """memory-etch benchmark provider using FTS5 + HRR + search_expanded.

    Uses a multi-strategy fallback: full query → keywords OR → single terms,
    then re-scores candidates against the original query for best matching.
    """

    name = "etch"
    description = "memory-etch: SQLite FTS5 + HRR persistent memory"
    kind = "local"

    def __init__(self) -> None:
        self._store = None
        self._db_path = None

    def prepare(self, store_dir: str, reset: bool = False) -> None:
        from memory_etch import EtchStore
        self._db_path = Path(store_dir) / "etch_memory.db"
        if reset and self._db_path.exists():
            self._db_path.unlink()
        self._store = EtchStore(str(self._db_path))

    def ingest(self, documents: list[Document]) -> None:
        if self._store is None:
            raise RuntimeError("Provider not prepared")
        for doc in documents:
            self._store.add_fact(content=doc.content, project=doc.user_id or "")
        self._flush_hrr()

    def retrieve(self, query: str, k: int = 10, user_id: Optional[str] = None):
        if self._store is None:
            raise RuntimeError("Provider not prepared")

        from memory_etch import EtchRetriever
        retriever = EtchRetriever(self._store)

        # Step 1: fetch candidates via search_expanded (broad FTS5)
        candidates = retriever.search_expanded(
            query, limit=k * 6, project=user_id or "",
        )

        if not candidates:
            # Step 2: try single-keyword fallback
            keywords = _extract_keywords(query)
            if keywords:
                for kw in keywords:
                    candidates = retriever.search_expanded(
                        kw, limit=k * 6, project=user_id or "",
                    )
                    if candidates:
                        break

        if not candidates:
            return [], {"strategy": "empty"}

        # Step 3: re-score WITH original query
        scored = retriever._score_candidates(query, candidates)
        scored.sort(key=lambda x: x.get("_score", 0), reverse=True)

        docs = [
            Document(
                id=str(r.get("fact_id", 0)),
                content=r.get("content", ""),
                user_id=r.get("project") or user_id,
            )
            for r in scored[:k]
        ]
        return docs, {"strategy": "scored", "count": len(docs)}

    def cleanup(self) -> None:
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    def _flush_hrr(self) -> None:
        if self._store is None:
            return
        try:
            self._store._hrr_flush_signal.set()
            thread = getattr(self._store, "_hrr_flush_thread", None)
            if thread and thread.is_alive():
                thread.join(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Example: Minimal JSON-based provider for comparison
# ---------------------------------------------------------------------------

class JsonMemoryProvider(MemoryProvider):
    """Minimal IN-MEMORY (non-persistent) provider for baseline comparison.

    Stores documents in a JSON file and performs naive word-overlap search.
    Useful as a "worst case" baseline for the benchmark.
    """

    name = "json-baseline"
    description = "Naive JSON in-memory with word-overlap search (baseline)"
    kind = "local"

    def __init__(self):
        self._docs: list[Document] = []

    def prepare(self, store_dir: str, reset: bool = False) -> None:
        self._docs = []

    def ingest(self, documents: list[Document]) -> None:
        self._docs = list(documents)

    def retrieve(self, query: str, k: int = 10, user_id: Optional[str] = None):
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for doc in self._docs:
            if user_id and doc.user_id != user_id:
                continue
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        docs = [doc for _, doc in scored[:k]]
        return docs, {}

    def cleanup(self) -> None:
        self._docs = []
