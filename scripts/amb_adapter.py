"""AMB adapter for memento — SQLite FTS5 + HRR persistent memory.

Drop this file into the AMB source tree and register it:

    cp scripts/amb_adapter.py <amb-repo>/src/memory_bench/memory/etch.py

Then add to ``<amb-repo>/src/memory_bench/memory/__init__.py``:

    from .etch import EtchMemoryProvider
    REGISTRY["etch"] = EtchMemoryProvider

Run:

    uv sync  # or: pip install -e .
    uv run amb run --dataset personamem --split 32k --memento --query-limit 20

Requires the ``memento`` package:
    pip install memento
    # or for development: pip install -e /path/to/memento
"""

import logging
from pathlib import Path
from typing import Optional

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


# ── Minimal AMB type stubs (standalone compatible) ──────────────────────

@dataclass
class Document:
    """Memory document as defined by AMB."""
    id: str
    content: str
    user_id: Optional[str] = None
    messages: Optional[list[dict]] = field(default_factory=dict)
    timestamp: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[dict] = None


class MemoryProvider(ABC):
    """Memory provider base class (AMB-compatible)."""
    name: str = ""
    description: str = ""
    kind: str = ""

    def prepare(self, store_dir: str, unit_ids: Optional[list[str]] = None,
                reset: bool = False) -> None:
        """Optional setup before ingest."""

    def initialize(self) -> None:
        """Optional one-time initialization."""

    @abstractmethod
    def ingest(self, documents: list[Document]) -> None:
        """Ingest documents into memory."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10,
                 user_id: Optional[str] = None,
                 query_timestamp: Optional[str] = None,
                 ) -> tuple[list[Document], Optional[dict]]:
        """Retrieve documents relevant to the query."""

    def cleanup(self) -> None:
        """Optional cleanup after evaluation."""

logger = logging.getLogger(__name__)

# ── Smart keyword extraction for FTS5 fallback ─────────────────────────

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
    """Extract meaningful search keywords, dropping stopwords and short tokens."""
    import re
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class EtchMemoryProvider(MemoryProvider):
    """memento: local SQLite FTS5 + HRR persistent memory.

    Features:
    - FTS5 full-text search (no external dependencies)
    - HRR (Holographic Reduced Representations) vector similarity when NumPy is available
    - Jaccard n-gram lexical re-ranking
    - Trust scoring with retrieval feedback loop
    - Topic-keyed upsert for evolving facts
    - Soft delete with audit trail

    No external embeddings or API keys required for core functionality.
    """

    name = "etch"
    description = (
        "memento: SQLite FTS5 + HRR persistent memory. "
        "FTS5 keyword search with HRR semantic similarity. "
        "Trust scoring, retrieval feedback, topic upsert."
    )
    kind = "local"
    concurrency = 1  # SQLite write lock — serialize

    def __init__(self) -> None:
        self._store: Optional["EtchStore"] = None
        self._db_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(
        self,
        store_dir: str,
        unit_ids: list[str] | None = None,
        reset: bool = True,
    ) -> None:
        """Initialize EtchStore in ``store_dir``.

        Creates the database file and runs schema migrations.
        When ``reset=True``, an existing database is removed first.
        """
        from memento import EtchStore

        self._db_path = Path(store_dir) / "etch_memory.db"
        if reset and self._db_path.exists():
            self._db_path.unlink()

        self._store = EtchStore(str(self._db_path))
        logger.info("EtchStore ready at %s", self._db_path)

    def cleanup(self) -> None:
        """Close the store and release resources."""
        if self._store is not None:
            try:
                self._store.close()
                logger.info("EtchStore closed")
            except Exception as exc:
                logger.warning("Error closing EtchStore: %s", exc)
            self._store = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        """Ingest documents as facts.

        Each document is stored as a fact. The ``user_id`` field maps
        to memento's ``project`` namespace for per-user isolation.
        Duplicate content is silently deduplicated (``INSERT OR IGNORE``).
        """
        if self._store is None:
            raise RuntimeError("Provider not prepared — call prepare() first")

        count = 0
        for doc in documents:
            self._store.add_fact(
                content=doc.content,
                project=doc.user_id or "",
            )
            count += 1

        # Flush any pending HRR vectors before the benchmark starts
        self._flush_hrr()

        logger.info("Ingested %d documents into etch", count)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        """Retrieve top-k relevant documents.

        Strategy: fetch candidates broadly via ``search_expanded`` (FTS5 with
        keyword breadth), then re-score ALL candidates against the ORIGINAL
        query for best semantic matching. This avoids the Jaccard/HRR penalty
        on long informative facts when using a truncated fallback query.
        """
        if self._store is None:
            raise RuntimeError("Provider not prepared — call prepare() first")

        from memento import EtchRetriever

        retriever = EtchRetriever(self._store)

        # Step 1: collect candidates via search_expanded (broad FTS5)
        # Uses OR-join and single-keyword fallback internally
        candidates = retriever.search_expanded(
            query, limit=k * 6, project=user_id or "",
        )

        if not candidates:
            return [], {"strategy": "auto", "count": 0}

        # Step 2: re-score ALL candidates with the ORIGINAL query
        # This fixes the key issue: fallback queries like "Bob" or
        # "Bob OR profession" lose semantic context, making Jaccard
        # favor short facts. Scoring with the full query "What is Bob's
        # profession?" gives better discrimination.
        scored = retriever._score_candidates(query, candidates)
        scored.sort(key=lambda x: x.get("_score", 0), reverse=True)

        # Step 3: take top-k and return
        docs = [
            Document(
                id=str(r.get("fact_id", 0)),
                content=r.get("content", ""),
                user_id=r.get("project") or user_id,
            )
            for r in scored[:k]
        ]
        return docs, {"strategy": "auto", "count": len(docs)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_hrr(self) -> None:
        """Flush pending HRR vectors synchronously."""
        if self._store is None or not hasattr(self._store, "_hrr_flush_signal"):
            return
        try:
            self._store._hrr_flush_signal.set()
            thread = getattr(self._store, "_hrr_flush_thread", None)
            if thread and thread.is_alive():
                thread.join(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hybrid: multi-strategy + embeddings RRF (etch-hybrid)
# ---------------------------------------------------------------------------

class EtchHybridMemoryProvider(EtchMemoryProvider):
    """Combined multi-strategy + embedding search fused via RRF.

    Uses the EtchRetriever multi-strategy fallback AND the EtchStore
    embedding search, then merges both result sets via RRF for maximum
    recall.

    Requires ``fastembed``:
        pip install memento[embeddings]
    """

    name = "etch-hybrid"
    description = (
        "memento with multi-strategy FTS5/HRR + BGE-small embeddings "
        "fused via RRF for maximum recall."
    )

    def prepare(
        self,
        store_dir: str,
        unit_ids: list[str] | None = None,
        reset: bool = True,
    ) -> None:
        if _HAS_FASTEMBED:
            try:
                from memento.embedding.fastembed_provider import FastembedProvider
                self._embedder = FastembedProvider()
            except Exception as exc:
                logger.warning("fastembed init failed: %s — falling back to FTS5+HRR", exc)
                self._embedder = None
        else:
            self._embedder = None

        from memento import EtchStore

        self._db_path = Path(store_dir) / "etch_memory.db"
        if reset and self._db_path.exists():
            self._db_path.unlink()

        if self._embedder:
            self._store = EtchStore(str(self._db_path), embedding_provider=self._embedder)
        else:
            self._store = EtchStore(str(self._db_path))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        if self._store is None:
            raise RuntimeError("Provider not prepared — call prepare() first")

        from memento import EtchRetriever

        def _search_retriever(q: str) -> list[Document]:
            retriever = EtchRetriever(self._store)
            results = retriever.search(q, limit=k * 3, project=user_id or "")
            return [
                Document(
                    id=str(r.get("fact_id", 0)),
                    content=r.get("content", ""),
                    user_id=r.get("project") or user_id,
                )
                for r in results
            ]

        # Multi-strategy FTS5 fallback — try progressively simpler queries
        keywords = _extract_keywords(query)
        docs: list[Document] = []

        # 1a: Full query
        docs = _search_retriever(query)

        # 1b: Extracted keywords
        if not docs and len(keywords) >= 2:
            docs = _search_retriever(" ".join(keywords))

        # 1c: Individual keywords — wide net
        if not docs:
            for kw in keywords:
                docs = _search_retriever(kw)
                if docs:
                    break

        # 1d: Ultra-wide: try mode="auto" as last resort
        if not docs:
            retriever = EtchRetriever(self._store)
            results = retriever.search(
                query, limit=k, project=user_id or "", mode="auto"
            )
            docs = [
                Document(
                    id=str(r.get("fact_id", 0)),
                    content=r.get("content", ""),
                    user_id=r.get("project") or user_id,
                )
                for r in results
            ]

        return docs, {"strategy": "multi_strategy", "count": len(docs)}


# ---------------------------------------------------------------------------
# Optional: Embedding-backed variant (etch-emb)
# ---------------------------------------------------------------------------

try:
    from memento.embedding.fastembed_provider import FastembedProvider as _FastembedProvider

    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False


class EtchEmbMemoryProvider(EtchMemoryProvider):
    """memento with BGE-small-en-v1.5 embeddings via fastembed.

    Falls back to FTS5 + HRR when the embedding model is unavailable.
    Requires ``fastembed``:
        pip install memento[embeddings]
    """

    name = "etch-emb"
    description = (
        "memento with BGE-small-en-v1.5 embeddings (fastembed). "
        "FTS5 + HRR + embedding vector search fused via RRF."
    )

    def prepare(
        self,
        store_dir: str,
        unit_ids: list[str] | None = None,
        reset: bool = True,
    ) -> None:
        if _HAS_FASTEMBED:
            try:
                from memento.embedding.fastembed_provider import FastembedProvider

                self._embedder = FastembedProvider()
            except Exception as exc:
                logger.warning("fastembed init failed: %s — falling back to FTS5+HRR", exc)
                self._embedder = None
        else:
            self._embedder = None

        # Create store WITH embedding provider wired in
        from memento import EtchStore

        self._db_path = Path(store_dir) / "etch_memory.db"
        if reset and self._db_path.exists():
            self._db_path.unlink()

        if self._embedder:
            self._store = EtchStore(str(self._db_path), embedding_provider=self._embedder)
            logger.info("EtchStore ready at %s (etch-emb mode)", self._db_path)
        else:
            self._store = EtchStore(str(self._db_path))
            logger.info("EtchStore ready at %s (FTS5+HRR fallback)", self._db_path)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        if self._store is None:
            raise RuntimeError("Provider not prepared — call prepare() first")

        # Use store.search() which does FTS5 + embedding RRF (new path)
        results = self._store.search(query, limit=k, project=user_id or "")

        docs = [
            Document(
                id=str(r.get("fact_id", 0)),
                content=r.get("content", ""),
                user_id=r.get("project") or user_id,
            )
            for r in results
        ]

        return docs, None
