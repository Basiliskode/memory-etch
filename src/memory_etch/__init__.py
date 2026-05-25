"""memory-etch — KISS persistent memory for AI agents.

A local-first, SQLite-backed memory system with:
- FTS5 full-text search
- HRR (Holographic Reduced Representations) vector similarity
- Jaccard n-gram re-ranking
- Entity extraction and N:M relationships
- Fact relations (compatible, conflicts_with, supersedes, etc.)
- Session tracking with timeline
- Active consolidation (LLM-decide on fact collision)
- Lightweight web viewer

Quick start:

    from memory_etch import EtchStore, EtchRetriever

    store = EtchStore("memory.db")
    fid = store.add_fact("Hermes Agent uses python-telegram-bot v21",
                         category="project", tags="python,telegram")

    retriever = EtchRetriever(store)
    results = retriever.search("telegram bot")
    for r in results:
        print(r["content"], r["_score"])
"""

from . import embedding, hrr, ingest  # noqa: I001
from .classifier import QueryClassifier
from .embedding import EmbeddingProvider, NoopProvider
from .etch import EtchMemoryProvider, _extractor_get_provider_config
from .store import EtchStore
from .interceptor import InterceptorHandle, intercept, teardown_all
from .interceptor.generic import GenericInterceptor
from .retrieval import EtchRetriever

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python 3.10+ always has importlib.metadata
    PackageNotFoundError = Exception  # type: ignore[assignment]


try:
    __version__ = version("memory-etch")
except PackageNotFoundError:
    __version__ = "1.1.0"

__all__ = [
    "EtchStore",
    "EtchRetriever",
    "EtchMemoryProvider",
    "_extractor_get_provider_config",
    "QueryClassifier",
    "embedding",
    "EmbeddingProvider",
    "NoopProvider",
    "hrr",
    "ingest",
    "InterceptorHandle",
    "intercept",
    "teardown_all",
    "GenericInterceptor",
    "__version__",
]
