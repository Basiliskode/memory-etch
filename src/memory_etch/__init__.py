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

from .store import EtchStore
from .retrieval import EtchRetriever
from .classifier import QueryClassifier
from .etch import EtchMemoryProvider, _extractor_get_provider_config
from . import hrr

__all__ = [
    "EtchStore",
    "EtchRetriever",
    "EtchMemoryProvider",
    "_extractor_get_provider_config",
    "QueryClassifier",
    "hrr",
]

__version__ = "0.3.0"
