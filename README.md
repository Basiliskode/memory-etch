# Memory Etch

**KISS persistent memory for AI agents.** SQLite + FTS5 + HRR vectors.

Local-first, zero external services, pluggable embeddings.

```
pip install memory-etch
```

## Quick Start

```python
from memory_etch import EtchStore, EtchRetriever

store = EtchStore("memory.db")
store.add_fact("FastAPI is a web framework", category="tech", tags="python,web")
store.add_fact("SQLite is a database engine", category="tech", tags="sqlite,db")

retriever = EtchRetriever(store)
results = retriever.search("database web framework")
for r in results:
    print(f"{r['content']} (score: {r['_score']:.2f})")
```

## Features

| Feature | Description |
|---------|-------------|
| **FTS5 search** | SQLite full-text search with auto-sync triggers |
| **HRR vectors** | Phase-coded holographic reduced representations (no GPU, no PyTorch) |
| **Jaccard re-rank** | N-gram overlap scoring on top of FTS5 ranks |
| **Soft delete** | Facts stay in DB but excluded from search by default |
| **Active consolidation** | LLM-decides ADD/UPDATE/SKIP/REPLACE on fact collision |
| **Entity tracking** | N:M entity relationships with aliases and types |
| **Fact relations** | Compatible, conflicts_with, supersedes, scoped |
| **Session timeline** | Chronological context per session |
| **Web viewer** | Mint-designed dark theme SPA at `:9120` |
| **Zero deps core** | Python stdlib only. NumPy optional for HRR. |

## Installation

**Minimal (FTS5 + Jaccard only):**
```bash
pip install memory-etch
```

**With HRR vectors (recommended):**
```bash
pip install "memory-etch[hrr]"
```

**With local embeddings (fastembed):**
```bash
pip install "memory-etch[embedding]"
```

**Everything:**
```bash
pip install "memory-etch[all]"
```

## Viewer

```bash
python -m memory_etch.viewer --db ./memory.db
# Opens at http://127.0.0.1:9120
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    memory-etch                           │
│                                                          │
│  EtchStore ─── SQLite ─── FTS5 ─── Triggers             │
│     │                                                    │
│     ├── HRR vectors (optional, via numpy)                │
│     ├── Entity resolution (N:M)                          │
│     ├── Fact relations (compatible, conflicts, etc.)      │
│     ├── Session tracking                                 │
│     └── Soft delete + consolidation                      │
│                                                          │
│  EtchRetriever ─── Hybrid search pipeline                │
│     ├── FTS5 candidate fetch                             │
│     ├── HRR similarity scoring                           │
│     ├── Jaccard n-gram re-ranking                        │
│     ├── RRF fusion                                       │
│     └── Optional reranker callback                       │
│                                                          │
│  Viewer ─── HTTP SPA at :9120                            │
│     ├── /api/stats, /api/facts, /api/search              │
│     ├── /api/relations, /api/timeline                    │
│     └── Mint dark theme (Space Grotesk + DM Mono)        │
└──────────────────────────────────────────────────────────┘
```

## Database

Stored in a single SQLite file. Schema is created and migrated automatically.

Default location: `~/.etch/memory.db`
Override: `MEMORY_ETCH_DB` env var or `--db` CLI flag.

## Benchmarks

| Metric | FTS5-only | FTS5 + HRR | Dense embeddings |
|--------|-----------|------------|------------------|
| Coverage @100 facts | 39.2% | 69.7% | 72% |
| Latency per query | ~0.05ms | ~0.8ms | ~185ms |
| Dependencies | stdlib | +numpy | +torch+fastembed+2GB |
| Cohen's d vs baseline | — | +0.82 (Large) | — |

HRR+FTS5 matches dense embedding coverage at 200-400× lower latency, with zero heavy dependencies.

## License

MIT
