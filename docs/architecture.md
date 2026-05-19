# Memory Etch — Architecture

## Core Components

### `EtchStore` (store.py)
SQLite-backed fact store. Manages schema creation, CRUD, FTS5 sync, HRR encoding, soft delete, consolidation, sessions, and fact relations.

**Key tables:**
- `facts` — Core fact storage with trust scores, HRR vectors, importance, project, session tracking
- `entities` — Entity names with types and aliases
- `fact_entities` — N:M relationship between facts and entities
- `facts_fts` — FTS5 virtual table synced via triggers
- `sessions` — Work sessions with start/end times
- `fact_relations` — Semantic relations between facts (compatible, conflicts_with, etc.)
- `extractions` — Auto-extraction audit log
- `turn_buffer` — Conversation turn buffer for LLM extraction

### `EtchRetriever` (retrieval.py)
Hybrid search with FTS5 candidates → HRR similarity → Jaccard n-gram re-ranking → RRF fusion.

Search pipeline:
1. Fetch `limit × 2` candidates from FTS5
2. Encode query as HRR phase vector
3. Score each candidate: `trust + FTS5_rank + Jaccard + HRR_similarity × weight`
4. Sort by combined score
5. Optional external reranker callback

### `hrr` (hrr.py)
Holographic Reduced Representations with phase encoding.

- `encode_atom(word, dim)` — Deterministic SHA-256 → phase vector
- `encode_text(text, dim)` — Bag-of-words bundle
- `bind(a, b)` — Phase addition (associates concepts)
- `unbind(memory, key)` — Phase subtraction (retrieval)
- `bundle(*vectors)` — Circular mean (superposition)
- `similarity(a, b)` — Phase cosine similarity

### `QueryClassifier` (classifier.py)
Rule-based intent classification for memory queries. Routes to entity probe, project search, relation retrieval, or general FTS5.

## Data Flow

```
User Query
    │
    ▼
QueryClassifier.classify()
    │
    ├─ "entity" → EtchStore.get_entities() + EtchRetriever.search()
    ├─ "project" → EtchStore.list_facts(project=...)
    ├─ "relation" → EtchStore.get_relations()
    ├─ "timeline" → EtchStore.get_timeline()
    └─ "search" → EtchRetriever.search()
```

## HRR Details

- Default dimension: 256 (reduced from 512, 25% latency saving, identical keyword coverage)
- Auto-detects dimension from existing database vectors
- Async flush via daemon thread sharing connection with RLock
- In-memory vector cache (LRU, max 500 entries)
- Graceful degradation to FTS5+Jaccard when numpy unavailable

## Performance

| Operation | Latency (warm) | Notes |
|-----------|----------------|-------|
| FTS5 candidate fetch | ~538μs | limit × 2 headroom |
| HRR query encode | ~705μs | dim=256 |
| Jaccard per candidate | ~111μs | Word-level |
| HRR decode + similarity | ~629μs | Per candidate |
| HRR flush (10 pending) | ~10ms | Async, non-blocking |
| Total prefetch | ~2.5ms | Cold cache first query |

Coverage benchmark at 100 facts: Etch 69.7% vs Default 39.2% (Cohen d=+0.82 Large, -87% tokens).
