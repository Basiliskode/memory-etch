# Embedding Vector Search (Optional)

Memory Etch supports **optional embedding vector search** using the
[BGE-M3](https://huggingface.co/BAAI/bge-m3) model via
[fastembed](https://github.com/qdrant/fastembed).

This feature is **purely optional** and is never a core dependency.
Without it, Memory Etch works exactly as before — FTS5-first with HRR
and Jaccard re-ranking.

## Installation

```bash
pip install memory-etch[bge-m3]
```

This installs `fastembed>=0.5.0` which pulls ONNX Runtime (~150 MB on disk).
**No model is downloaded at install time.** The model is downloaded on the
first call to `BgeM3Plugin.encode()`.

> **System Requirements**: BGE-M3 requires ~2 GB of RAM at inference time.
> It runs entirely on CPU via ONNX Runtime — no GPU needed.

## Usage

### 1. Basic encoding

```python
from memory_etch.plugins.bge_m3 import BgeM3Plugin

plugin = BgeM3Plugin()

# First call triggers model download (cached thereafter)
vec = plugin.encode("Your text here")
assert len(vec) == 1024  # BGE-M3 dimension
```

### 2. Wire into EtchRetriever

Pass a `compute_embedding` callable to `EtchRetriever`. When set, search
results are fused via RRF (Reciprocal Rank Fusion) between the FTS5 and
vector streams.

```python
from memory_etch import EtchStore, EtchRetriever
from memory_etch.plugins.bge_m3 import BgeM3Plugin

store = EtchStore("memory.db")
plugin = BgeM3Plugin()

retriever = EtchRetriever(
    store=store,
    compute_embedding=plugin.encode,
)

# Results are now RRF-fused between FTS5 and vector similarity
results = retriever.search("machine learning with Python")
```

When `compute_embedding=None` (the default), search falls back to the
existing FTS5 + HRR + Jaccard pipeline with zero changes.

### 3. Add facts with pre-computed embeddings

Not needed for normal use — `compute_embedding` generates embeddings
automatically at search time. But you can store them explicitly:

```python
blob = struct.pack("1024f", *plugin.encode("fact content"))
fid = store.add_fact("fact content", embedding=blob)
```

### 4. Manual vector search

```python
query_blob = struct.pack("1024f", *plugin.encode("search query"))
results = store.search_by_vector(query_blob, limit=5)
```

## Backfill Existing Facts

If you already have facts in your database, run the backfill script to
compute and store their embeddings:

```bash
pip install memory-etch[bge-m3]
python scripts/backfill_embeddings.py path/to/memory.db --batch 32
```

This queries facts where `embedding IS NULL`, encodes them in batches,
and stores the BLOB vectors.

## How It Works

1. **FTS5**: Always runs first — fetches candidates with headroom (limit × 2)
2. **Optional vector**: If `compute_embedding` is set, the query is encoded
   and `search_by_vector()` does a cosine-similarity scan over stored
   embeddings (SQL pre-filtered by project/category/trust)
3. **RRF fusion**: Both streams are merged via Reciprocal Rank Fusion with
   k=60. Items appearing in both streams get a boosted score.

## Caching & Performance

- The fastembed model is cached in `~/.cache/huggingface/` after first
  download (~2.3 GB disk)
- Vector search uses **pure Python cosine similarity** over SQLite
  BLOBs (via `struct.unpack`). This is O(n * d) where n = candidate count
  and d = 1024. Acceptable for < 100K facts.
- For larger datasets, consider pre-filtering with project/category to
  reduce candidate count.

## Extras Reference

| Extra | Dependency | Purpose |
|-------|-----------|---------|
| `[hrr]` | numpy ≥ 1.24.0 | HRR vector similarity |
| `[bge-m3]` | fastembed ≥ 0.5.0 | BGE-M3 embedding search |
| `[embedding]` | fastembed ≥ 0.5.0 | Deprecated alias for [bge-m3] |
| `[all]` | hrr + bge-m3 | Everything |

## Security & Privacy

- All embeddings are stored as local SQLite BLOBs
- No external API calls — everything runs on your machine
- No automatic model download at import time (lazy load on first `encode()`)
