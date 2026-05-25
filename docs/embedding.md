# Embedding Vector Search (Optional)

Memory Etch supports **optional embedding vector search** using
[fastembed](https://github.com/qdrant/fastembed) or an Ollama embedding server.

This feature is **purely optional** and is never a core dependency.
Without it, Memory Etch works exactly as before — FTS5-first, with optional HRR
and Jaccard re-ranking when the `hrr` extra is installed.

## Installation

```bash
pip install "memory-etch[embeddings]"
```

This installs `fastembed>=0.5.0` and `httpx>=0.27`. `fastembed` pulls ONNX Runtime
(~150 MB on disk). **No model is downloaded at install time.** Models are
downloaded lazily on first use.

> **System Requirements**: BGE-M3 requires ~2 GB of RAM at inference time.
> It runs entirely on CPU via ONNX Runtime — no GPU needed.

## Usage

### 1. Basic provider usage

```python
from memory_etch import EtchStore
from memory_etch.embedding import FastembedProvider, OllamaProvider

store = EtchStore("memory.db", embedding_provider=FastembedProvider())

# Or use an existing Ollama server.
store = EtchStore(
    "memory.db",
    embedding_provider=OllamaProvider(model="nomic-embed-text"),
)
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
| `[embedding]` | fastembed ≥ 0.5.0, httpx ≥ 0.27 | Fastembed and Ollama providers |
| `[embeddings]` | fastembed ≥ 0.5.0, httpx ≥ 0.27, numpy ≥ 1.24.0 | Dense embeddings plus numeric helpers |
| `[bge-m3]` | fastembed ≥ 0.5.0 | Legacy BGE-M3 plugin support |
| `[mcp]` | mcp ≥ 1.0.0 | MCP stdio server |
| `[benchmark]` | google-genai ≥ 2.4.0 | Synthetic benchmark runner |
| `[openai]` | openai ≥ 1.0.0 | OpenAI interceptor support |
| `[anthropic]` | anthropic ≥ 0.30.0 | Anthropic interceptor support |
| `[dev]` | build, pytest, pytest-cov, ruff, twine | Local development and release checks |
| `[all]` | hrr + embeddings + mcp + openai + anthropic | Runtime feature bundle |

## Security & Privacy

- All embeddings are stored as local SQLite BLOBs
- No external API calls — everything runs on your machine
- No automatic model download at import time (lazy load on first `encode()`)
