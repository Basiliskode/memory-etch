# EtchRetriever

```python
class EtchRetriever:
    """Hybrid search over an EtchStore — combines FTS5, HRR vectors,
    Jaccard similarity, and optional embedding vector search with RRF fusion."""

    def __init__(
        self,
        store: EtchStore,
        hrr_dim: int = 256,
        hrr_weight: float = 0.3,
        reranker: Callable | None = None,
        rerank_min_score: float = 0.0,
        compute_embedding: Callable[[str], list[float]] | None = None,
    ) -> None: ...
```

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `EtchStore` | — | An initialized EtchStore instance |
| `hrr_dim` | `int` | `256` | HRR vector dimension |
| `hrr_weight` | `float` | `0.3` | Blend weight for HRR vs FTS5 (0.0 = FTS5 only, 1.0 = HRR only) |
| `reranker` | `Callable \| None` | `None` | Optional callback `reranker(query, candidates) → ranked candidates` |
| `rerank_min_score` | `float` | `0.0` | Minimum top score to skip reranker |
| `compute_embedding` | `Callable \| None` | `None` | Optional `encode(text: str) → list[float]` for embedding search |

## Methods

### Primary Search

```python
def search(
    self,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    project: str = "",
) -> list[dict]: ...
```

Hybrid search combining FTS5 + HRR similarity + Jaccard re-rank. When `compute_embedding` is configured, results are fused via Reciprocal Rank Fusion (RRF) between FTS5 and vector streams. Returns results sorted by combined `score`.

Search strategy:
1. FTS5 candidate fetch (limit × 2 for headroom)
2. HRR phase vector similarity (if NumPy available)
3. Jaccard n-gram overlap for lexical re-ranking
4. Optional embedding vector search (if `compute_embedding` provided)
5. RRF fusion of FTS5 and vector streams

### Utility Searches

```python
def probe(self, topic: str, limit: int = 10, project: str = "") -> list[dict]: ...
```

Search by topic tag or content keyword. Matches facts where the tag or content contains *topic*.

---

```python
def related(self, topic: str, limit: int = 10) -> list[dict]: ...
```

Find facts related to a topic via entities + FTS5. First searches entity associations, then falls back to FTS5 content match.

---

```python
def contradict(self, limit: int = 10) -> list[dict]: ...
```

Find contradictions — first checks `fact_relations` for known contradictions, then falls back to heuristic scan of facts sharing the same category. Returns contradictory fact pairs.
