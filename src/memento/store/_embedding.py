"""Embedding helpers — compute and search by embedding vectors.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
from typing import Optional

from ..embedding import NoopProvider

logger = logging.getLogger(__name__)


def _maybe_store_embedding(store, fact_id: int, content: str) -> None:
    """Compute and store an embedding for a fact if the provider is active.

    Skips if the provider is NoopProvider (not configured).
    If the provider raises, the fact simply has embedding=NULL.
    """
    if isinstance(store._embedding_provider, NoopProvider):
        return
    try:
        vec = store._embedding_provider.embed([content])
        if vec and vec[0]:
            import struct

            blob = struct.pack(f"{len(vec[0])}f", *vec[0])
            store._conn.execute(
                "UPDATE facts SET embedding=? WHERE fact_id=?",
                (blob, fact_id),
            )
            store._conn.commit()
    except Exception:
        logger.exception("Embedding computation failed for fact %d", fact_id)


def _search_by_embedding(
    store,
    query_emb: list[float],
    k: int,
    scope: str = "",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
) -> list[int]:
    """Search facts by embedding vector similarity (dot product).

    Loads stored BLOBs as float32 ndarray, L2-normalizes, computes
    dot product with query vector, returns top-k fact IDs.

    Supports optional scope and source filtering.

    Returns empty list if numpy is unavailable or no embeddings exist.
    """
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:
        return []

    with store._lock:
        sql = (
            "SELECT fact_id, embedding FROM facts "
            "WHERE embedding IS NOT NULL AND (deleted IS NULL OR deleted = 0)"
        )
        params: list = []
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if source_harness:
            sql += " AND source_harness = ?"
            params.append(source_harness)
        if source_agent:
            sql += " AND source_agent = ?"
            params.append(source_agent)
        if source_kind:
            sql += " AND source_kind = ?"
            params.append(source_kind)
        rows = store._conn.execute(sql, params).fetchall()

    if not rows:
        return []

    n_dim = len(query_emb)
    embs = []
    ids = []
    for r in rows:
        blob = r["embedding"]
        if not blob:
            continue
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) != n_dim:
                continue  # skip mismatched dimensions
            embs.append(vec)
            ids.append(r["fact_id"])
        except Exception:
            continue

    if not embs:
        return []

    # Stack into (N, dim) matrix
    matrix = np.stack(embs, axis=0)
    # L2-normalize (already normalized for fastembed, but be safe)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    # Query vector
    q = np.array(query_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # Dot product (cosine similarity for unit vectors)
    scores = matrix @ q

    # Sort by score descending, get top-k
    top_indices = np.argsort(scores)[::-1][:k]
    return [ids[i] for i in top_indices]
