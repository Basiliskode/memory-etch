"""Hybrid retriever for Memory Etch — combines FTS5, HRR vectors, Jaccard similarity,
and optional embedding vector search with RRF fusion.

Search strategy:
1. FTS5 candidate fetch (limit × 2 for scoring headroom)
2. HRR phase vector similarity (if numpy available)
3. Jaccard n-gram overlap for lexical re-ranking
4. Optional embedding vector search (if ``compute_embedding`` provided)
5. RRF (Reciprocal Rank Fusion) of FTS5 and vector streams
"""

import logging
import re
from typing import Any, Optional, Callable

from . import hrr
from .store import EtchStore

logger = logging.getLogger(__name__)

# Default HRR weight — 0.3 keeps HRR as a tiebreaker over FTS5
_DEFAULT_HRR_WEIGHT = 0.3
_DEFAULT_FTS5_LIMIT_MULTIPLIER = 2


class EtchRetriever:
    """Hybrid search over an EtchStore.

    Args:
        store: EtchStore instance.
        hrr_dim: HRR vector dimension (default: 256).
        hrr_weight: Blend weight for HRR vs FTS5 (0.0 = FTS5 only, 1.0 = HRR only).
        reranker: Optional callback reranker(query, candidates) → ranked candidates.
        rerank_min_score: Minimum top score to skip reranker (0.0 = always rerank).
        compute_embedding: Optional callable ``encode(text: str) → list[float]``.
            When ``None``, vector search path is skipped gracefully.
    """

    def __init__(
        self,
        store: EtchStore,
        hrr_dim: int = 256,
        hrr_weight: float = _DEFAULT_HRR_WEIGHT,
        reranker: Optional[Callable] = None,
        rerank_min_score: float = 0.0,
        compute_embedding: Optional[Callable[[str], list[float]]] = None,
    ):
        self._store = store
        self._hrr_dim = hrr_dim
        self._hrr_weight = hrr_weight
        self._reranker = reranker
        self._rerank_min_score = rerank_min_score
        self._compute_embedding = compute_embedding

    def search(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
    ) -> list[dict]:
        """Hybrid search: FTS5 + optional HRR + Jaccard + optional vector.

        When ``compute_embedding`` is configured, results are fused via
        ``_rrf_merge`` between the FTS5 and vector streams.

        Args:
            query: Search text.
            limit: Max results.
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.

        Returns list of dicts sorted by combined relevance score (``score`` key).
        """
        fts5_stream = self._fts_candidates(query, limit * 2, exclude_deleted, project)
        if not fts5_stream:
            return []

        # Scored FTS5 candidates
        scored = self._score_candidates(query, fts5_stream)
        scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
        for r in scored:
            r.pop("_hrr_vec", None)

        # Vector stream (optional)
        vector_stream: list[dict] = []
        if self._compute_embedding is not None:
            try:
                q_vec = self._compute_embedding(query)
                if q_vec:
                    import struct
                    vec_bytes = struct.pack(f"{len(q_vec)}f", *q_vec)
                    vector_stream = self._store.search_by_vector(
                        vec_bytes, limit=limit * 2, project=project,
                    )
            except Exception:
                logger.exception("Vector search failed, falling back to FTS5-only")

        # RRF fusion
        merged = self._rrf_merge(scored, vector_stream, limit=limit, k=60)

        # Apply reranker if configured
        if self._reranker and merged:
            try:
                top_score = merged[0].get("score", 0)
                if top_score < self._rerank_min_score or self._rerank_min_score <= 0:
                    reranked = self._reranker(query, merged)
                    if reranked:
                        return reranked
            except Exception:
                logger.exception("Reranker failed, returning RRF results")

        return merged

    def _fts_candidates(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
    ) -> list[dict]:
        """Fetch candidates from FTS5 with headroom for re-scoring.

        Optionally filters by project.
        """
        with self._store._lock:
            try:
                fetch_limit = limit * _DEFAULT_FTS5_LIMIT_MULTIPLIER

                sql = """SELECT f.fact_id, f.content, f.category, f.tags,
                                f.trust_score, f.hrr_vector, f.created_at, f.updated_at,
                                f.project
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?"""
                params: list = [query]
                conditions: list[str] = []
                if exclude_deleted:
                    conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
                if project:
                    conditions.append("f.project = ?")
                    params.append(project)
                if conditions:
                    sql += " AND " + " AND ".join(conditions)
                sql += " ORDER BY fts.rank LIMIT ?"
                params.append(fetch_limit)

                rows = self._store._conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                logger.exception("FTS5 search failed")
                return []

    def _score_candidates(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Score candidates with hybrid FTS5 + HRR + Jaccard."""
        if not candidates:
            return []

        # 1. Compute HRR query vector (if available)
        query_vec = None
        if hrr.HAS_NUMPY and self._hrr_weight > 0:
            try:
                query_vec = hrr.encode_text(query, self._hrr_dim)
            except Exception:
                logger.exception("HRR query encoding failed")

        # 2. Score each candidate
        for c in candidates:
            score = c.get("trust_score", 0.5)  # base trust

            # FTS5 rank contribution (normalize to 0-1)
            fts_rank = getattr(c, "rank", 0) if hasattr(c, "rank") else 0.5
            score += fts_rank * 0.3

            # Jaccard n-gram overlap
            jac = self._jaccard_ngram(query, c.get("content", ""))
            score += jac * 0.2

            # HRR similarity
            hrr_sim = 0.0
            if query_vec is not None and hrr.HAS_NUMPY:
                try:
                    blob = c.get("hrr_vector")
                    if blob:
                        existing_vec = self._store._get_hrr_cached(c["fact_id"])
                        if existing_vec is not None:
                            hrr_sim = max(0, hrr.similarity(query_vec, existing_vec))
                        else:
                            vec = hrr.bytes_to_phases(blob)
                            hrr_sim = max(0, hrr.similarity(query_vec, vec))
                except Exception:
                    pass

            score += hrr_sim * self._hrr_weight

            c["_score"] = score
            c["_hrr_sim"] = hrr_sim

        return candidates

    @staticmethod
    def _jaccard_ngram(a: str, b: str, n: int = 3) -> float:
        """Token-level Jaccard similarity on n-grams."""
        if not a or not b:
            return 0.0
        a_tokens = set(re.findall(r"\w+", a.lower()))
        b_tokens = set(re.findall(r"\w+", b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    @staticmethod
    def _rrf_merge(
        stream_a: list[dict],
        stream_b: list[dict],
        limit: int,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two ranked streams.

        Items appearing in both streams get a boosted rank.
        When one stream is empty, the other is returned with RRF scores applied.

        Args:
            stream_a: First ranked list (must have ``fact_id`` key).
            stream_b: Second ranked list.
            limit: Max items to return.
            k: RRF constant (default 60).

        Returns:
            List of merged dicts with a ``score`` key.
        """
        if not stream_a and not stream_b:
            return []
        if not stream_b:
            # Single source — assign RRF scores
            result = []
            for rank, item in enumerate(stream_a):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]
        if not stream_a:
            result = []
            for rank, item in enumerate(stream_b):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]

        scores: dict[int, float] = {}
        items: dict[int, dict] = {}

        for rank, item in enumerate(stream_a):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        for rank, item in enumerate(stream_b):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        ranked = sorted(scores.keys(), key=lambda fid: scores[fid], reverse=True)
        result = []
        for fid in ranked[:limit]:
            d = dict(items[fid])
            d["score"] = scores[fid]
            result.append(d)
        return result

    def probe(self, topic: str, limit: int = 10, project: str = "") -> list[dict]:
        """Search by topic tag or content keyword.

        Matches facts where the tag or content contains *topic*.
        """
        with self._store._lock:
            conditions: list[str] = ["(f.deleted IS NULL OR f.deleted = 0)"]
            params: list = []
            if project:
                conditions.append("f.project = ?")
                params.append(project)
            conditions.append("(f.tags LIKE ? OR f.content LIKE ?)")
            params.extend([f"%{topic}%", f"%{topic}%"])
            w = " AND ".join(conditions)
            rows = self._store._conn.execute(
                f"SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score, "
                f"f.project, f.created_at, f.session_id "
                f"FROM facts f WHERE {w} "
                f"ORDER BY f.trust_score DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        results = [dict(r) for r in rows]
        for r in results:
            r["_score"] = r.get("trust_score", 0.5)
        return results

    def contradict(self, limit: int = 10) -> list[dict]:
        """Find contradictions — known (fact_relations) then algorithmic.

        Returns up to *limit* contradictory fact pairs.
        """
        # 1. Known contradictions from fact_relations
        known = self._store.get_contradictions(limit)
        for r in known:
            r["source"] = "fact_relations"
        if known:
            return known[:limit]

        # 2. Algorithmic fallback: scan for contradictory content in same project
        with self._store._lock:
            rows = self._store._conn.execute(
                """SELECT f1.fact_id AS fact_id_a, f1.content AS content_a,
                          f2.fact_id AS fact_id_b, f2.content AS content_b,
                          0.5 AS confidence
                   FROM facts f1
                   JOIN facts f2 ON f1.project = f2.project AND f1.fact_id < f2.fact_id
                   WHERE (f1.deleted IS NULL OR f1.deleted = 0)
                     AND (f2.deleted IS NULL OR f2.deleted = 0)
                     AND f1.category = f2.category
                   ORDER BY RANDOM()
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
