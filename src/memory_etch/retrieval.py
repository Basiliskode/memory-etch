"""Hybrid retriever for Memory Etch — combines FTS5, HRR vectors, and Jaccard similarity.

Search strategy:
1. FTS5 candidate fetch (limit × 2 for scoring headroom)
2. HRR phase vector similarity (if numpy available)
3. Jaccard n-gram overlap for lexical re-ranking
4. RRF-style fusion of all scores
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
    """

    def __init__(
        self,
        store: EtchStore,
        hrr_dim: int = 256,
        hrr_weight: float = _DEFAULT_HRR_WEIGHT,
        reranker: Optional[Callable] = None,
        rerank_min_score: float = 0.0,
    ):
        self._store = store
        self._hrr_dim = hrr_dim
        self._hrr_weight = hrr_weight
        self._reranker = reranker
        self._rerank_min_score = rerank_min_score

    def search(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
    ) -> list[dict]:
        """Hybrid search: FTS5 + optional HRR + Jaccard.

        Returns list of dicts sorted by combined relevance score.
        """
        candidates = self._fts_candidates(query, limit, exclude_deleted)
        if not candidates:
            return []

        scored = self._score_candidates(query, candidates)
        scored.sort(key=lambda x: x["_score"], reverse=True)

        results = scored[:limit]
        for r in results:
            r.pop("_hrr_vec", None)

        # Optional reranker
        if self._reranker:
            try:
                top_score = results[0].get("_score", 0)
                if top_score < self._rerank_min_score or self._rerank_min_score <= 0:
                    reranked = self._reranker(query, results)
                    if reranked:
                        return reranked
            except Exception:
                logger.exception("Reranker failed, returning FTS5+HRR results")

        return results

    def _fts_candidates(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
    ) -> list[dict]:
        """Fetch candidates from FTS5 with headroom for re-scoring."""
        with self._store._lock:
            try:
                limit_mult = self._store._conn.execute(
                    "PRAGMA table_info(facts)"
                ).fetchall()
                # Use doubled limit for scoring headroom
                fetch_limit = limit * _DEFAULT_FTS5_LIMIT_MULTIPLIER

                sql = """SELECT f.fact_id, f.content, f.category, f.tags,
                                f.trust_score, f.hrr_vector, f.created_at, f.updated_at
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?
                         ORDER BY fts.rank
                         LIMIT ?"""
                params: list = [query, fetch_limit]
                if exclude_deleted:
                    # Inject deleted filter before ORDER BY
                    sql = sql.replace(
                        "ORDER BY fts.rank",
                        "AND (f.deleted IS NULL OR f.deleted = 0) ORDER BY fts.rank",
                    )
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
