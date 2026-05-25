"""Search operations — FTS5 search, metadata search, vector similarity search.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
import struct
from typing import Optional

from memory_etch.embedding import NoopProvider
from memory_etch.store._crud import _reinforce_facts
from memory_etch.store._embedding import _search_by_embedding
from memory_etch.store._schema import _sanitize_fts5

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal: FTS5-only search_facts (first definition in old store.py)
# ------------------------------------------------------------------


def _search_facts_fts5(
    store,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """Full-text search via FTS5.

    Default scope is ``'canonical'`` — only canonical facts are returned
    unless an explicit scope is requested.
    """
    with store._lock:
        try:
            sql = """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                            f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count,
                            f.importance, f.session_id,
                            f.source_harness, f.source_agent, f.source_kind, f.scope,
                            f.fact_type
                     FROM facts f
                     JOIN facts_fts fts ON fts.rowid = f.fact_id
                     WHERE facts_fts MATCH ?"""
            params: list = [query]
            conditions: list[str] = []
            if exclude_deleted:
                conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
            if scope:
                conditions.append("f.scope = ?")
                params.append(scope)
            if source_harness:
                conditions.append("f.source_harness = ?")
                params.append(source_harness)
            if source_agent:
                conditions.append("f.source_agent = ?")
                params.append(source_agent)
            if source_kind:
                conditions.append("f.source_kind = ?")
                params.append(source_kind)
            if fact_type:
                conditions.append("f.fact_type = ?")
                params.append(fact_type)
            if conditions:
                sql += " AND " + " AND ".join(conditions)
            sql += " ORDER BY fts.rank LIMIT ?"
            params.append(limit)
            rows = store._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ------------------------------------------------------------------
# Public: backward-compat search_facts (alias for search, second def)
# ------------------------------------------------------------------


def search_facts(
    store,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    project: str = "",
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """Alias for ``search``."""
    return store.search(query, limit=limit, exclude_deleted=exclude_deleted, project=project,
                       scope=scope, source_harness=source_harness, source_agent=source_agent,
                       source_kind=source_kind, fact_type=fact_type)


# ------------------------------------------------------------------
# search_by_metadata
# ------------------------------------------------------------------


def search_by_metadata(
    store,
    what: Optional[str] = None,
    why: Optional[str] = None,
    where_text: Optional[str] = None,
    learned: Optional[str] = None,
    limit: int = 10,
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """Search facts by structured metadata fields.

    Builds SQL WHERE clauses for each non-None field using
    ``LIKE '%value%'`` for partial matching. All non-None fields
    are combined with AND.

    Args:
        what: Filter by ``what`` field (partial match).
        why: Filter by ``why`` field (partial match).
        where_text: Filter by ``where_text`` field (partial match).
        learned: Filter by ``learned`` field (partial match).
        limit: Max results (default: 10).
        scope: Scope filter (default: ``'canonical'``).
        source_harness: Optional source harness filter.
        source_agent: Optional source agent filter.
        source_kind: Optional source kind filter.

    Returns:
        List of fact dicts matching all provided filters.
    """
    conditions: list[str] = ["(f.deleted IS NULL OR f.deleted = 0)"]
    params: list = []

    field_map = {
        "what": what,
        "why": why,
        "where_text": where_text,
        "learned": learned,
    }

    for col, val in field_map.items():
        if val is not None:
            conditions.append(f"f.{col} LIKE ?")
            params.append(f"%{val}%")

    if scope:
        conditions.append("f.scope = ?")
        params.append(scope)
    if source_harness:
        conditions.append("f.source_harness = ?")
        params.append(source_harness)
    if source_agent:
        conditions.append("f.source_agent = ?")
        params.append(source_agent)
    if source_kind:
        conditions.append("f.source_kind = ?")
        params.append(source_kind)
    if fact_type:
        conditions.append("f.fact_type = ?")
        params.append(fact_type)

    with store._lock:
        w = " AND ".join(conditions)
        rows = store._conn.execute(
            f"SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score, "
            f"f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count, "
            f"f.importance, f.session_id, f.what, f.why, f.where_text, f.learned, "
            f"f.source_harness, f.source_agent, f.source_kind, f.scope, f.fact_type "
            f"FROM facts f WHERE {w} ORDER BY f.trust_score DESC LIMIT ?",
            params + [limit],
        ).fetchall()

    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# search_by_vector
# ------------------------------------------------------------------


def search_by_vector(
    store,
    query_vector: bytes,
    limit: int = 10,
    min_trust: float = 0.0,
    category: str = "",
    project: str = "",
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
) -> list[dict]:
    """Search facts by embedding vector (cosine similarity).

    SQL pre-filter narrows candidates; Python ``struct.unpack`` decodes
    float32 arrays and computes cosine similarity in a loop.

    Retrieved facts get a small trust boost (retrieval feedback loop).

    Args:
        scope: Scope filter (default: ``'canonical'``).
        source_harness: Optional source harness filter.
        source_agent: Optional source agent filter.
        source_kind: Optional source kind filter.

    Returns list of fact dicts sorted by cosine similarity descending.
    """
    with store._lock:
        conditions = ["(f.deleted IS NULL OR f.deleted = 0)", "f.embedding IS NOT NULL"]
        params: list = []
        if min_trust > 0:
            conditions.append("f.trust_score >= ?")
            params.append(min_trust)
        if category:
            conditions.append("f.category = ?")
            params.append(category)
        if project:
            conditions.append("f.project = ?")
            params.append(project)
        if scope:
            conditions.append("f.scope = ?")
            params.append(scope)
        if source_harness:
            conditions.append("f.source_harness = ?")
            params.append(source_harness)
        if source_agent:
            conditions.append("f.source_agent = ?")
            params.append(source_agent)
        if source_kind:
            conditions.append("f.source_kind = ?")
            params.append(source_kind)

        w = " AND ".join(conditions)
        rows = store._conn.execute(
            f"SELECT fact_id, content, embedding, trust_score, category, project "
            f"FROM facts f WHERE {w}",
            params,
        ).fetchall()

    n_floats = len(query_vector) // 4
    try:
        q = struct.unpack(f"{n_floats}f", query_vector)
    except struct.error:
        return []

    norm_q = sum(a * a for a in q) ** 0.5
    if norm_q == 0:
        return []

    scored: list[tuple[float, dict]] = []
    for r in rows:
        blob = r["embedding"]
        if not blob:
            continue
        try:
            v = struct.unpack(f"{len(blob) // 4}f", blob)
        except struct.error:
            continue
        dot = sum(a * b for a, b in zip(q, v))
        norm_v = sum(b * b for b in v) ** 0.5
        sim = dot / (norm_q * norm_v) if norm_v > 0 else 0.0
        d = dict(r)
        d.pop("embedding", None)
        scored.append((sim, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [d for _, d in scored[:limit]]
    # Reinforce retrieved facts
    with store._lock:
        _reinforce_facts(store, [r["fact_id"] for r in results])
    return results


# ------------------------------------------------------------------
# _rrf_merge — Reciprocal Rank Fusion
# ------------------------------------------------------------------


def _rrf_merge(
    stream_a: list[dict],
    stream_b: list[dict],
    limit: int,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked streams.

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


# ------------------------------------------------------------------
# search — Hybrid FTS5 + Vector search with RRF fusion
# ------------------------------------------------------------------


def search(
    store,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    project: str = "",
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """Hybrid search: FTS5 + optional embedding vector search fused via RRF.

    Always returns FTS5 results. If an embedding provider is configured
    (non-NoopProvider), the query is embedded and vector search results
    are fused via Reciprocal Rank Fusion.

    Default scope is ``'canonical'`` — only canonical facts are returned
    unless an explicit scope is requested.

    Args:
        query: Search text.
        limit: Max results.
        exclude_deleted: Whether to exclude soft-deleted facts.
        project: Optional project filter.
        scope: Scope filter (default: ``'canonical'``). Pass ``'inbox'``,
            ``'personal'``, or ``'ephemeral'`` for other scopes.
        source_harness: Optional source harness filter.
        source_agent: Optional source agent filter.
        source_kind: Optional source kind filter.

    Returns list of dicts sorted by combined relevance score (``score`` key).
    """
    with store._lock:
        try:
            safe_query = _sanitize_fts5(query)
            sql = """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                            f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count,
                            f.importance, f.session_id,
                            f.source_harness, f.source_agent, f.source_kind, f.scope,
                            f.fact_type
                     FROM facts f
                     JOIN facts_fts fts ON fts.rowid = f.fact_id
                     WHERE facts_fts MATCH ?"""
            params: list = [safe_query]
            conditions: list[str] = []
            if exclude_deleted:
                conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
            if project:
                conditions.append("f.project = ?")
                params.append(project)
            if scope:
                conditions.append("f.scope = ?")
                params.append(scope)
            if source_harness:
                conditions.append("f.source_harness = ?")
                params.append(source_harness)
            if source_agent:
                conditions.append("f.source_agent = ?")
                params.append(source_agent)
            if source_kind:
                conditions.append("f.source_kind = ?")
                params.append(source_kind)
            if fact_type:
                conditions.append("f.fact_type = ?")
                params.append(fact_type)
            if conditions:
                sql += " AND " + " AND ".join(conditions)
            sql += " ORDER BY fts.rank LIMIT ?"
            params.append(limit * 2)  # fetch extra for RRF headroom
            rows = store._conn.execute(sql, params).fetchall()
            fts_results = [dict(r) for r in rows]
        except Exception:
            fts_results = []

        # Vector stream via embedding (optional)
        emb_results: list[dict] = []
        if not isinstance(store._embedding_provider, NoopProvider):
            try:
                q_vec = store._embedding_provider.embed_query(query)
                top_ids = _search_by_embedding(store, q_vec, k=limit * 2, scope=scope,
                                               source_harness=source_harness,
                                               source_agent=source_agent,
                                               source_kind=source_kind)
                if top_ids:
                    placeholders = ",".join("?" for _ in top_ids)
                    with store._lock:
                        rows = store._conn.execute(
                            f"""SELECT fact_id, content, category, tags,
                                       trust_score, created_at, updated_at,
                                       project, topic_key, revision_count,
                                       importance, session_id
                                FROM facts
                                WHERE fact_id IN ({placeholders})
                                AND (deleted IS NULL OR deleted = 0)""",
                            top_ids,
                        ).fetchall()
                        # Re-sort by the embedding rank order
                        id_order = {fid: i for i, fid in enumerate(top_ids)}
                        sorted_rows = sorted(
                            rows, key=lambda r: id_order.get(r["fact_id"], 999)
                        )
                        emb_results = [dict(r) for r in sorted_rows]
            except Exception:
                logger.exception("Embedding search failed")

        # RRF merge
        merged = _rrf_merge(fts_results, emb_results, limit=limit, k=60)

        # Progressive disclosure: add summary (first 200 chars) to each result
        for item in merged:
            if "content" in item and "summary" not in item:
                item["summary"] = item["content"][:200]

        # Retrieval feedback loop — reinforce returned facts
        if merged:
            _reinforce_facts(store, [r["fact_id"] for r in merged])

        # Track last_retrieved_at for eviction
        if merged:
            fids = [r["fact_id"] for r in merged]
            placeholders = ",".join("?" for _ in fids)
            store._conn.execute(
                f"UPDATE facts SET last_retrieved_at = CURRENT_TIMESTAMP "
                f"WHERE fact_id IN ({placeholders})",
                fids,
            )
            store._conn.commit()

        return merged


# ------------------------------------------------------------------
# query — Structured query DSL with 20+ filter keys
# ------------------------------------------------------------------


def query(store, query_dict: dict) -> dict:
    """Run a structured query with multiple filters combined.

    Builds a dynamic SQL query from the provided *query_dict* keys.
    All keys are optional — an empty dict returns all non-deleted facts
    (up to the default limit of 50).

    Args:
        query_dict: Dict with optional keys:

            - **search** (*str*): FTS5 full-text search.
            - **fact_type** (*str*): Registered fact type.
            - **project** (*str*): Project / workspace name.
            - **category** (*str*): Fact category.
            - **scope** (*str*): Hive Memory scope (canonical, inbox,
              personal, ephemeral).
            - **tags** (*str*): Tags substring match (LIKE).
            - **min_trust** / **max_trust** (*float*): Numeric range.
            - **min_importance** / **max_importance** (*float*): Range.
            - **has_what** / **has_why** / **has_where** / **has_learned**
              (*bool*): Filter by non-empty structured fields.
            - **created_after** / **created_before** (*str*): ISO datetime.
            - **related_to** (*int*): Fact ID — finds connected facts.
            - **relation_type** (*str*): Relation type when *related_to*
              is used (e.g. ``"compatible"``, ``"derived_from"``).
            - **relation_direction** (*str*): ``"any"`` (default),
              ``"outgoing"``, or ``"incoming"``.
            - **order_by** (*str*): ``"created_at"`` (default),
              ``"trust_score"``, ``"importance"``, ``"updated_at"``.
            - **order_dir** (*str*): ``"desc"`` (default) or ``"asc"``.
            - **limit** (*int*): Max results (default 50, max 200).
            - **offset** (*int*): Pagination offset (default 0).

    Returns:
        Dict with keys:

        - **results**: List of matching fact dicts.
        - **total**: Total count *without* limit/offset applied.
        - **query**: Human-readable query summary string.
    """
    # ---- Extract all params with defaults ----
    search = query_dict.get("search")
    fact_type = query_dict.get("fact_type")
    project = query_dict.get("project")
    category = query_dict.get("category")
    scope = query_dict.get("scope")
    tags = query_dict.get("tags")
    min_trust = query_dict.get("min_trust")
    max_trust = query_dict.get("max_trust")
    min_importance = query_dict.get("min_importance")
    max_importance = query_dict.get("max_importance")
    has_what = query_dict.get("has_what")
    has_why = query_dict.get("has_why")
    has_where = query_dict.get("has_where")
    has_learned = query_dict.get("has_learned")
    created_after = query_dict.get("created_after")
    created_before = query_dict.get("created_before")
    related_to = query_dict.get("related_to")
    relation_type = query_dict.get("relation_type")
    relation_direction = query_dict.get("relation_direction", "any")
    order_by = query_dict.get("order_by", "created_at")
    order_dir = query_dict.get("order_dir", "desc")
    limit = query_dict.get("limit", 50)
    offset = query_dict.get("offset", 0)

    # Clamp limit
    if limit < 0:
        limit = 0
    elif limit > 200:
        limit = 200

    # Validate sort
    valid_order_cols = {"created_at", "trust_score", "importance", "updated_at"}
    if order_by not in valid_order_cols:
        order_by = "created_at"
    if order_dir not in ("asc", "desc"):
        order_dir = "desc"

    # ---- Build query summary string ----
    qs_parts: list[str] = []
    for k, v in [
        ("search", search),
        ("fact_type", fact_type),
        ("project", project),
        ("category", category),
        ("scope", scope),
        ("tags", tags),
        ("min_trust", min_trust),
        ("max_trust", max_trust),
        ("min_importance", min_importance),
        ("max_importance", max_importance),
        ("has_what", has_what if has_what else None),
        ("has_why", has_why if has_why else None),
        ("has_where", has_where if has_where else None),
        ("has_learned", has_learned if has_learned else None),
        ("created_after", created_after),
        ("created_before", created_before),
        ("related_to", related_to),
        ("relation_type", relation_type),
        ("relation_direction", relation_direction if relation_direction != "any" else None),
    ]:
        if v is not None and v != "":
            qs_parts.append(f"{k}={v}")
    query_summary = "&".join(qs_parts)

    # ---- SQL assembly ----
    columns = (
        "f.fact_id, f.content, f.category, f.tags, f.trust_score,"
        " f.importance, f.project, f.scope, f.fact_type,"
        " f.created_at, f.updated_at, f.what, f.why, f.where_text, f.learned"
    )
    conditions: list[str] = [
        "(f.deleted IS NULL OR f.deleted = 0)",
    ]
    params: dict[str, object] = {}
    from_clause = "FROM facts f"

    # FTS5 search
    if search is not None:
        sanitized = _sanitize_fts5(search)
        if sanitized:
            conditions.append(
                "EXISTS ("
                "SELECT 1 FROM facts_fts fts "
                "WHERE fts.rowid = f.fact_id AND facts_fts MATCH :search"
                ")"
            )
            params["search"] = sanitized

    # Equality filters
    if fact_type:
        conditions.append("f.fact_type = :fact_type")
        params["fact_type"] = fact_type
    if project:
        conditions.append("f.project = :project")
        params["project"] = project
    if category:
        conditions.append("f.category = :category")
        params["category"] = category
    if scope:
        conditions.append("f.scope = :scope")
        params["scope"] = scope

    # Tags (LIKE substring match)
    if tags:
        conditions.append("f.tags LIKE :tags")
        params["tags"] = f"%{tags}%"

    # Numeric range
    if min_trust is not None:
        conditions.append("f.trust_score >= :min_trust")
        params["min_trust"] = min_trust
    if max_trust is not None:
        conditions.append("f.trust_score <= :max_trust")
        params["max_trust"] = max_trust
    if min_importance is not None:
        conditions.append("f.importance >= :min_importance")
        params["min_importance"] = min_importance
    if max_importance is not None:
        conditions.append("f.importance <= :max_importance")
        params["max_importance"] = max_importance

    # Structured field existence
    if has_what:
        conditions.append("(f.what IS NOT NULL AND f.what != '')")
    if has_why:
        conditions.append("(f.why IS NOT NULL AND f.why != '')")
    if has_where:
        conditions.append("(f.where_text IS NOT NULL AND f.where_text != '')")
    if has_learned:
        conditions.append("(f.learned IS NOT NULL AND f.learned != '')")

    # Time range
    if created_after:
        conditions.append("f.created_at >= :created_after")
        params["created_after"] = created_after
    if created_before:
        conditions.append("f.created_at <= :created_before")
        params["created_before"] = created_before

    # Relation join
    if related_to is not None:
        from_clause += (
            " JOIN fact_relations fr"
            " ON (f.fact_id = fr.fact_id_a OR f.fact_id = fr.fact_id_b)"
            " AND (fr.fact_id_a = :related_to OR fr.fact_id_b = :related_to)"
            " AND f.fact_id != :related_to"
        )
        params["related_to"] = related_to
        if relation_type:
            conditions.append("fr.relation_type = :relation_type")
            params["relation_type"] = relation_type
        if relation_direction == "outgoing":
            conditions.append("fr.fact_id_a = :related_to")
        elif relation_direction == "incoming":
            conditions.append("fr.fact_id_b = :related_to")

    where_clause = " AND ".join(conditions)

    with store._lock:
        # Count (no limit/offset)
        count_sql = f"SELECT COUNT(*) AS cnt {from_clause} WHERE {where_clause}"
        total_row = store._conn.execute(count_sql, params).fetchone()
        total = total_row["cnt"] if total_row else 0

        # Results
        sql = (
            f"SELECT {columns} {from_clause}"
            f" WHERE {where_clause}"
            f" ORDER BY f.{order_by} {order_dir}"
            f" LIMIT :limit OFFSET :offset"
        )
        result_params = dict(params)
        result_params["limit"] = limit
        result_params["offset"] = offset
        rows = store._conn.execute(sql, result_params).fetchall()
        results = [dict(r) for r in rows]

    return {
        "results": results,
        "total": total,
        "query": query_summary,
    }
