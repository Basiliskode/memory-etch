"""Core CRUD operations — get, list, update, purge, evict, entities, reinforce.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import hashlib
import logging
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from memento import hrr
from memento.store._schema import VALID_SCOPES, _sanitize_fts5

logger = logging.getLogger(__name__)


def get_fact(store, fact_id: int) -> Optional[dict]:
    """Get a single fact by ID.

    Args:
        fact_id: ID of the fact to retrieve.

    Returns:
        Fact dict (excluding ``hrr_vector`` and ``embedding`` blobs),
        or None if not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM facts WHERE fact_id=?", (fact_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d.pop("hrr_vector", None)
    d.pop("embedding", None)
    return d


def get_fact_full(store, fact_id: int) -> Optional[dict]:
    """Alias for ``get_fact`` — returns full fact content with all fields.

    Args:
        fact_id: ID of the fact to retrieve.

    Returns:
        Full fact dict (excluding large blobs), or None if not found.
    """
    return get_fact(store, fact_id)


def list_facts(
    store,
    category: str = "",
    project: str = "",
    limit: int = 50,
    offset: int = 0,
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """List facts with optional filters. Returns a flat list of fact dicts.

    Args:
        scope: Scope filter (default: ``'canonical'``).
        source_harness: Optional source harness filter.
        source_agent: Optional source agent filter.
        source_kind: Optional source kind filter.
    """
    with store._lock:
        where = ["(deleted IS NULL OR deleted = 0)"]
        params: list = []
        if category:
            where.append("category = ?")
            params.append(category)
        if project:
            where.append("project = ?")
            params.append(project)
        if scope:
            where.append("scope = ?")
            params.append(scope)
        if source_harness:
            where.append("source_harness = ?")
            params.append(source_harness)
        if source_agent:
            where.append("source_agent = ?")
            params.append(source_agent)
        if source_kind:
            where.append("source_kind = ?")
            params.append(source_kind)
        if fact_type:
            where.append("fact_type = ?")
            params.append(fact_type)
        w = " AND ".join(where)

        rows = store._conn.execute(
            f"SELECT fact_id, content, category, tags, trust_score, project, "
            f"created_at, updated_at, topic_key, revision_count, importance, session_id, "
            f"source_harness, source_agent, source_kind, scope, fact_type "
            f"FROM facts WHERE {w} ORDER BY trust_score DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

    return [dict(r) for r in rows]


def update_fact(store, fact_id: int, **kwargs) -> bool:
    """Update fact fields.

    Allowed keys: ``content``, ``category``, ``tags``, ``trust_score``,
    ``importance``, ``project``.

    Args:
        fact_id: ID of the fact to update.
        **kwargs: Field-value pairs to update.

    Returns:
        True if the fact was updated, False if no valid fields given.
    """
    allowed = {"content", "category", "tags", "trust_score", "importance", "project", "fact_type"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    # Capture fields being changed (excluding auto-set updated_at)
    changed_fields = list(updates.keys())
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [fact_id]
    with store._lock:
        # Capture old values before update
        old_row = store._conn.execute(
            f"SELECT {', '.join(changed_fields)} FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        old_values = dict(zip(changed_fields, old_row)) if old_row else {}
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        store._conn.execute(f"UPDATE facts SET {set_clause} WHERE fact_id=?", vals)
        store._conn.commit()
        store._log_event("fact_updated", fact_id=fact_id, project=fact_project,
                         metadata={"fields": old_values})
        store._conn.commit()  # close implicit transaction opened by _log_event
        store._invalidate_hrr_cache(fact_id)
    return True


def purge_facts(store, dry_run: bool = True) -> dict:
    """Purge low-value facts: >90d old, low trust <0.3, low importance <0.5.

    Returns stats about what would be / was deleted.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    with store._lock:
        candidates = store._conn.execute(
            """SELECT fact_id, content, trust_score, importance, created_at FROM facts
               WHERE (deleted IS NULL OR deleted = 0)
                 AND created_at < ?
                 AND trust_score < 0.3
                 AND importance < 0.5""",
            (cutoff,),
        ).fetchall()

        if dry_run:
            return {
                "action": "dry_run",
                "candidates": len(candidates),
                "detail": [dict(r) for r in candidates[:10]],
            }

        count = 0
        for row in candidates:
            store.soft_delete_fact(row["fact_id"], reason="auto_purge")
            count += 1
        store._log_event("facts_purged", metadata={"count": count})
        store._conn.commit()
        return {"action": "purged", "count": count}


def evict_stale(
    store,
    min_trust: float = 0.1,
    max_days: int = 30,
) -> int:
    """Soft-delete stale facts with low trust and old retrieval age.

    Evicts facts where:
    - ``trust_score < min_trust`` AND
    - ``last_retrieved_at`` is more than ``max_days`` ago AND
    - fact is not already deleted

    Also evicts facts that were never retrieved (``last_retrieved_at IS NULL``)
    if created more than 7 days ago.

    Args:
        min_trust: Minimum trust score threshold (default: 0.1).
        max_days: Maximum days since last retrieval (default: 30).

    Returns:
        Number of facts soft-deleted.
    """
    with store._lock:
        # Condition 1: retrieved facts that are stale
        cursor1 = store._conn.execute(
            """UPDATE facts SET
                   deleted = 1,
                   deleted_reason = 'eviction: trust=' || ROUND(trust_score, 3)
                       || ' last_retrieved=' || COALESCE(last_retrieved_at, 'never'),
                   updated_at = CURRENT_TIMESTAMP
               WHERE (deleted IS NULL OR deleted = 0)
                 AND trust_score < ?
                 AND last_retrieved_at IS NOT NULL
                 AND julianday('now') - julianday(last_retrieved_at) > ?""",
            (min_trust, max_days),
        )
        count1 = cursor1.rowcount

        # Condition 2: never-retrieved facts older than 7 days
        cursor2 = store._conn.execute(
            """UPDATE facts SET
                   deleted = 1,
                   deleted_reason = 'eviction: never retrieved, trust='
                       || ROUND(trust_score, 3),
                   updated_at = CURRENT_TIMESTAMP
               WHERE (deleted IS NULL OR deleted = 0)
                 AND trust_score < ?
                 AND last_retrieved_at IS NULL
                 AND created_at < datetime('now', '-7 days')""",
            (min_trust,),
        )
        count2 = cursor2.rowcount

        store._conn.commit()

    total = count1 + count2
    if total:
        logger.info("Evicted %d stale facts (retrieved=%d, never_retrieved=%d)",
                    total, count1, count2)
    return total


# ------------------------------------------------------------------
# Entities
# ------------------------------------------------------------------


def _ensure_entity(store, fact_id: int, name: str, entity_type: str = "unknown") -> int:
    """Upsert an entity and create N:M link to fact. Called inside store._lock."""
    with store._lock:
        store._conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
            (name.lower(), entity_type),
        )
        row = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ?", (name.lower(),)
        ).fetchone()
        if row:
            eid = row["entity_id"]
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fact_id, eid),
            )
            return eid
    return 0


def get_entities(store, fact_id: int) -> list[dict]:
    """Get entities associated with a fact.

    Args:
        fact_id: ID of the fact.

    Returns:
        List of entity dicts with keys ``entity_id``, ``name``, ``entity_type``.
    """
    with store._lock:
        rows = store._conn.execute(
            """SELECT e.entity_id, e.name, e.entity_type
               FROM entities e
               JOIN fact_entities fe ON fe.entity_id = e.entity_id
               WHERE fe.fact_id = ?""",
            (fact_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _reinforce_facts(store, fact_ids: list[int]) -> None:
    """Boost trust_score and increment retrieval_count for retrieved facts.

    Each retrieval gives a small trust boost (0.01), capping at 1.0.
    Called internally after search to implement the retrieval feedback loop.
    """
    if not fact_ids:
        return
    placeholders = ",".join("?" for _ in fact_ids)
    store._conn.execute(
        f"""UPDATE facts SET
                retrieval_count = retrieval_count + 1,
                trust_score = MIN(1.0, ROUND(trust_score + 0.01, 4))
            WHERE fact_id IN ({placeholders})""",
        fact_ids,
    )
    store._conn.commit()


# ------------------------------------------------------------------
# Phase 3b: add_fact, conflict detection, consolidation, soft-delete
# ------------------------------------------------------------------


def add_fact(
    store,
    content: str,
    category: str = "general",
    tags: str = "",
    trust_score: Optional[float] = None,
    importance: Optional[float] = None,
    project: str = "",
    session_id: str = "",
    topic_key: str = "",
    entities: Optional[list[str]] = None,
    embedding: Optional[bytes] = None,
    what: Optional[str] = None,
    why: Optional[str] = None,
    where_text: Optional[str] = None,
    learned: Optional[str] = None,
    return_metadata: bool = False,
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
    scope: str = "canonical",
) -> int | dict:
    """Insert a new fact.

    When tags contain ``topic:<name>``, the topic_key is auto-extracted
    and an existing fact with the same key is UPDATEd (topic upsert).

    Content hash dedup: if the same ``content + project`` exists (lifetime
    dedup), ``duplicate_count`` is incremented and the existing ``fact_id``
    is returned (no new row created).

    Args:
        content: Fact text content.
        category: Fact category (e.g. "general", "project", "user_pref").
        tags: Comma-separated tags. ``topic:<name>`` triggers topic upsert.
        trust_score: Initial trust score (default: 0.5).
        importance: Fact importance (default: 0.5).
        project: Optional project namespace.
        session_id: Optional session identifier.
        topic_key: Optional topic key for upsert behavior.
        entities: Optional list of entity names to associate.
        embedding: Optional pre-computed embedding bytes.
        what: Optional structured "what" field.
        why: Optional structured "why" field.
        where_text: Optional structured "where" field (``where`` is a
            SQL reserved word, so we use ``where_text``).
        learned: Optional structured "learned" field.
        return_metadata: If True, returns a dict with ``id``, ``status``,
            and optional ``conflicts_with``. If False (default), returns
            the ``fact_id`` as an int (backward compat).

    Returns:
        The ``fact_id`` (int) by default, or a dict with metadata when
        ``return_metadata=True``.

    Raises:
        sqlite3.Error: On database-level errors.
    """
    if trust_score is None:
        trust_score = 0.5
    if importance is None:
        importance = 0.5

    # Hive Memory scope validation
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"Invalid scope: '{scope}'. Must be one of: {', '.join(sorted(VALID_SCOPES))}"
        )

    # Typed fact validation
    if fact_type:
        store._validate_fact_type(fact_type, what, why, where_text, learned)

    # SHA-256 content hash for lifetime dedup
    content_hash = hashlib.sha256(
        content.encode() + str(project or "").encode()
        + str(scope or "canonical").encode()
    ).hexdigest()

    # Structured field values.
    # For INSERT we default to empty string; for UPDATE we pass None
    # so that COALESCE preserves the existing value.
    what_val: Optional[str] = what
    why_val: Optional[str] = why
    where_val: Optional[str] = where_text
    learned_val: Optional[str] = learned

    # Auto-extract topic_key from tags if not explicitly provided
    if not topic_key:
        m = re.search(r"(?:^|,)topic:([^,]+)", tags)
        if m:
            topic_key = "topic:" + m.group(1).strip()

    with store._lock:
        # Auto-vivify workspace
        store._ensure_workspace(project)

        # ---- Content hash dedup (lifetime) ----
        dedup_row = store._conn.execute(
            """SELECT fact_id, duplicate_count FROM facts
               WHERE content_hash = ? AND project IS ? AND scope IS ?
               AND (deleted IS NULL OR deleted = 0)""",
            (content_hash, project, scope),
        ).fetchone()
        if dedup_row:
            dedup_id = dedup_row["fact_id"]
            original_dup_count = dedup_row["duplicate_count"]
            store._conn.execute(
                """UPDATE facts SET duplicate_count = duplicate_count + 1,
                   updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?""",
                (dedup_id,),
            )
            store._conn.commit()
            store._invalidate_hrr_cache(dedup_id)
            store._log_event("fact_deduped", fact_id=dedup_id, project=project,
                            metadata={"duplicate_count": original_dup_count + 1, "original_fact_id": dedup_id})
            store._conn.commit()  # close implicit transaction opened by _log_event
            if project:
                store._conn.execute(
                    "UPDATE workspaces SET last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                    (project,),
                )
                store._conn.commit()
            if return_metadata:
                return {"id": dedup_id, "status": "dedup"}
            return dedup_id

        # ---- Topic upsert ----
        if topic_key:
            existing = store._conn.execute(
                "SELECT fact_id, content, revision_count FROM facts "
                "WHERE topic_key = ? AND (deleted IS NULL OR deleted = 0) LIMIT 1",
                (topic_key,),
            ).fetchone()
            if existing:
                eid = existing["fact_id"]
                store._conn.execute(
                    """UPDATE facts SET content = ?, updated_at = CURRENT_TIMESTAMP,
                       revision_count = revision_count + 1, category = ?, tags = ?,
                       trust_score = ?, importance = ?, project = ?, session_id = ?,
                       embedding = COALESCE(?, embedding),
                       what = COALESCE(?, what), why = COALESCE(?, why),
                       where_text = COALESCE(?, where_text),
                       learned = COALESCE(?, learned),
                       content_hash = ?,
                       source_harness = ?, source_agent = ?, source_kind = ?,
                       fact_type = COALESCE(?, fact_type), scope = ?
                    WHERE fact_id = ?""",
                    (content, category, tags, trust_score, importance,
                     project, session_id, embedding,
                     what_val, why_val, where_val, learned_val,
                     content_hash,
                     source_harness, source_agent, source_kind,
                     fact_type or None, scope,
                     eid),
                )
                store._conn.commit()
                store._invalidate_hrr_cache(eid)
                store._log_event("fact_added", fact_id=eid, project=project,
                                metadata={"category": category, "topic_key": topic_key, "scope": scope})
                store._conn.commit()  # close implicit transaction opened by _log_event
                if project:
                    store._conn.execute(
                        "UPDATE workspaces SET last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                        (project,),
                    )
                    store._conn.commit()
                if hrr.HAS_NUMPY:
                    store._pending_hrr.append((eid, content))
                    store._signal_flush()
                # Compute embedding if provider is active (skip if pre-supplied)
                if embedding is None:
                    store._maybe_store_embedding(eid, content)
                if return_metadata:
                    conflicts = store._detect_conflicts(
                        content=content, fact_id=eid,
                        project=project, topic_key=topic_key,
                    )
                    return {
                        "id": eid, "status": "updated",
                        "conflicts_with": conflicts,
                    }
                return eid

        # ---- Normal INSERT ----
        try:
            cur = store._conn.execute(
                """INSERT INTO facts
                   (content, category, tags, trust_score, importance,
                    project, session_id, topic_key, embedding,
                    what, why, where_text, learned, content_hash,
                    source_harness, source_agent, source_kind,
                    fact_type, scope)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?)""",
                (content, category, tags, trust_score, importance,
                 project, session_id, topic_key, embedding,
                 what_val or "", why_val or "", where_val or "",
                 learned_val or "", content_hash,
                 source_harness, source_agent, source_kind,
                 fact_type, scope),
            )
            is_new = cur.rowcount > 0
            store._conn.commit()
            fact_id = cur.lastrowid if is_new else 0
            if project and is_new:
                store._conn.execute(
                    "UPDATE workspaces SET fact_count = fact_count + 1, last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                    (project,),
                )
                store._conn.commit()
            elif project:
                store._conn.execute(
                    "UPDATE workspaces SET last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                    (project,),
                )
                store._conn.commit()
        except sqlite3.IntegrityError:
            logger.warning("Duplicate fact (content collision): %s", content[:60])
            row = store._conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?", (content,)
            ).fetchone()
            rid = row["fact_id"] if row else 0
            if rid:
                store._log_event("fact_deduped", fact_id=rid, project=project,
                                metadata={"original_fact_id": rid, "duplicate_count": 0})
                store._conn.commit()  # close implicit transaction opened by _log_event
            if return_metadata:
                return {"id": rid, "status": "dedup"}
            return rid

        if fact_id:
            store._log_event("fact_added", fact_id=fact_id, project=project,
                            metadata={"category": category, "topic_key": topic_key, "scope": scope})
            store._conn.commit()  # close implicit transaction opened by _log_event

        if fact_id and hrr.HAS_NUMPY:
            store._pending_hrr.append((fact_id, content))
            store._signal_flush()

        if entities:
            for entity_name in entities:
                store._ensure_entity(fact_id, entity_name)

        # Compute embedding if provider is active and not pre-supplied
        if fact_id and embedding is None:
            store._maybe_store_embedding(fact_id, content)

        # Final commit: close any implicit transaction left by _ensure_entity,
        # _maybe_store_embedding (NoopProvider case), or other DML in this block
        store._conn.commit()

    # ---- Conflict surfacing (outside lock) ----
    if return_metadata and fact_id:
        conflicts = store._detect_conflicts(
            content=content,
            fact_id=fact_id,
            project=project,
            topic_key=topic_key,
        )
        return {"id": fact_id, "status": "created", "conflicts_with": conflicts}

    return fact_id


def _detect_conflicts(
    store,
    content: str,
    fact_id: int,
    project: str,
    topic_key: str,
    limit: int = 5,
) -> list[dict]:
    """Search for existing facts that conflict with a newly added fact.

    Uses FTS5 with an OR-based query (non-trivial content tokens) and
    topic_key matching to detect conflicts.

    Args:
        content: The content of the newly added fact.
        fact_id: The ID of the newly added fact (excluded from results).
        project: Project namespace to scope the search.
        topic_key: Topic key of the new fact (for topic-based matching).
        limit: Max conflict candidates to return.

    Returns:
        List of dicts with keys ``id``, ``content``, ``score``.
    """
    # Build an OR-based FTS5 query from non-trivial content words.
    # FTS5 default MATCH requires ALL terms (AND), which is too strict
    # for conflict detection — we want any content overlap.
    words = content.split()
    # Filter out very short tokens (likely stop words / noise)
    sig_words = [w for w in words if len(w) >= 3]
    if not sig_words:
        return []

    or_query = " OR ".join(_sanitize_fts5(w) for w in sig_words)
    if not or_query.strip():
        return []

    try:
        params: list[str | int] = [or_query, fact_id]
        project_filter = "AND f.project IS ?"
        params.append(project)
        with store._lock:
            rows = store._conn.execute(
                f"""SELECT f.fact_id, f.content, f.topic_key, fts.rank
                    FROM facts f
                    JOIN facts_fts fts ON fts.rowid = f.fact_id
                    WHERE facts_fts MATCH ?
                    AND f.fact_id != ?
                    AND (f.deleted IS NULL OR f.deleted = 0)
                    {project_filter}
                    ORDER BY fts.rank
                    LIMIT ?""",
                params + [limit],
            ).fetchall()

        conflicts: list[dict] = []
        for row in rows:
            # FTS5 rank in SELECT returns -BM25.
            # rank = 0 -> perfect match (all query terms found)
            # rank < 0 -> partial match (some terms matched)
            rank = row["rank"]
            # Any non-zero rank means FTS5 detected overlap.
            is_similar = rank < 0
            same_topic = (
                topic_key and topic_key == row.get("topic_key", "") and topic_key
            )
            if is_similar or same_topic:
                score = -rank if rank < 0 else 0.0
                conflicts.append({
                    "id": row["fact_id"],
                    "content": row["content"],
                    "score": round(score, 4),
                })
        return conflicts
    except Exception:
        logger.exception("Conflict detection failed for fact %d", fact_id)
        return []


def add_fact_with_consolidation(
    store,
    content: str,
    category: str = "general",
    tags: str = "",
    trust_score: Optional[float] = None,
    importance: Optional[float] = None,
    project: str = "",
    session_id: str = "",
    topic_key: str = "",
    entities: Optional[list[str]] = None,
    search_fn: Optional[Callable] = None,
    llm_decide_fn: Optional[Callable] = None,
) -> dict:
    """Add a fact with active consolidation -- merges or deletes old facts on collision.

    Returns {"action": "added"|"merged"|"skipped"|"error", "fact_id": int, "detail": str}.
    """
    # Fast path: no consolidation needed
    if not search_fn or not llm_decide_fn:
        fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        return {"action": "added" if fid else "error", "fact_id": fid, "detail": ""}

    # Search for similar existing facts
    try:
        results = search_fn(query=content, limit=3)
    except TypeError:
        # search_fn doesn't support all kwargs
        results = search_fn(content, limit=3) if callable(search_fn) else []

    if not results:
        fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        return {"action": "added", "fact_id": fid, "detail": "no collision"}

    # Check Jaccard similarity for fast-path collision detection
    tokens_new = set(content.lower().split())
    best_sim = 0.0
    best_result = None
    for r in results:
        r_content = r.get("content", r.get("text", str(r)))
        tokens_existing = set(r_content.lower().split())
        if not tokens_new or not tokens_existing:
            continue
        jac = len(tokens_new & tokens_existing) / len(tokens_new | tokens_existing)
        if jac > best_sim:
            best_sim = jac
            best_result = r

    # Jaccard < 0.4 -> no significant overlap -> ADD directly
    if best_sim < 0.4:
        fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        return {"action": "added", "fact_id": fid, "detail": f"jaccard={best_sim:.2f} < 0.4"}

    # Jaccard >= 0.4 -> let LLM decide
    try:
        decision = llm_decide_fn(new_content=content, existing=best_result)
    except Exception as exc:
        logger.warning("LLM consolidation failed (%s), falling back to ADD", exc)
        fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        return {"action": "added", "fact_id": fid, "detail": f"llm_fallback: {exc}"}

    action = (decision or {}).get("action", "ADD")
    existing_fid = None
    if best_result:
        existing_fid = best_result.get("fact_id") or best_result.get("id")

    if action == "SKIP":
        return {"action": "skipped", "fact_id": existing_fid, "detail": decision.get("reason", "llm decided to skip")}

    if action == "UPDATE" or action == "MERGE":
        if existing_fid:
            merged = decision.get("merged_content", content)
            with store._lock:
                store._conn.execute(
                    "UPDATE facts SET content=?, updated_at=CURRENT_TIMESTAMP, revision_count=revision_count+1 WHERE fact_id=?",
                    (merged, existing_fid),
                )
                store._conn.commit()
                store._log_event("fact_merged", fact_id=existing_fid, project=project,
                                metadata={"replaced_fact_id": existing_fid})
                store._conn.commit()  # close implicit transaction opened by _log_event
            store._invalidate_hrr_cache(existing_fid)
            return {"action": "merged", "fact_id": existing_fid, "detail": f"updated #{existing_fid}"}

    if action == "REPLACE":
        if existing_fid:
            store.soft_delete_fact(existing_fid, reason="replaced")
        fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        if existing_fid and fid:
            with store._lock:
                store._log_event("fact_replaced", fact_id=fid, project=project,
                                metadata={"replaced_fact_id": existing_fid})
                store._conn.commit()  # close implicit transaction opened by _log_event
            # The new fact is derived from the (now-soft-deleted) original
            store._add_derivation_link(existing_fid, fid, judged_by="system")
        return {"action": "merged", "fact_id": fid, "detail": f"replaced #{existing_fid}"}

    # Default: ADD
    fid = store.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
    return {"action": "added", "fact_id": fid, "detail": f"action={action} defaulted to ADD"}


def soft_delete_fact(store, fact_id: int, reason: str = "") -> bool:
    """Soft-delete a fact.

    The fact remains in the database but is excluded from searches
    by default. A deleted reason is recorded for audit.

    Args:
        fact_id: ID of the fact to soft-delete.
        reason: Optional reason for deletion.

    Returns:
        True if a fact was soft-deleted, False if already deleted or not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        cur = store._conn.execute(
            "UPDATE facts SET deleted=1, deleted_reason=? WHERE fact_id=? AND (deleted IS NULL OR deleted=0)",
            (reason, fact_id),
        )
        store._conn.commit()
        if cur.rowcount > 0:
            store._log_event("fact_soft_deleted", fact_id=fact_id, project=fact_project,
                            metadata={"reason": reason})
            store._conn.commit()  # close implicit transaction opened by _log_event
            if fact_project:
                store._conn.execute(
                    "UPDATE workspaces SET fact_count = MAX(0, fact_count - 1), last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                    (fact_project,),
                )
                store._conn.commit()
        store._invalidate_hrr_cache(fact_id)
    return cur.rowcount > 0


def restore_fact(store, fact_id: int) -> bool:
    """Restore a previously soft-deleted or archived fact.

    Sets ``deleted = 0`` and clears ``deleted_reason``, making the fact
    visible in searches again. No-op if the fact is already active.

    Args:
        fact_id: ID of the fact to restore.

    Returns:
        True if a fact was restored, False if not found or already active.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        cur = store._conn.execute(
            "UPDATE facts SET deleted=0, deleted_reason='' "
            "WHERE fact_id=? AND deleted=1",
            (fact_id,),
        )
        store._conn.commit()
        if cur.rowcount > 0:
            store._log_event("fact_restored", fact_id=fact_id, project=fact_project, metadata={})
            store._conn.commit()  # close implicit transaction opened by _log_event
            if fact_project:
                store._conn.execute(
                    "UPDATE workspaces SET fact_count = fact_count + 1, last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                    (fact_project,),
                )
                store._conn.commit()
        store._invalidate_hrr_cache(fact_id)
    return cur.rowcount > 0


def remove_fact(store, fact_id: int) -> bool:
    """Permanently delete a fact from the database.

    This operation cannot be undone. Consider ``soft_delete_fact``
    for reversible deletion.

    Args:
        fact_id: ID of the fact to permanently delete.

    Returns:
        True if the fact was deleted.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        store._conn.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
        store._conn.commit()
        store._log_event("fact_removed", fact_id=fact_id, project=fact_project,
                        metadata={"permanent": True})
        store._conn.commit()  # close implicit transaction opened by _log_event
        store._invalidate_hrr_cache(fact_id)
        if fact_project:
            store._conn.execute(
                "UPDATE workspaces SET fact_count = MAX(0, fact_count - 1), last_active = datetime('now'), updated_at = datetime('now') WHERE name = ?",
                (fact_project,),
            )
            store._conn.commit()
    return True
