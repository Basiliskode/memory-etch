"""Distributed Sync — CDC export/import, peer management, conflict resolution.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def node_id(store) -> str:
    """Get this store's unique node ID (UUID v4), creating it if needed.

    The node_id is persisted in the store_meta table and survives
    store open/close cycles.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT value FROM store_meta WHERE key = 'node_id'"
        ).fetchone()
        if row:
            return row["value"]
        nid = str(uuid.uuid4())
        store._conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('node_id', ?)",
            (nid,),
        )
        store._conn.commit()
        return nid


def sync_prepare(store, event_cursor: int = 0) -> dict:
    """Build a sync bundle of all changes since the given event cursor.

    Uses the append-only event_log as a change-data-capture stream.
    Facts, relations, and workspaces affected by events after the
    cursor are collected into a portable JSON-serializable bundle.

    Args:
        event_cursor: event_id to start from (0 = all history).

    Returns:
        Bundle dict with keys: version, node_id, created_at,
        since_cursor, until_cursor, facts, relations, workspaces,
        deleted_fact_ids, purged_fact_ids.
    """
    with store._lock:
        events = store._conn.execute(
            "SELECT * FROM event_log WHERE event_id > ? ORDER BY event_id",
            (event_cursor,),
        ).fetchall()

        until_cursor = event_cursor
        if not events:
            return {
                "version": 2,
                "node_id": node_id(store),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "since_cursor": event_cursor,
                "until_cursor": until_cursor,
                "facts": [],
                "relations": [],
                "workspaces": [],
                "deleted_fact_ids": [],
                "purged_fact_ids": [],
            }

        # Collect affected fact_ids from events
        affected_fact_ids: set[int] = set()
        deleted_fact_ids: list[int] = []
        purged_fact_ids: list[int] = []

        for ev in events:
            e = dict(ev)
            fid = e.get("fact_id")
            etype = e.get("event_type", "")

            if fid is not None:
                affected_fact_ids.add(fid)

            if etype == "fact_soft_deleted" and fid is not None:
                deleted_fact_ids.append(fid)

            if etype == "fact_removed" and fid is not None:
                purged_fact_ids.append(fid)

        until_cursor = max(e["event_id"] for e in events)

        # Query facts (exclude hrr_vector and embedding)
        facts = []
        if affected_fact_ids:
            placeholders = ",".join("?" for _ in affected_fact_ids)
            facts = store._conn.execute(
                f"""SELECT fact_id, content, category, tags, trust_score,
                           retrieval_count, helpful_count, created_at, updated_at,
                           reinforcement_count, consolidated, importance,
                           session_id, topic_key, revision_count, project,
                           deleted, deleted_reason, replaced_by, content_hash,
                           duplicate_count, last_retrieved_at, what, why,
                           where_text, learned, source_harness, source_agent,
                           source_kind, scope, fact_type
                    FROM facts WHERE fact_id IN ({placeholders})""",
                list(affected_fact_ids),
            ).fetchall()

        # Query relations for affected facts
        relations = []
        if affected_fact_ids:
            placeholders = ",".join("?" for _ in affected_fact_ids)
            relations = store._conn.execute(
                f"""SELECT * FROM fact_relations
                    WHERE fact_id_a IN ({placeholders})
                       OR fact_id_b IN ({placeholders})""",
                list(affected_fact_ids) + list(affected_fact_ids),
            ).fetchall()

        # Query workspaces for affected projects
        projects = set()
        for ev in events:
            meta = ev["project"] or ""
            if meta:
                projects.add(meta)

        workspaces = []
        if projects:
            placeholders = ",".join("?" for _ in projects)
            workspaces = store._conn.execute(
                f"SELECT * FROM workspaces WHERE name IN ({placeholders})",
                list(projects),
            ).fetchall()

    return {
        "version": 2,
        "node_id": node_id(store),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "since_cursor": event_cursor,
        "until_cursor": until_cursor,
        "facts": [dict(r) for r in facts],
        "relations": [dict(r) for r in relations],
        "workspaces": [dict(r) for r in workspaces],
        "deleted_fact_ids": deleted_fact_ids,
        "purged_fact_ids": purged_fact_ids,
    }


def sync_apply(store, bundle: dict, strategy: str = "lww") -> dict:
    """Apply a sync bundle from another EtchStore instance.

    Imports facts, relations, and workspaces with conflict resolution.
    Uses content_hash as universal identity across instances.

    Args:
        bundle: Sync bundle dict from ``sync_prepare()``.
        strategy: Conflict resolution strategy:
            - ``"lww"`` (last-write-wins, default)
            - ``"skip"`` (skip conflicting facts)
            - ``"flag"`` (create conflict entries, skip fact)

    Returns:
        Report dict with: facts_imported, facts_skipped,
        facts_conflicted, relations_imported, workspaces_created,
        deleted_locally, errors.
    """
    required_keys = ("version", "node_id", "since_cursor", "until_cursor", "facts")
    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"Missing required key in bundle: {key}")

    report = {
        "facts_imported": 0,
        "facts_skipped": 0,
        "facts_conflicted": 0,
        "relations_imported": 0,
        "workspaces_created": 0,
        "deleted_locally": 0,
        "errors": [],
    }

    with store._lock:
        # ---- 1. Deleted facts ----
        deleted_hashes: set[str] = set()
        for deleted_fid in bundle.get("deleted_fact_ids", []):
            # Look up local fact with same content_hash (we need to correlate
            # across instances — find by content_hash in the bundle facts)
            matching_facts = [
                f for f in bundle["facts"]
                if f.get("fact_id") == deleted_fid
            ]
            for mf in matching_facts:
                remote_hash = mf.get("content_hash") or hashlib.sha256(
                    mf["content"].encode() + str(mf.get("project", "")).encode()
                ).hexdigest()
                deleted_hashes.add(remote_hash)
                local = store._conn.execute(
                    """SELECT fact_id, deleted FROM facts
                       WHERE content_hash = ? AND project IS ?
                       AND (deleted IS NULL OR deleted = 0)
                       LIMIT 1""",
                    (remote_hash, mf.get("project", "")),
                ).fetchone()
                if local:
                    store._conn.execute(
                        "UPDATE facts SET deleted=1, deleted_reason='sync_remote_delete' WHERE fact_id=?",
                        (local["fact_id"],),
                    )
                    store._log_event(
                        "fact_soft_deleted",
                        fact_id=local["fact_id"],
                        project=mf.get("project", ""),
                        metadata={"reason": "sync_remote_delete"},
                    )
                    report["deleted_locally"] += 1

        # ---- 2. Import facts ----
        for fact in bundle["facts"]:
            content = fact.get("content", "")
            project = fact.get("project", "")
            # Use the bundle's content_hash when available (preserves identity
            # across instances even when content has diverged). Fall back to
            # recomputing for backward compatibility with older bundles.
            remote_hash = fact.get("content_hash") or hashlib.sha256(
                content.encode() + str(project).encode()
            ).hexdigest()

            # Skip facts that were already handled via deleted_fact_ids
            if remote_hash in deleted_hashes:
                continue

            local = store._conn.execute(
                """SELECT fact_id, content, updated_at, deleted FROM facts
                   WHERE content_hash = ? AND project IS ?
                   AND (deleted IS NULL OR deleted = 0)
                   LIMIT 1""",
                (remote_hash, project),
            ).fetchone()

            if not local:
                # ---- Insert new fact ----
                store._conn.execute(
                    """INSERT INTO facts
                       (content, category, tags, trust_score, importance,
                        project, session_id, topic_key,
                        what, why, where_text, learned, content_hash,
                        source_harness, source_agent, source_kind,
                        fact_type, scope, updated_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        content,
                        fact.get("category", "general"),
                        fact.get("tags", ""),
                        fact.get("trust_score", 0.5),
                        fact.get("importance", 0.5),
                        project,
                        fact.get("session_id", ""),
                        fact.get("topic_key", ""),
                        fact.get("what", ""),
                        fact.get("why", ""),
                        fact.get("where_text", ""),
                        fact.get("learned", ""),
                        remote_hash,
                        fact.get("source_harness", ""),
                        fact.get("source_agent", ""),
                        fact.get("source_kind", ""),
                        fact.get("fact_type", ""),
                        fact.get("scope", "canonical"),
                        fact.get("updated_at"),
                        fact.get("created_at"),
                    ),
                )
                store._log_event(
                    "fact_added",
                    project=project,
                    metadata={"source": "sync", "remote_node": bundle.get("node_id", "")},
                )
                report["facts_imported"] += 1

            else:
                local_fid = local["fact_id"]
                local_content = local["content"]
                local_updated = local["updated_at"] or ""
                remote_updated = fact.get("updated_at", "")

                if local_content == content:
                    # Same content — LWW on metadata
                    if remote_updated and remote_updated >= local_updated:
                        store._conn.execute(
                            """UPDATE facts SET
                               trust_score = ?, importance = ?, tags = ?,
                               category = ?, topic_key = ?,
                               source_harness = ?, source_agent = ?, source_kind = ?,
                               fact_type = ?, scope = ?, updated_at = ?
                            WHERE fact_id = ?""",
                            (
                                fact.get("trust_score", 0.5),
                                fact.get("importance", 0.5),
                                fact.get("tags", ""),
                                fact.get("category", "general"),
                                fact.get("topic_key", ""),
                                fact.get("source_harness", ""),
                                fact.get("source_agent", ""),
                                fact.get("source_kind", ""),
                                fact.get("fact_type", ""),
                                fact.get("scope", "canonical"),
                                remote_updated,
                                local_fid,
                            ),
                        )
                    report["facts_imported"] += 1
                else:
                    # Content differs — CONFLICT
                    if strategy == "skip":
                        report["facts_skipped"] += 1
                    elif strategy == "flag":
                        # Build local metadata for comparison
                        local_meta = {
                            "content": local_content,
                            "category": fact.get("category", ""),
                            "tags": fact.get("tags", ""),
                            "trust_score": fact.get("trust_score", 0.5),
                            "importance": fact.get("importance", 0.5),
                            "project": project,
                            "updated_at": local_updated,
                        }
                        store._conn.execute(
                            """INSERT INTO sync_conflicts
                               (content_hash, local_fact_id, local_content,
                                local_metadata, remote_data)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                remote_hash,
                                local_fid,
                                local_content,
                                json.dumps(local_meta),
                                json.dumps(fact, default=str),
                            ),
                        )
                        report["facts_conflicted"] += 1
                    else:
                        # lww — can't auto-resolve content conflict
                        report["facts_skipped"] += 1

                # Ensure workspace exists for this fact's project
                if project:
                    store._ensure_workspace(project)

        # ---- 3. Import relations ----
        for rel in bundle.get("relations", []):
            if not rel.get("fact_id_a") or not rel.get("fact_id_b"):
                continue
            try:
                store._conn.execute(
                    """INSERT OR IGNORE INTO fact_relations
                       (fact_id_a, fact_id_b, relation_type, confidence, judged_by)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        rel["fact_id_a"],
                        rel["fact_id_b"],
                        rel.get("relation_type", "related"),
                        rel.get("confidence", 0.5),
                        rel.get("judged_by", "sync"),
                    ),
                )
                report["relations_imported"] += 1
            except Exception as exc:
                report["errors"].append(f"relation import failed: {exc}")

        # ---- 4. Import workspaces ----
        for ws in bundle.get("workspaces", []):
            name = ws.get("name", "")
            if name:
                store._ensure_workspace(name)
                report["workspaces_created"] += 1

        store._conn.commit()

    return report


def sync_to_file(store, path: str, event_cursor: int = 0) -> dict:
    """Export a sync bundle to a JSON file.

    Convenience wrapper: calls ``sync_prepare()`` and writes the
    result to a JSON file.

    Args:
        path: Output file path.
        event_cursor: event_id cursor to start from.

    Returns:
        Bundle metadata dict with keys: node_id, since_cursor,
        until_cursor, fact_count.
    """
    bundle = sync_prepare(store, event_cursor)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
    return {
        "node_id": bundle["node_id"],
        "since_cursor": bundle["since_cursor"],
        "until_cursor": bundle["until_cursor"],
        "fact_count": len(bundle["facts"]),
    }


def sync_from_file(store, path: str, strategy: str = "lww") -> dict:
    """Import a sync bundle from a JSON file.

    Convenience wrapper: reads a JSON sync bundle from a file and
    calls ``sync_apply()``.

    Args:
        path: Input file path.
        strategy: Conflict resolution strategy.

    Returns:
        Import report dict from ``sync_apply()``.
    """
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    return sync_apply(store, bundle, strategy=strategy)


def register_peer(store, name: str, address: str, kind: str = "file") -> dict:
    """Register a sync peer.

    Args:
        name: Unique name for this peer.
        address: File path or URL for the sync bundle.
        kind: Transport kind (``"file"`` or ``"http"``).

    Returns:
        Peer dict with peer_id, name, kind, address.

    Raises:
        ValueError: If name is already registered.
    """
    with store._lock:
        existing = store._conn.execute(
            "SELECT peer_id FROM sync_peers WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            raise ValueError(f"Peer '{name}' is already registered")
        cur = store._conn.execute(
            "INSERT INTO sync_peers (name, kind, address) VALUES (?, ?, ?)",
            (name, kind, address),
        )
        store._conn.commit()
        return {
            "peer_id": cur.lastrowid,
            "name": name,
            "kind": kind,
            "address": address,
        }


def unregister_peer(store, name: str) -> bool:
    """Remove a peer registration.

    Args:
        name: Name of the peer to unregister.

    Returns:
        True if a peer was deleted, False if not found.
    """
    with store._lock:
        cur = store._conn.execute(
            "DELETE FROM sync_peers WHERE name = ?", (name,)
        )
        store._conn.commit()
        return cur.rowcount > 0


def list_peers(store) -> list[dict]:
    """List all registered sync peers.

    Returns:
        List of peer dicts, ordered by name.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM sync_peers ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def sync_with_peer(
    store, peer_name: str, direction: str = "both", strategy: str = "lww"
) -> dict:
    """Run a full sync cycle with a registered peer.

    For ``"file"`` peers, exports to and/or imports from the peer's
    configured file path.

    Args:
        peer_name: Name of the registered peer.
        direction: ``"push"`` (export), ``"pull"`` (import), or
            ``"both"`` (export then import, default).
        strategy: Conflict resolution strategy.

    Returns:
        Combined report dict with keys: peer_name, direction,
        push_result, pull_result.

    Raises:
        ValueError: If peer is not found.
    """
    with store._lock:
        peer = store._conn.execute(
            "SELECT * FROM sync_peers WHERE name = ?", (peer_name,)
        ).fetchone()
    if not peer:
        raise ValueError(f"Peer '{peer_name}' not found")

    peer = dict(peer)
    address = peer.get("address", "")
    cursor = peer.get("last_sync_cursor", 0)
    report: dict = {
        "peer_name": peer_name,
        "direction": direction,
        "push_result": None,
        "pull_result": None,
    }

    if direction in ("push", "both"):
        bundle = sync_prepare(store, cursor)
        sync_to_file(store, address, cursor)
        push_meta = {
            "node_id": bundle["node_id"],
            "since_cursor": bundle["since_cursor"],
            "until_cursor": bundle["until_cursor"],
            "fact_count": len(bundle["facts"]),
        }
        report["push_result"] = push_meta
        store._log_event(
            "sync_push",
            metadata={
                "peer": peer_name,
                "fact_count": len(bundle["facts"]),
                "cursor_from": cursor,
                "cursor_to": bundle["until_cursor"],
            },
        )

        # Update cursor
        with store._lock:
            store._conn.execute(
                "UPDATE sync_peers SET last_sync_cursor = ?, last_sync_at = datetime('now') WHERE name = ?",
                (bundle["until_cursor"], peer_name),
            )
            store._conn.commit()

    if direction in ("pull", "both"):
        if os.path.exists(address):
            pull_report = sync_from_file(store, address, strategy=strategy)
            report["pull_result"] = pull_report
            store._log_event(
                "sync_pull",
                metadata={
                    "peer": peer_name,
                    "facts_imported": pull_report.get("facts_imported", 0),
                    "conflicts": pull_report.get("facts_conflicted", 0),
                },
            )

            # Update cursor from bundle
            with store._lock:
                row = store._conn.execute(
                    "SELECT value FROM store_meta WHERE key = 'node_id'"
                ).fetchone()
                our_node_id = row["value"] if row else ""
                try:
                    with open(address, "r", encoding="utf-8") as f:
                        read_bundle = json.load(f)
                    if read_bundle.get("node_id") != our_node_id:
                        store._conn.execute(
                            "UPDATE sync_peers SET last_sync_cursor = ?, last_sync_at = datetime('now') WHERE name = ?",
                            (read_bundle.get("until_cursor", cursor), peer_name),
                        )
                        store._conn.commit()
                except Exception:
                    pass

    return report


def get_sync_conflicts(store, status: Optional[str] = "unresolved") -> list[dict]:
    """List sync conflicts, optionally filtered by status.

    Args:
        status: Filter by conflict status (``"unresolved"``,
            ``"resolved_keep_local"``, etc.), or None for all.

    Returns:
        List of conflict dicts, newest first.
    """
    with store._lock:
        if status is not None:
            rows = store._conn.execute(
                "SELECT * FROM sync_conflicts WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT * FROM sync_conflicts ORDER BY created_at DESC"
            ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["remote_data"] = json.loads(d.get("remote_data", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            d["local_metadata"] = json.loads(d.get("local_metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        result.append(d)
    return result


def resolve_sync_conflict(
    store, conflict_id: int, resolution: str, keep_content: bool = False
) -> bool:
    """Resolve a sync conflict.

    Args:
        conflict_id: The conflict to resolve.
        resolution: One of:
            - ``"keep_local"`` — keep existing fact, discard remote
            - ``"keep_remote"`` — update local fact with remote content+metadata
            - ``"keep_both"`` — add remote content as a separate fact
        keep_content: If True AND resolution is ``"keep_remote"``,
            the local fact's content is overwritten with remote content.

    Returns:
        True if resolved successfully.

    Raises:
        ValueError: If conflict not found or resolution is invalid.
    """
    valid_resolutions = {"keep_local", "keep_remote", "keep_both"}
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Invalid resolution: '{resolution}'. Must be one of: {', '.join(sorted(valid_resolutions))}"
        )

    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM sync_conflicts WHERE conflict_id = ?", (conflict_id,)
        ).fetchone()

    if not row:
        raise ValueError(f"Conflict {conflict_id} not found")

    conflict = dict(row)
    try:
        remote_data = json.loads(conflict.get("remote_data", "{}"))
    except (json.JSONDecodeError, TypeError):
        remote_data = {}

    local_fid = conflict.get("local_fact_id")

    if resolution == "keep_local":
        with store._lock:
            store._conn.execute(
                """UPDATE sync_conflicts
                   SET status = 'resolved_keep_local', resolved_at = datetime('now')
                   WHERE conflict_id = ?""",
                (conflict_id,),
            )
            store._conn.commit()

    elif resolution == "keep_remote":
        if local_fid is None:
            raise ValueError("Cannot resolve: local_fact_id is NULL")

        if keep_content:
            # Override content AND metadata
            remote_hash = hashlib.sha256(
                remote_data.get("content", "").encode()
                + str(remote_data.get("project", "")).encode()
            ).hexdigest()
            with store._lock:
                store._conn.execute(
                    """UPDATE facts SET
                       content = ?, content_hash = ?, category = ?, tags = ?,
                       trust_score = ?, importance = ?, topic_key = ?,
                       what = ?, why = ?, where_text = ?, learned = ?,
                       source_harness = ?, source_agent = ?, source_kind = ?,
                       fact_type = ?, scope = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?""",
                    (
                        remote_data.get("content", ""),
                        remote_hash,
                        remote_data.get("category", "general"),
                        remote_data.get("tags", ""),
                        remote_data.get("trust_score", 0.5),
                        remote_data.get("importance", 0.5),
                        remote_data.get("topic_key", ""),
                        remote_data.get("what", ""),
                        remote_data.get("why", ""),
                        remote_data.get("where_text", ""),
                        remote_data.get("learned", ""),
                        remote_data.get("source_harness", ""),
                        remote_data.get("source_agent", ""),
                        remote_data.get("source_kind", ""),
                        remote_data.get("fact_type", ""),
                        remote_data.get("scope", "canonical"),
                        local_fid,
                    ),
                )
                store._conn.execute(
                    """UPDATE sync_conflicts
                       SET status = 'resolved_keep_remote', resolved_at = datetime('now')
                       WHERE conflict_id = ?""",
                    (conflict_id,),
                )
                store._conn.commit()
                store._log_event(
                    "sync_conflict_resolved",
                    metadata={
                        "conflict_id": conflict_id,
                        "resolution": "keep_remote",
                        "keep_content": True,
                    },
                )
        else:
            # Only update metadata, keep local content
            with store._lock:
                store._conn.execute(
                    """UPDATE facts SET
                       category = ?, tags = ?, trust_score = ?, importance = ?,
                       topic_key = ?, what = ?, why = ?, where_text = ?, learned = ?,
                       source_harness = ?, source_agent = ?, source_kind = ?,
                       fact_type = ?, scope = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?""",
                    (
                        remote_data.get("category", "general"),
                        remote_data.get("tags", ""),
                        remote_data.get("trust_score", 0.5),
                        remote_data.get("importance", 0.5),
                        remote_data.get("topic_key", ""),
                        remote_data.get("what", ""),
                        remote_data.get("why", ""),
                        remote_data.get("where_text", ""),
                        remote_data.get("learned", ""),
                        remote_data.get("source_harness", ""),
                        remote_data.get("source_agent", ""),
                        remote_data.get("source_kind", ""),
                        remote_data.get("fact_type", ""),
                        remote_data.get("scope", "canonical"),
                        local_fid,
                    ),
                )
                store._conn.execute(
                    """UPDATE sync_conflicts
                       SET status = 'resolved_keep_remote', resolved_at = datetime('now')
                       WHERE conflict_id = ?""",
                    (conflict_id,),
                )
                store._conn.commit()
                store._log_event(
                    "sync_conflict_resolved",
                    metadata={
                        "conflict_id": conflict_id,
                        "resolution": "keep_remote",
                        "keep_content": False,
                    },
                )
        store._invalidate_hrr_cache(local_fid)

    elif resolution == "keep_both":
        # Add remote content as a new fact (bypass dedup since content differs)
        remote_content = remote_data.get("content", "")
        remote_project = remote_data.get("project", "")
        # Use add_fact's normal flow — since content differs from local,
        # it will be inserted as new.
        new_fid = store.add_fact(
            content=remote_content,
            category=remote_data.get("category", "general"),
            tags=remote_data.get("tags", ""),
            trust_score=remote_data.get("trust_score"),
            importance=remote_data.get("importance"),
            project=remote_project,
            session_id=remote_data.get("session_id", ""),
            topic_key=remote_data.get("topic_key", ""),
            what=remote_data.get("what"),
            why=remote_data.get("why"),
            where_text=remote_data.get("where_text"),
            learned=remote_data.get("learned"),
            source_harness=remote_data.get("source_harness", ""),
            source_agent=remote_data.get("source_agent", ""),
            source_kind=remote_data.get("source_kind", ""),
            fact_type=remote_data.get("fact_type", ""),
            scope=remote_data.get("scope", "canonical"),
        )
        with store._lock:
            store._conn.execute(
                """UPDATE sync_conflicts
                   SET status = 'resolved_keep_both', resolved_at = datetime('now')
                   WHERE conflict_id = ?""",
                (conflict_id,),
            )
            store._conn.commit()
            store._log_event(
                "sync_conflict_resolved",
                metadata={
                    "conflict_id": conflict_id,
                    "resolution": "keep_both",
                    "new_fact_id": new_fid,
                },
            )

    return True
