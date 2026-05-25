"""Garbage Collection — hard delete, orphan cleanup, pruning, vacuum.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def gc(store, config: Optional[dict] = None, dry_run: bool = False) -> dict:
    """Run garbage collection across all cleanup phases.

    Each phase is optional and configurable via *config*.
    When *dry_run* is True, reports what WOULD be done without
    actually modifying data.

    Phases (run in order):
        1. hard_delete — permanently remove soft-deleted facts
        2. orphan_cleanup — clean orphan relations, entities, workspaces
        3. prune_event_log — delete old event_log entries
        4. prune_turn_buffer — delete old turn_buffer entries
        5. snapshot_retention — delete old/excess snapshots
        6. vacuum — VACUUM if free page threshold is exceeded

    Args:
        config: Optional dict overriding any ``_GC_DEFAULT_CONFIG`` key.
        dry_run: If True, report counts without modifying data.

    Returns:
        Dict with per-phase stats, ``duration_ms``, and ``dry_run`` flag.
    """
    t0 = time.time()
    cfg = {**store._GC_DEFAULT_CONFIG, **(config or {})}

    phases = {
        "hard_delete": _gc_hard_delete(store, cfg, dry_run),
        "orphan_cleanup": _gc_orphan_cleanup(store, cfg, dry_run),
        "prune_event_log": _gc_prune_event_log(store, cfg, dry_run),
        "prune_turn_buffer": _gc_prune_turn_buffer(store, cfg, dry_run),
        "snapshot_retention": _gc_snapshot_retention(store, cfg, dry_run),
        "vacuum": _gc_vacuum(store, cfg, dry_run),
    }

    duration = int((time.time() - t0) * 1000)
    logger.info("GC complete in %dms (dry_run=%s)", duration, dry_run)
    return {
        "phases": phases,
        "duration_ms": duration,
        "dry_run": dry_run,
    }


def _gc_hard_delete(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 1: Permanently delete facts soft-deleted > N days ago."""
    days = cfg["hard_delete_days"]
    with store._lock:
        if dry_run:
            count = store._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE deleted = 1 AND updated_at < datetime('now', '-' || ? || ' days')",
                (days,),
            ).fetchone()[0]
            return {"deleted": count}

        # Delete related fact_entities first
        store._conn.execute(
            "DELETE FROM fact_entities WHERE fact_id IN ("
            "SELECT fact_id FROM facts WHERE deleted = 1 AND updated_at < datetime('now', '-' || ? || ' days')"
            ")",
            (days,),
        )

        cursor = store._conn.execute(
            "DELETE FROM facts WHERE deleted = 1 AND updated_at < datetime('now', '-' || ? || ' days')",
            (days,),
        )
        count = cursor.rowcount
        store._conn.commit()
        if count:
            store._log_event("gc_hard_deleted", metadata={"count": count})
            store._conn.commit()  # close implicit transaction opened by _log_event
        logger.info("GC hard_delete: %d facts permanently deleted", count)
        return {"deleted": count}


def _gc_orphan_cleanup(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 2: Remove orphan relations, entities, and empty workspaces."""
    with store._lock:
        if dry_run:
            rel_count = store._conn.execute(
                "SELECT COUNT(*) FROM fact_relations WHERE fact_id_a NOT IN (SELECT fact_id FROM facts) OR fact_id_b NOT IN (SELECT fact_id FROM facts)"
            ).fetchone()[0]
            fe_count = store._conn.execute(
                "SELECT COUNT(*) FROM fact_entities WHERE fact_id NOT IN (SELECT fact_id FROM facts)"
            ).fetchone()[0]
            e_count = store._conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM fact_entities)"
            ).fetchone()[0]
            ws_count = store._conn.execute(
                "SELECT COUNT(*) FROM workspaces WHERE fact_count = 0 AND (deleted IS NULL OR deleted = 0) AND last_active < datetime('now', '-90 days')"
            ).fetchone()[0]
            return {"fact_relations": rel_count, "fact_entities": fe_count, "entities": e_count, "workspaces": ws_count}

        c1 = store._conn.execute(
            "DELETE FROM fact_relations WHERE fact_id_a NOT IN (SELECT fact_id FROM facts) OR fact_id_b NOT IN (SELECT fact_id FROM facts)"
        )
        rel_count = c1.rowcount

        c2 = store._conn.execute(
            "DELETE FROM fact_entities WHERE fact_id NOT IN (SELECT fact_id FROM facts)"
        )
        fe_count = c2.rowcount

        c3 = store._conn.execute(
            "DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM fact_entities)"
        )
        e_count = c3.rowcount

        c4 = store._conn.execute(
            "UPDATE workspaces SET deleted = 1, updated_at = datetime('now') WHERE fact_count = 0 AND (deleted IS NULL OR deleted = 0) AND last_active < datetime('now', '-90 days')"
        )
        ws_count = c4.rowcount

        store._conn.commit()
        total = rel_count + fe_count + e_count + ws_count
        if total:
            store._log_event("gc_orphans_removed", metadata={
                "fact_relations": rel_count,
                "fact_entities": fe_count,
                "entities": e_count,
                "workspaces": ws_count,
            })
            store._conn.commit()  # close implicit transaction opened by _log_event
        logger.info("GC orphan_cleanup: fact_relations=%d fact_entities=%d entities=%d workspaces=%d",
                    rel_count, fe_count, e_count, ws_count)
        return {"fact_relations": rel_count, "fact_entities": fe_count, "entities": e_count, "workspaces": ws_count}


def _gc_prune_event_log(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 3: Delete event_log entries older than N days."""
    days = cfg["prune_event_log_days"]
    with store._lock:
        if dry_run:
            count = store._conn.execute(
                "SELECT COUNT(*) FROM event_log WHERE created_at < datetime('now', '-' || ? || ' days')",
                (days,),
            ).fetchone()[0]
            return {"deleted": count}

        cursor = store._conn.execute(
            "DELETE FROM event_log WHERE created_at < datetime('now', '-' || ? || ' days')",
            (days,),
        )
        count = cursor.rowcount
        store._conn.commit()
        if count:
            store._log_event("gc_event_log_pruned", metadata={"count": count})
            store._conn.commit()  # close implicit transaction opened by _log_event
        logger.info("GC prune_event_log: %d entries deleted", count)
        return {"deleted": count}


def _gc_prune_turn_buffer(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 4: Delete turn_buffer entries older than N days."""
    days = cfg["prune_turn_buffer_days"]
    with store._lock:
        if dry_run:
            count = store._conn.execute(
                "SELECT COUNT(*) FROM turn_buffer WHERE created_at < datetime('now', '-' || ? || ' days')",
                (days,),
            ).fetchone()[0]
            return {"deleted": count}

        cursor = store._conn.execute(
            "DELETE FROM turn_buffer WHERE created_at < datetime('now', '-' || ? || ' days')",
            (days,),
        )
        count = cursor.rowcount
        store._conn.commit()
        if count:
            store._log_event("gc_turn_buffer_pruned", metadata={"count": count})
            store._conn.commit()  # close implicit transaction opened by _log_event
        logger.info("GC prune_turn_buffer: %d entries deleted", count)
        return {"deleted": count}


def _gc_snapshot_retention(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 5: Delete old snapshots and enforce per-project retention."""
    keep = cfg["snapshot_keep"]
    max_age = cfg["snapshot_max_age_days"]
    with store._lock:
        if dry_run:
            age_count = store._conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE created_at < datetime('now', '-' || ? || ' days')",
                (max_age,),
            ).fetchone()[0]
            # Count retention-based candidates among non-age-deleted
            extra = 0
            projects = store._conn.execute(
                "SELECT DISTINCT COALESCE(project, '') AS proj FROM snapshots WHERE created_at >= datetime('now', '-' || ? || ' days')",
                (max_age,),
            ).fetchall()
            for row in projects:
                proj = row["proj"]
                total = store._conn.execute(
                    "SELECT COUNT(*) FROM snapshots WHERE project = ? AND created_at >= datetime('now', '-' || ? || ' days')",
                    (proj, max_age),
                ).fetchone()[0]
                if total > keep:
                    extra += total - keep
            return {"deleted": age_count + extra}

        count = 0

        # 1. Delete by age
        c = store._conn.execute(
            "DELETE FROM snapshots WHERE created_at < datetime('now', '-' || ? || ' days')",
            (max_age,),
        )
        count += c.rowcount

        # 2. Enforce per-project retention (on remaining snapshots)
        projects = store._conn.execute(
            "SELECT DISTINCT COALESCE(project, '') AS proj FROM snapshots"
        ).fetchall()
        for row in projects:
            proj = row["proj"]
            snap_rows = store._conn.execute(
                "SELECT snapshot_id FROM snapshots WHERE project = ? ORDER BY created_at DESC",
                (proj,),
            ).fetchall()
            if len(snap_rows) > keep:
                ids = [r[0] for r in snap_rows[keep:]]
                placeholders = ",".join("?" for _ in ids)
                c = store._conn.execute(
                    f"DELETE FROM snapshots WHERE snapshot_id IN ({placeholders})",
                    ids,
                )
                count += c.rowcount

        store._conn.commit()
        if count:
            store._log_event("gc_snapshots_removed", metadata={"count": count})
            store._conn.commit()  # close implicit transaction opened by _log_event
        logger.info("GC snapshot_retention: %d snapshots deleted", count)
        return {"deleted": count}


def _gc_vacuum(store, cfg: dict, dry_run: bool) -> dict:
    """Phase 6: VACUUM if free page threshold is exceeded."""
    pct = cfg["vacuum_threshold_pct"]
    with store._lock:
        freelist = store._conn.execute("PRAGMA freelist_count").fetchone()[0]
        total = store._conn.execute("PRAGMA page_count").fetchone()[0]
        needs_vacuum = total > 0 and (freelist * 100 // total) >= pct
        if needs_vacuum and not dry_run:
            # Commit any pending transaction before VACUUM
            store._conn.commit()
            store._conn.execute("VACUUM")
            logger.info("GC vacuum completed")
    return {"vacuumed": needs_vacuum}
