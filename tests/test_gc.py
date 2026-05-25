"""Tests for GC/Compaction system — EtchStore.gc() and helpers."""

from unittest.mock import MagicMock

import pytest

from memory_etch.store import EtchStore


def _exec(conn, sql, params=None):
    """Execute a single SQL statement — reliable alternative to executescript."""
    if params:
        return conn.execute(sql, params)
    return conn.execute(sql)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


@pytest.fixture
def store_with_old_data(store):
    """Store with soft-deleted facts, old logs, turns, and snapshots."""
    # Hard-delete candidates (soft-deleted + aged)
    for i in range(3):
        fid = store.add_fact(f"old fact {i}", project="test")
        store.soft_delete_fact(fid)
    store._conn.execute(
        "UPDATE facts SET updated_at = datetime('now', '-60 days') WHERE deleted = 1"
    )

    # Old event_log entries
    store._conn.execute(
        "INSERT INTO event_log (event_type, created_at) VALUES (?, datetime('now', ?))",
        ("test_old", "-100 days"),
    )
    store._conn.execute(
        "INSERT INTO event_log (event_type, created_at) VALUES (?, datetime('now', ?))",
        ("test_recent", "-1 hour"),
    )

    # Old turn_buffer entries
    store._conn.execute(
        "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES (?, ?, ?, datetime('now', ?))",
        ("s1", "user", "old turn", "-60 days"),
    )
    store._conn.execute(
        "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES (?, ?, ?, datetime('now', ?))",
        ("s1", "user", "recent turn", "-1 hour"),
    )

    # Old snapshots
    store.create_snapshot("old_snap1", project="test")
    store.create_snapshot("old_snap2", project="test")
    store.create_snapshot("old_snap3", project="test")
    store._conn.execute(
        "UPDATE snapshots SET created_at = datetime('now', '-400 days') WHERE name IN ('old_snap1', 'old_snap2')"
    )

    # Orphan data
    store._conn.execute(
        "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
        ("orphan_entity", "test"),
    )

    # Orphan workspace (empty, not deleted, old last_active)
    store.create_workspace("empty_old_ws")
    store._conn.execute(
        "UPDATE workspaces SET fact_count = 0, last_active = datetime('now', '-100 days') WHERE name = ?",
        ("empty_old_ws",),
    )

    store._conn.commit()  # close any implicit transaction from raw execute calls
    return store


def _age_fact(store, fid: int, days_ago: int = 60) -> None:
    """Manually age a fact's updated_at for testing."""
    store._conn.execute(
        "UPDATE facts SET updated_at = datetime('now', ? || ' days') WHERE fact_id = ?",
        (f"-{days_ago}", fid),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────────────────────────────────────

class TestGCDefaults:
    def test_defaults_exist(self):
        """_GC_DEFAULT_CONFIG is a class-level dict with all expected keys."""
        assert hasattr(EtchStore, "_GC_DEFAULT_CONFIG")
        cfg = EtchStore._GC_DEFAULT_CONFIG
        assert isinstance(cfg, dict)
        assert cfg["hard_delete_days"] == 30
        assert cfg["prune_event_log_days"] == 90
        assert cfg["prune_turn_buffer_days"] == 30
        assert cfg["snapshot_keep"] == 10
        assert cfg["snapshot_max_age_days"] == 365
        assert cfg["vacuum_threshold_pct"] == 20

    def test_defaults_are_sensible(self):
        cfg = EtchStore._GC_DEFAULT_CONFIG
        assert cfg["hard_delete_days"] >= 1
        assert cfg["prune_event_log_days"] >= cfg["hard_delete_days"]
        assert cfg["snapshot_keep"] >= 1
        assert 1 <= cfg["vacuum_threshold_pct"] <= 100


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Hard Delete
# ─────────────────────────────────────────────────────────────────────────────

class TestHardDelete:
    def test_hard_deletes_soft_deleted_facts(self, store):
        """Aged soft-deleted facts are permanently removed."""
        fid = store.add_fact("to delete permanently")
        store.soft_delete_fact(fid, reason="test")
        _age_fact(store, fid, 60)  # must be aged to qualify
        result = store.gc({"hard_delete_days": 1})
        assert result["phases"]["hard_delete"]["deleted"] == 1
        assert store.get_fact(fid) is None  # fully gone

    def test_fts5_cleans_up_on_hard_delete(self, store):
        """FTS5 index is cleaned when facts are hard-deleted."""
        fid = store.add_fact("unique searchable content for gc test")
        store.soft_delete_fact(fid, reason="test")
        _age_fact(store, fid, 60)
        store.gc({"hard_delete_days": 1})
        results = store.search_facts("unique searchable content")
        assert len(results) == 0

    def test_dry_run_does_not_delete(self, store):
        """Dry run reports count but leaves facts intact."""
        fid = store.add_fact("dry run fact")
        store.soft_delete_fact(fid, reason="test")
        _age_fact(store, fid, 60)
        result = store.gc({"hard_delete_days": 1}, dry_run=True)
        assert result["dry_run"] is True
        assert result["phases"]["hard_delete"]["deleted"] == 1
        # Fact should still exist (as soft-deleted)
        fact = store.get_fact(fid)
        assert fact is not None
        assert fact["deleted"] == 1

    def test_skips_recently_deleted_with_default_config(self, store):
        """Default 30-day threshold skips just-deleted facts."""
        fid = store.add_fact("recently deleted")
        store.soft_delete_fact(fid, reason="test")
        # fact was just soft-deleted (updated_at = now), default 30d threshold skips it
        result = store.gc()
        assert result["phases"]["hard_delete"]["deleted"] == 0

    def test_fact_entities_cleaned_on_hard_delete(self, store):
        """Related fact_entities rows are removed with the fact."""
        fid = store.add_fact("entity test fact", entities=["test-entity"])
        store.soft_delete_fact(fid, reason="test")
        _age_fact(store, fid, 60)
        store.gc({"hard_delete_days": 1})
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?", (fid,)
        ).fetchone()[0]
        assert remaining == 0

    def test_logs_event_on_hard_delete(self, store):
        """gc_hard_deleted event is logged to event_log."""
        fid = store.add_fact("log test fact")
        store.soft_delete_fact(fid, reason="test")
        _age_fact(store, fid, 60)
        store.gc({"hard_delete_days": 1})
        events = store.get_event_log(event_type="gc_hard_deleted")
        assert len(events) >= 1
        assert events[0]["metadata"].get("count") == 1

    def test_hard_delete_aged_facts(self, store_with_old_data):
        """Aged soft-deleted facts are cleaned with small threshold."""
        result = store_with_old_data.gc({"hard_delete_days": 1})
        assert result["phases"]["hard_delete"]["deleted"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Orphan Cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestOrphanCleanup:
    def test_orphan_relations_cleaned(self, store):
        """fact_relations pointing to non-existent facts are removed."""
        store.add_fact("test fact")
        # Disable FK enforcement temporarily via execute
        store._conn.execute("PRAGMA foreign_keys=OFF")
        store._conn.execute(
            "INSERT INTO fact_relations (fact_id_a, fact_id_b, relation_type) VALUES (?, ?, ?)",
            (1, 99999, "related"),
        )
        store._conn.execute("PRAGMA foreign_keys=ON")
        result = store.gc()
        assert result["phases"]["orphan_cleanup"]["fact_relations"] >= 1
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM fact_relations"
        ).fetchone()[0]
        assert remaining == 0

    def test_orphan_fact_entities_cleaned(self, store):
        """fact_entities pointing to non-existent facts are removed."""
        fid = store.add_fact("test fact", entities=["some-entity"])
        # Disable FK enforcement temporarily
        store._conn.execute("PRAGMA foreign_keys=OFF")
        store._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
        store._conn.execute("PRAGMA foreign_keys=ON")
        result = store.gc()
        assert result["phases"]["orphan_cleanup"]["fact_entities"] >= 1

    def test_orphan_entities_cleaned(self, store):
        """Entities with no fact_entities are removed."""
        _exec(store._conn, "INSERT INTO entities (name, entity_type) VALUES ('lonely', 'test')")
        result = store.gc()
        assert result["phases"]["orphan_cleanup"]["entities"] >= 1
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE name = 'lonely'"
        ).fetchone()[0]
        assert remaining == 0

    def test_empty_workspace_soft_deleted(self, store):
        """Empty old workspace is soft-deleted."""
        store.create_workspace("orphan_ws")
        _exec(store._conn,
            "UPDATE workspaces SET last_active = datetime('now', '-100 days'), fact_count = 0 WHERE name = 'orphan_ws'"
        )
        result = store.gc()
        assert result["phases"]["orphan_cleanup"]["workspaces"] >= 1
        ws = store.get_workspace("orphan_ws")
        assert ws is None  # soft-deleted, so get_workspace returns None

    def test_active_workspace_not_cleaned(self, store):
        """Workspace with recent activity is not soft-deleted."""
        store.add_fact("active ws fact", project="active_ws")
        result = store.gc()
        ws = store.get_workspace("active_ws")
        assert ws is not None

    def test_dry_run_orphans(self, store):
        """Dry run reports orphan counts without deleting."""
        _exec(store._conn, "INSERT INTO entities (name, entity_type) VALUES ('dry_orphan', 'test')")
        result = store.gc(dry_run=True)
        assert result["phases"]["orphan_cleanup"]["entities"] >= 1
        # Entity should still exist
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE name = 'dry_orphan'"
        ).fetchone()[0]
        assert remaining == 1

    def test_logs_event_on_orphan_cleanup(self, store):
        """gc_orphans_removed event is logged."""
        _exec(store._conn, "INSERT INTO entities (name, entity_type) VALUES ('log_orphan', 'test')")
        store.gc()
        events = store.get_event_log(event_type="gc_orphans_removed")
        assert len(events) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Prune Event Log
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneEventLog:
    def test_old_entries_deleted(self, store):
        """Event log entries older than prune_event_log_days are deleted."""
        _exec(store._conn,
            "INSERT INTO event_log (event_type, created_at) VALUES "
            "('test_old', datetime('now', '-100 days'))"
        )
        result = store.gc({"prune_event_log_days": 30})
        assert result["phases"]["prune_event_log"]["deleted"] >= 1

    def test_recent_entries_kept(self, store):
        """Recent event log entries are preserved."""
        _exec(store._conn,
            "INSERT INTO event_log (event_type, created_at) VALUES "
            "('test_recent', datetime('now', '-1 hour'))"
        )
        result = store.gc({"prune_event_log_days": 30})
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'test_recent'"
        ).fetchone()[0]
        assert remaining == 1

    def test_dry_run_event_log(self, store):
        """Dry run reports event_log count without deleting."""
        _exec(store._conn,
            "INSERT INTO event_log (event_type, created_at) VALUES "
            "('dry_test', datetime('now', '-100 days'))"
        )
        result = store.gc({"prune_event_log_days": 30}, dry_run=True)
        assert result["phases"]["prune_event_log"]["deleted"] >= 1
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'dry_test'"
        ).fetchone()[0]
        assert remaining == 1

    def test_logs_event_on_prune(self, store):
        """gc_event_log_pruned event is logged."""
        _exec(store._conn,
            "INSERT INTO event_log (event_type, created_at) VALUES "
            "('test_log_event', datetime('now', '-100 days'))"
        )
        store.gc({"prune_event_log_days": 30})
        events = store.get_event_log(event_type="gc_event_log_pruned")
        assert len(events) >= 1

    def test_large_threshold_keeps_all(self, store):
        """A very large prune threshold keeps all entries."""
        _exec(store._conn,
            "INSERT INTO event_log (event_type, created_at) VALUES "
            "('large_test', datetime('now', '-1 hour'))"
        )
        result = store.gc({"prune_event_log_days": 9999})
        assert result["phases"]["prune_event_log"]["deleted"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Prune Turn Buffer
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneTurnBuffer:
    def test_old_entries_deleted(self, store):
        """Turn buffer entries older than prune_turn_buffer_days are deleted."""
        _exec(store._conn,
            "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES "
            "('s1', 'user', 'old turn', datetime('now', '-60 days'))"
        )
        result = store.gc({"prune_turn_buffer_days": 7})
        assert result["phases"]["prune_turn_buffer"]["deleted"] >= 1

    def test_recent_entries_kept(self, store):
        """Recent turn buffer entries are preserved."""
        _exec(store._conn,
            "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES "
            "('s1', 'user', 'fresh turn', datetime('now', '-1 hour'))"
        )
        result = store.gc({"prune_turn_buffer_days": 7})
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE content = 'fresh turn'"
        ).fetchone()[0]
        assert remaining == 1

    def test_dry_run_turn_buffer(self, store):
        """Dry run reports turn_buffer count without deleting."""
        _exec(store._conn,
            "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES "
            "('s1', 'user', 'dry turn', datetime('now', '-60 days'))"
        )
        result = store.gc({"prune_turn_buffer_days": 7}, dry_run=True)
        assert result["phases"]["prune_turn_buffer"]["deleted"] >= 1
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE content = 'dry turn'"
        ).fetchone()[0]
        assert remaining == 1

    def test_logs_event_on_prune(self, store):
        """gc_turn_buffer_pruned event is logged."""
        _exec(store._conn,
            "INSERT INTO turn_buffer (session_id, role, content, created_at) VALUES "
            "('s1', 'user', 'log turn', datetime('now', '-60 days'))"
        )
        store.gc({"prune_turn_buffer_days": 7})
        events = store.get_event_log(event_type="gc_turn_buffer_pruned")
        assert len(events) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Snapshot Retention
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRetention:
    def test_old_snapshots_deleted(self, store):
        """Snapshots older than snapshot_max_age_days are deleted."""
        store.create_snapshot("ancient_snap", project="test")
        _exec(store._conn,
            "UPDATE snapshots SET created_at = datetime('now', '-400 days') WHERE name = 'ancient_snap'"
        )
        result = store.gc({"snapshot_max_age_days": 30})
        assert result["phases"]["snapshot_retention"]["deleted"] >= 1
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE name = 'ancient_snap'"
        ).fetchone()[0]
        assert remaining == 0

    def test_keeps_last_n_per_project(self, store):
        """Only the last N snapshots per project are retained."""
        for i in range(5):
            store.create_snapshot(f"proj_snap_{i}", project="myproject")
        result = store.gc({"snapshot_keep": 2, "snapshot_max_age_days": 9999})
        assert result["phases"]["snapshot_retention"]["deleted"] == 3
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE project = 'myproject'"
        ).fetchone()[0]
        assert remaining == 2

    def test_empty_project_treated_as_group(self, store):
        """Snapshots with empty project string are grouped together."""
        for i in range(4):
            store.create_snapshot(f"global_snap_{i}")  # project=''
        result = store.gc({"snapshot_keep": 1, "snapshot_max_age_days": 9999})
        assert result["phases"]["snapshot_retention"]["deleted"] == 3

    def test_dry_run_reports_without_deleting(self, store):
        """Dry run reports snapshot count without deleting."""
        for i in range(3):
            store.create_snapshot(f"dry_snap_{i}", project="dryproj")
        result = store.gc({"snapshot_keep": 1, "snapshot_max_age_days": 9999}, dry_run=True)
        assert result["phases"]["snapshot_retention"]["deleted"] == 2
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE project = 'dryproj'"
        ).fetchone()[0]
        assert remaining == 3  # none deleted

    def test_logs_event_on_snapshot_deletion(self, store):
        """gc_snapshots_removed event is logged."""
        for i in range(3):
            store.create_snapshot(f"log_snap_{i}", project="logproj")
        store.gc({"snapshot_keep": 1, "snapshot_max_age_days": 9999})
        events = store.get_event_log(event_type="gc_snapshots_removed")
        assert len(events) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Vacuum
# ─────────────────────────────────────────────────────────────────────────────

class TestVacuum:
    def test_vacuum_not_triggered_by_default(self, store):
        """Vacuum is not triggered when free pages are below default threshold."""
        result = store.gc()
        assert result["phases"]["vacuum"]["vacuumed"] is False

    def test_vacuum_triggered_when_threshold_met(self, store):
        """Vacuum runs when free page percentage meets threshold."""
        # Wrap connection to intercept PRAGMA queries
        original_conn = store._conn
        mock_conn = MagicMock(spec=original_conn)

        def mock_execute(sql, *args, **kwargs):
            if sql == "PRAGMA freelist_count":
                c = MagicMock()
                c.fetchone.return_value = (50,)
                return c
            if sql == "PRAGMA page_count":
                c = MagicMock()
                c.fetchone.return_value = (100,)
                return c
            if sql == "VACUUM":
                return MagicMock()
            # For everything else, delegate
            return original_conn.execute(sql, *args, **kwargs)

        mock_conn.execute.side_effect = mock_execute
        store._conn = mock_conn

        result = store.gc({"vacuum_threshold_pct": 10})
        store._conn = original_conn  # restore

        assert result["phases"]["vacuum"]["vacuumed"] is True

    def test_vacuum_dry_run_reports_without_vacuuming(self, store):
        """Dry run reports vacuum needed without running VACUUM."""
        original_conn = store._conn
        mock_conn = MagicMock(spec=original_conn)
        vacuum_called = []

        def mock_execute(sql, *args, **kwargs):
            if sql == "PRAGMA freelist_count":
                c = MagicMock()
                c.fetchone.return_value = (50,)
                return c
            if sql == "PRAGMA page_count":
                c = MagicMock()
                c.fetchone.return_value = (100,)
                return c
            if sql == "VACUUM":
                vacuum_called.append(True)
                return MagicMock()
            return original_conn.execute(sql, *args, **kwargs)

        mock_conn.execute.side_effect = mock_execute
        store._conn = mock_conn

        result = store.gc({"vacuum_threshold_pct": 10}, dry_run=True)
        store._conn = original_conn

        assert result["phases"]["vacuum"]["vacuumed"] is True
        assert len(vacuum_called) == 0  # VACUUM was NOT called

    def test_vacuum_not_triggered_with_zero_pages(self, store):
        """No vacuum when page_count is 0."""
        original_conn = store._conn
        mock_conn = MagicMock(spec=original_conn)

        def mock_execute(sql, *args, **kwargs):
            if sql == "PRAGMA freelist_count":
                c = MagicMock()
                c.fetchone.return_value = (0,)
                return c
            if sql == "PRAGMA page_count":
                c = MagicMock()
                c.fetchone.return_value = (0,)
                return c
            return original_conn.execute(sql, *args, **kwargs)

        mock_conn.execute.side_effect = mock_execute
        store._conn = mock_conn

        result = store.gc({"vacuum_threshold_pct": 1})
        store._conn = original_conn

        assert result["phases"]["vacuum"]["vacuumed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Dry Run (cross-phase)
# ─────────────────────────────────────────────────────────────────────────────

class TestGCDryRun:
    def test_dry_run_returns_correct_flag(self, store):
        """Result includes dry_run flag matching input."""
        result = store.gc(dry_run=True)
        assert result["dry_run"] is True
        result2 = store.gc(dry_run=False)
        assert result2["dry_run"] is False

    def test_dry_run_all_phases_report_counts(self, store):
        """All phases report counts in dry run mode."""
        result = store.gc(dry_run=True)
        for phase in ("hard_delete", "orphan_cleanup", "prune_event_log",
                       "prune_turn_buffer", "snapshot_retention", "vacuum"):
            assert phase in result["phases"]
            assert isinstance(result["phases"][phase], dict)

    def test_dry_run_no_data_modified(self, store):
        """No data is modified after a dry run."""
        store.add_fact("test fact for dry run")
        store.create_snapshot("dry_snapshot")
        fact_count_before = store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE deleted = 0"
        ).fetchone()[0]
        snapshot_count_before = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots"
        ).fetchone()[0]

        store.gc(dry_run=True)

        fact_count_after = store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE deleted = 0"
        ).fetchone()[0]
        snapshot_count_after = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots"
        ).fetchone()[0]

        assert fact_count_after == fact_count_before
        assert snapshot_count_after == snapshot_count_before


# ─────────────────────────────────────────────────────────────────────────────
# Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGCIntegration:
    def test_full_gc_returns_all_phases(self, store):
        """Full gc() run returns all 6 phases plus metadata."""
        result = store.gc()
        assert "phases" in result
        assert "duration_ms" in result
        assert "dry_run" in result
        assert result["dry_run"] is False
        expected_phases = [
            "hard_delete", "orphan_cleanup", "prune_event_log",
            "prune_turn_buffer", "snapshot_retention", "vacuum",
        ]
        for phase in expected_phases:
            assert phase in result["phases"]

    def test_gc_includes_timing(self, store):
        """duration_ms is a non-negative integer."""
        result = store.gc()
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0

    def test_gc_with_custom_config(self, store):
        """Custom config overrides defaults."""
        result = store.gc({"hard_delete_days": 0, "vacuum_threshold_pct": 99})
        assert result["phases"]["hard_delete"]["deleted"] == 0
        assert result["phases"]["vacuum"]["vacuumed"] is False

    def test_gc_on_empty_store(self, store):
        """GC runs without error on completely empty store."""
        result = store.gc()
        assert result["phases"]["hard_delete"]["deleted"] == 0
        assert result["phases"]["orphan_cleanup"]["fact_relations"] == 0
        assert result["phases"]["prune_event_log"]["deleted"] == 0
        assert result["phases"]["prune_turn_buffer"]["deleted"] == 0
        assert result["phases"]["snapshot_retention"]["deleted"] == 0
        assert result["phases"]["vacuum"]["vacuumed"] is False

    def test_gc_with_all_phases_active(self, store_with_old_data):
        """Full GC with aggressive settings cleans all stale data."""
        result = store_with_old_data.gc({
            "hard_delete_days": 1,
            "prune_event_log_days": 0,
            "prune_turn_buffer_days": 0,
            "snapshot_keep": 0,
            "snapshot_max_age_days": 0,
            "vacuum_threshold_pct": 1,
        })
        # hard_delete should find all 3 aged soft-deleted facts
        assert result["phases"]["hard_delete"]["deleted"] == 3
        # orphan cleanup should find orphan entity
        assert result["phases"]["orphan_cleanup"]["entities"] >= 1
        # prune_event_log should find old entries
        assert result["phases"]["prune_event_log"]["deleted"] > 0
        # prune_turn_buffer should find old entries
        assert result["phases"]["prune_turn_buffer"]["deleted"] > 0
        # snapshot retention should clean up
        assert result["phases"]["snapshot_retention"]["deleted"] > 0

    def test_gc_idempotent(self, store_with_old_data):
        """Running GC twice produces zero additional cleanup on second pass."""
        cfg = {
            "hard_delete_days": 1,
            "prune_event_log_days": 0,
            "prune_turn_buffer_days": 0,
            "snapshot_keep": 1,
            "snapshot_max_age_days": 0,
        }
        store_with_old_data.gc(cfg)
        second = store_with_old_data.gc(cfg)
        # Second pass should produce zeros
        assert second["phases"]["hard_delete"]["deleted"] == 0
        assert second["phases"]["prune_event_log"]["deleted"] == 0
        assert second["phases"]["prune_turn_buffer"]["deleted"] == 0

    def test_return_format_matches_spec(self, store):
        """Return dict follows the specified format exactly."""
        result = store.gc()
        assert isinstance(result, dict)
        assert isinstance(result["phases"], dict)
        assert isinstance(result["duration_ms"], int)
        assert isinstance(result["dry_run"], bool)
        # Each phase has expected keys
        assert set(result["phases"]["hard_delete"].keys()) == {"deleted"}
        assert set(result["phases"]["orphan_cleanup"].keys()) == {
            "fact_relations", "fact_entities", "entities", "workspaces"
        }
        assert set(result["phases"]["prune_event_log"].keys()) == {"deleted"}
        assert set(result["phases"]["prune_turn_buffer"].keys()) == {"deleted"}
        assert set(result["phases"]["snapshot_retention"].keys()) == {"deleted"}
        assert set(result["phases"]["vacuum"].keys()) == {"vacuumed"}

    def test_gc_config_override_merges_with_defaults(self, store):
        """Partial config overrides merge with defaults."""
        result = store.gc({"hard_delete_days": 0})
        # hard_delete_days is overridden
        # All other values should use defaults
        assert result["phases"]["prune_event_log"]["deleted"] == 0  # default 90d, no old entries
        assert result["phases"]["vacuum"]["vacuumed"] is False   # default 20%


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestGCEdgeCases:
    def test_gc_with_none_config(self, store):
        """Passing None for config uses defaults."""
        result = store.gc(config=None)
        assert result["phases"]["hard_delete"]["deleted"] == 0

    def test_gc_orphan_cleanup_no_side_effects(self, store):
        """Orphan cleanup doesn't remove valid data."""
        fid_a = store.add_fact("fact A", entities=["entity-a"])
        fid_b = store.add_fact("fact B", entities=["entity-b"])
        store.add_relation(fid_a, fid_b, "related")
        result = store.gc()
        assert result["phases"]["orphan_cleanup"]["fact_relations"] == 0
        assert result["phases"]["orphan_cleanup"]["fact_entities"] == 0
        assert result["phases"]["orphan_cleanup"]["entities"] == 0

    def test_gc_snapshot_retention_multiple_projects(self, store):
        """Multiple project groups each get independent retention."""
        for i in range(4):
            store.create_snapshot(f"proj_a_{i}", project="project_a")
        for i in range(6):
            store.create_snapshot(f"proj_b_{i}", project="project_b")
        result = store.gc({"snapshot_keep": 2, "snapshot_max_age_days": 9999})
        # project_a: 4 - 2 = 2 deleted
        # project_b: 6 - 2 = 4 deleted
        # Total: 6
        assert result["phases"]["snapshot_retention"]["deleted"] == 6

    def test_gc_snapshot_retention_below_keep(self, store):
        """No snapshots deleted when count is below keep threshold."""
        store.create_snapshot("only_one")
        result = store.gc({"snapshot_keep": 5, "snapshot_max_age_days": 9999})
        assert result["phases"]["snapshot_retention"]["deleted"] == 0
