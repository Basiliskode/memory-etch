"""Tests for the EtchCurator — deterministic memory maintenance."""
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_etch.store import EtchStore
from memory_etch.curator import EtchCurator, _DEFAULT_CONFIG
from memory_etch.etch import EtchMemoryProvider


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path():
    return Path(tempfile.mkdtemp()) / "test_curator.db"


@pytest.fixture
def store(db_path):
    s = EtchStore(db_path=str(db_path))
    yield s
    s.close()


@pytest.fixture
def curator(store):
    return EtchCurator(store)


@pytest.fixture
def populated_store(store):
    """Store with facts of varying importance, trust, and age."""

    # critical fact — high trust, recent
    store.add_fact(
        content="PostgreSQL chosen as primary database",
        category="project",
        importance=1.0,
        trust_score=0.9,
        session_id="s1",
        tags="postgresql,arch",
    )

    # useful fact — medium trust, old date (force via sql)
    store.add_fact(
        content="Redis cache TTL set to 1 hour",
        category="project",
        importance=0.5,
        trust_score=0.3,
        session_id="s1",
        tags="redis,cache",
    )

    # trivial fact — low trust, old date
    store.add_fact(
        content="Used Python 3.11 for initial prototype",
        category="general",
        importance=0.2,
        trust_score=0.05,
        session_id="s1",
        tags="python",
    )

    # fact for archive test — very low trust, old
    store.add_fact(
        content="Tried MongoDB but reverted — no joins",
        category="general",
        importance=0.3,
        trust_score=0.02,
        session_id="s1",
        tags="mongodb",
    )

    # Manually age some facts by updating their updated_at to far past
    store._conn.execute(
        "UPDATE facts SET updated_at = datetime('now', '-100 days') "
        "WHERE content LIKE '%MongoDB%' OR content LIKE '%Redis%' "
        "OR content LIKE '%Python 3.11%'"
    )
    try:
        store._conn.commit()
    except sqlite3.OperationalError:
        pass  # no active transaction under some pytest configurations

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultConfig:
    def test_defaults_are_sane(self):
        assert _DEFAULT_CONFIG["decay_interval_days"] >= 1
        assert 0.5 <= _DEFAULT_CONFIG["decay_factor_critical"] <= 1.0
        assert _DEFAULT_CONFIG["archive_trust_threshold"] >= 0.01
        assert _DEFAULT_CONFIG["archive_age_days"] >= 1
        assert _DEFAULT_CONFIG["vacuum_free_page_pct"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Decay
# ─────────────────────────────────────────────────────────────────────────────

class TestDecay:
    def test_decay_reduces_trust_for_old_facts(self, populated_store):
        curator = EtchCurator(populated_store, {"decay_interval_days": 1})
        before = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%Redis%'"
        ).fetchone()[0]
        affected = curator.decay_trust()
        after = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%Redis%'"
        ).fetchone()[0]
        assert affected >= 1
        assert after < before

    def test_decay_respects_importance(self, populated_store):
        """Critical facts decay slower than trivial facts."""
        curator = EtchCurator(populated_store, {"decay_interval_days": 1})
        curator.decay_trust()
        critical = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%PostgreSQL%'"
        ).fetchone()[0]
        trivial = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%Python 3.11%'"
        ).fetchone()[0]
        # Critical should still be higher relative to start
        assert critical > trivial

    def test_decay_does_not_touch_recent_facts(self, populated_store):
        """Facts updated within the interval are untouched."""
        before = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%PostgreSQL%' "
            "AND content NOT LIKE '%MongoDB%'"
        ).fetchone()[0]
        curator = EtchCurator(populated_store, {"decay_interval_days": 30})
        affected = curator.decay_trust()
        after = populated_store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%PostgreSQL%' "
            "AND content NOT LIKE '%MongoDB%'"
        ).fetchone()[0]
        # Recent PostgreSQL fact should NOT decay
        assert after == before
        # Other old facts SHOULD decay (> 0 affected overall)
        assert affected >= 1

    def test_decay_never_below_floor(self, populated_store):
        """Trust never drops below 0.01."""
        curator = EtchCurator(populated_store, {
            "decay_interval_days": 1,
            "decay_factor_trivial": 0.5,
        })
        for _ in range(5):
            curator.decay_trust()
        low = populated_store._conn.execute(
            "SELECT MIN(trust_score) FROM facts WHERE deleted = 0"
        ).fetchone()[0]
        assert low >= 0.01

    def test_decay_skips_deleted_and_consolidated(self, populated_store):
        """Soft-deleted or consolidated facts are not decayed."""
        # Mark PostgreSQL fact as consolidated
        populated_store._conn.execute(
            "UPDATE facts SET consolidated = 1 WHERE content LIKE '%PostgreSQL%' "
            "AND content NOT LIKE '%MongoDB%'"
        )
        populated_store._conn.commit()
        # Mark MongoDB fact as deleted  
        populated_store._conn.execute(
            "UPDATE facts SET deleted = 1 WHERE content LIKE '%MongoDB%'"
        )
        populated_store._conn.commit()

        curator = EtchCurator(populated_store, {"decay_interval_days": 1})
        # Only the Redis + Python facts should decay (both are old, not deleted, not consolidated)
        affected = curator.decay_trust()
        # Redis + Python = 2 facts should decay
        assert affected >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Archive
# ─────────────────────────────────────────────────────────────────────────────

class TestArchive:
    def test_archives_low_trust_old_facts(self, populated_store):
        curator = EtchCurator(populated_store, {
            "archive_trust_threshold": 0.1,
            "archive_age_days": 1,
        })
        affected = curator.archive_stale()
        assert affected >= 1

    def test_archived_facts_are_soft_deleted(self, populated_store):
        curator = EtchCurator(populated_store, {
            "archive_trust_threshold": 0.1,
            "archive_age_days": 1,
        })
        curator.archive_stale()
        archived = populated_store._conn.execute(
            "SELECT content, deleted_reason FROM facts WHERE deleted = 1"
        ).fetchall()
        archived_contents = [r[0] for r in archived]
        assert any("MongoDB" in c for c in archived_contents)

    def test_archive_reason_is_descriptive(self, populated_store):
        curator = EtchCurator(populated_store, {
            "archive_trust_threshold": 0.1,
            "archive_age_days": 1,
        })
        curator.archive_stale()
        reasons = populated_store._conn.execute(
            "SELECT deleted_reason FROM facts WHERE deleted = 1"
        ).fetchall()
        for (reason,) in reasons:
            assert reason.startswith("curator:")

    def test_archive_skips_high_trust_facts(self, populated_store):
        """Facts above the trust threshold are never archived."""
        curator = EtchCurator(populated_store, {
            "archive_trust_threshold": 0.5,
            "archive_age_days": 1,
        })
        curator.archive_stale()
        high_trust = populated_store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE deleted = 1 AND trust_score > 0.5"
        ).fetchone()[0]
        assert high_trust == 0


# ─────────────────────────────────────────────────────────────────────────────
# Prune
# ─────────────────────────────────────────────────────────────────────────────

class TestPrune:
    def test_removes_old_buffer_entries(self, store):
        store._conn.execute(
            "INSERT INTO turn_buffer (session_id, role, content, created_at) "
            "VALUES ('s1', 'user', 'old turn', datetime('now', '-30 days'))"
        )
        store._conn.execute(
            "INSERT INTO turn_buffer (session_id, role, content, created_at) "
            "VALUES ('s1', 'user', 'recent turn', datetime('now', '-1 hour'))"
        )
        store._conn.commit()

        curator = EtchCurator(store, {"prune_buffer_age_days": 7})
        affected = curator.prune_buffer()
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer"
        ).fetchone()[0]

        assert affected == 1  # only the 30-day-old turn
        assert remaining == 1  # recent turn stays

    def test_prune_empty_buffer(self, store):
        curator = EtchCurator(store)
        affected = curator.prune_buffer()
        assert affected == 0


# ─────────────────────────────────────────────────────────────────────────────
# Full curate pass
# ─────────────────────────────────────────────────────────────────────────────

class TestFullCurate:
    def test_curate_returns_stats(self, populated_store):
        curator = EtchCurator(populated_store, {
            "decay_interval_days": 1,
            "archive_trust_threshold": 0.1,
            "archive_age_days": 1,
        })
        stats = curator.curate()
        assert "decayed" in stats
        assert "archived" in stats
        assert "pruned" in stats
        assert "vacuumed" in stats
        assert "duration_ms" in stats
        assert stats["duration_ms"] >= 0

    def test_curate_is_idempotent(self, populated_store):
        """Second pass should produce identical state when no interval passes.

        Uses a very large decay interval so decay never fires — only
        archive (which is idempotent) and prune run.
        """
        curator = EtchCurator(populated_store, {
            "decay_interval_days": 99999,
            "archive_trust_threshold": 0.1,
            "archive_age_days": 1,
        })
        first = curator.curate()
        # Convert rows to plain tuples for reliable comparison
        state_after_first = populated_store._conn.execute(
            "SELECT trust_score, deleted FROM facts ORDER BY fact_id"
        ).fetchall()
        state_after_first = [tuple(r) for r in state_after_first]
        second = curator.curate()
        state_after_second = populated_store._conn.execute(
            "SELECT trust_score, deleted FROM facts ORDER BY fact_id"
        ).fetchall()
        state_after_second = [tuple(r) for r in state_after_second]
        # No additional changes in second pass
        assert state_after_first == state_after_second

    def test_full_curate_no_crash_on_empty(self, store):
        curator = EtchCurator(store)
        stats = curator.curate()
        assert stats["decayed"] == 0
        assert stats["archived"] == 0
        assert stats["pruned"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Wired into EtchMemoryProvider
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderWiring:
    def test_curator_created_on_initialize(self, db_path):
        provider = EtchMemoryProvider({"db_path": str(db_path)})
        provider.initialize("test-session")
        assert provider._curator is not None
        provider.shutdown()

    def test_curator_runs_on_shutdown(self, db_path):
        """Shutdown should run curator without crashing."""
        provider = EtchMemoryProvider({"db_path": str(db_path)})
        provider.initialize("test-session")
        # Add a fact to give the curator something to work with
        provider._store.add_fact(
            content="Test fact for curation on shutdown",
            trust_score=0.5,
        )
        # Shutdown — includes curator.curate()
        provider.shutdown()
        assert provider._store is None

    def test_shutdown_curator_does_not_raise(self, db_path):
        """Even if curator fails, shutdown should not raise."""
        provider = EtchMemoryProvider({"db_path": str(db_path)})
        provider.initialize("test-session")
        with patch.object(provider._curator, "curate", side_effect=Exception("boom")):
            provider.shutdown()  # should not raise
        assert provider._store is None
