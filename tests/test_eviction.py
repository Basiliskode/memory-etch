"""Tests for auto-eviction: last_retrieved_at tracking and evict_stale."""

from pathlib import Path

import pytest

from memento.store import EtchStore  # noqa: I001

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return Path(tmp_path) / "test_eviction.db"


@pytest.fixture
def store(db_path):
    s = EtchStore(str(db_path), auto_migrate=True)
    yield s
    s.close()


@pytest.fixture
def store_with_facts(store):
    """Store with facts of varying trust and age."""
    # High-trust recent fact
    f1 = store.add_fact(
        "Active high-trust fact",
        trust_score=0.9,
        session_id="s1",
    )
    # Low-trust recent fact
    f2 = store.add_fact(
        "Recent low-trust fact",
        trust_score=0.05,
        session_id="s1",
    )
    # Low-trust old fact (age via SQL)
    f3 = store.add_fact(
        "Old low-trust fact",
        trust_score=0.05,
        session_id="s1",
    )
    # Medium-trust old fact
    f4 = store.add_fact(
        "Old medium-trust fact",
        trust_score=0.3,
        session_id="s1",
    )
    return store, {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


# ---------------------------------------------------------------------------
# last_retrieved_at column
# ---------------------------------------------------------------------------

class TestLastRetrievedAtColumn:
    """The last_retrieved_at column must exist after migration."""

    def test_column_exists_in_schema(self, store):
        cols = {
            r["name"]
            for r in store._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        assert "last_retrieved_at" in cols


# ---------------------------------------------------------------------------
# evict_stale
# ---------------------------------------------------------------------------

class TestEvictStale:
    """evict_stale should soft-delete facts with low trust and old retrieval age."""

    def test_evicts_low_trust_old_fact(self, store):
        """Fact with trust < 0.1 and last_retrieved > 30 days ago is evicted."""
        fid = store.add_fact("Stale low-trust fact", trust_score=0.05)
        # Manually set last_retrieved_at to far past (hold lock to serialize
        # with the background HRR flush thread on the same connection).
        with store._lock:
            store._conn.execute(
                "UPDATE facts SET last_retrieved_at = datetime('now', '-45 days') WHERE fact_id = ?",
                (fid,),
            )
            store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count >= 1
        # Fact is now soft-deleted
        fact = store.get_fact(fid)
        assert fact["deleted"] == 1

    def test_keeps_high_trust_old_fact(self, store):
        """Fact with trust >= min_trust is NOT evicted."""
        fid = store.add_fact("High trust old fact", trust_score=0.5)
        with store._lock:
            store._conn.execute(
                "UPDATE facts SET last_retrieved_at = datetime('now', '-45 days') WHERE fact_id = ?",
                (fid,),
            )
            store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count == 0
        fact = store.get_fact(fid)
        assert fact["deleted"] == 0

    def test_keeps_low_trust_recently_retrieved(self, store):
        """Fact with low trust but recent retrieval is NOT evicted."""
        fid = store.add_fact("Recently retrieved low-trust", trust_score=0.05)
        with store._lock:
            store._conn.execute(
                "UPDATE facts SET last_retrieved_at = datetime('now', '-2 days') WHERE fact_id = ?",
                (fid,),
            )
            store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count == 0

    def test_handles_never_retrieved_facts(self, store):
        """Facts with last_retrieved_at IS NULL and created > 7d ago are evicted."""
        fid = store.add_fact("Never retrieved old fact", trust_score=0.05)
        # Force created_at to old date — hold lock to serialize with
        # the background HRR flush thread on the same connection.
        with store._lock:
            store._conn.execute(
                "UPDATE facts SET created_at = datetime('now', '-14 days') WHERE fact_id = ?",
                (fid,),
            )
            store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count >= 1
        fact = store.get_fact(fid)
        assert fact["deleted"] == 1

    def test_skips_never_retrieved_recent_facts(self, store):
        """Facts with last_retrieved_at IS NULL and created < 7d ago are kept."""
        fid = store.add_fact("Recently created never retrieved", trust_score=0.05)
        # created_at is default (now), so it's recent
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count == 0
        fact = store.get_fact(fid)
        assert fact["deleted"] == 0

    def test_returns_count(self, store):
        """evict_stale returns the number of evicted facts."""
        with store._lock:
            for i in range(3):
                fid = store.add_fact(f"Stale fact {i}", trust_score=0.05)
                store._conn.execute(
                    "UPDATE facts SET last_retrieved_at = datetime('now', '-60 days') WHERE fact_id = ?",  # noqa: E501
                    (fid,),
                )
            store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count == 3

    def test_skips_already_deleted_facts(self, store):
        """Already soft-deleted facts are not evicted again."""
        fid = store.add_fact("Already deleted", trust_score=0.05)
        store.soft_delete_fact(fid, reason="manual")
        store._conn.execute(
            "UPDATE facts SET last_retrieved_at = datetime('now', '-60 days') WHERE fact_id = ?",
            (fid,),
        )
        store._conn.commit()
        count = store.evict_stale(min_trust=0.1, max_days=30)
        assert count == 0
