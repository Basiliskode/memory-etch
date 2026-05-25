"""Tests for Phase 2 Data Quality: content hash dedup and conflict surfacing."""

import hashlib

import pytest

from memory_etch import EtchStore


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


def _content_hash(content: str, project: str = "", scope: str = "canonical") -> str:
    """Compute the same content_hash the store would."""
    return hashlib.sha256(
        content.encode() + project.encode() + scope.encode()
    ).hexdigest()


class TestContentHashDedup:
    """Task 2.1 — Content hash dedup (lifetime).

    The content_hash mechanism provides LIFETIME dedup and duplicate_count
    tracking within the same project and scope.
    """

    def test_same_content_returns_existing_id(self, store):
        """Two identical content+project returns same ID."""
        fid1 = store.add_fact("Dedup test content", project="test-proj")
        fid2 = store.add_fact("Dedup test content", project="test-proj")
        assert fid2 == fid1, "Same content should return existing ID"

    def test_duplicate_count_incremented_on_dedup(self, store):
        """Dedup hit increments duplicate_count."""
        fid1 = store.add_fact("Dedup count test", project="proj-a")
        fid2 = store.add_fact("Dedup count test", project="proj-a")
        assert fid2 == fid1
        fact = store.get_fact(fid2)
        assert fact["duplicate_count"] >= 1, (
            f"Expected duplicate_count >= 1, got {fact['duplicate_count']}"
        )

    def test_duplicate_count_multiple_hits(self, store):
        """Multiple dedup hits accumulate duplicate_count."""
        fid1 = store.add_fact("Multiple dedup test", project="proj-b")
        fid_ref = fid1
        for _ in range(3):
            fid = store.add_fact("Multiple dedup test", project="proj-b")
            assert fid == fid_ref
        fact = store.get_fact(fid_ref)
        assert fact["duplicate_count"] >= 3, (
            f"Expected duplicate_count >= 3, got {fact['duplicate_count']}"
        )

    def test_different_content_normal_insert(self, store):
        """Different content → they get different fact_ids."""
        fid1 = store.add_fact("Content alpha", project="proj")
        fid2 = store.add_fact("Content beta", project="proj")
        assert fid2 != fid1
        assert fid2 > fid1  # autoincrement

    def test_same_content_different_project_does_not_increment_duplicate(self, store):
        """Same content, different project: content_hash is different,
        so a distinct fact is created and duplicate_count stays at 0."""
        fid1 = store.add_fact("Shared content", project="proj-x")
        fid2 = store.add_fact("Shared content", project="proj-y")
        assert fid2 != fid1
        fact = store.get_fact(fid1)
        assert fact["duplicate_count"] == 0, (
            f"Expected duplicate_count=0 for different projects, "
            f"got {fact['duplicate_count']}"
        )

    def test_same_content_no_project_dedup(self, store):
        """Same content with no project → dedup works (empty string project)."""
        fid1 = store.add_fact("No project dedup")
        fid2 = store.add_fact("No project dedup")
        assert fid2 == fid1

    def test_content_hash_column_stored_on_insert(self, store):
        """New facts have content_hash populated."""
        fid = store.add_fact("Hash column test", project="hash-proj")
        fact = store.get_fact(fid)
        expected_hash = _content_hash("Hash column test", "hash-proj")
        assert fact["content_hash"] == expected_hash, (
            f"Expected content_hash={expected_hash}, got {fact['content_hash']}"
        )

    def test_content_hash_also_set_on_topic_upsert(self, store):
        """Topic upsert also sets content_hash."""
        fid1 = store.add_fact(
            "Topic hash v1",
            topic_key="topic:hash-test",
            project="hash-upsert",
        )
        fact = store.get_fact(fid1)
        expected = _content_hash("Topic hash v1", "hash-upsert")
        assert fact["content_hash"] == expected

        fid2 = store.add_fact(
            "Topic hash v2",
            topic_key="topic:hash-test",
            project="hash-upsert",
        )
        assert fid2 == fid1
        fact = store.get_fact(fid2)
        expected2 = _content_hash("Topic hash v2", "hash-upsert")
        assert fact["content_hash"] == expected2, (
            "Topic upsert should update content_hash"
        )


class TestDedupLifetime:
    """Lifetime dedup behavior — no time window."""

    def test_dedup_increments_count(self, store):
        """Same content+project: dedup, increments duplicate_count."""
        fid1 = store.add_fact("Lifetime dedup test", project="lt-proj")
        fid2 = store.add_fact("Lifetime dedup test", project="lt-proj")
        assert fid2 == fid1
        fact = store.get_fact(fid1)
        assert fact["duplicate_count"] >= 1

    def test_old_content_still_deduped(self, store):
        """Same content+project, even if created long ago: dedup still works."""
        fid1 = store.add_fact("Old content dedup", project="old-proj")

        # Push the fact back by 120 seconds
        with store._lock:
            store._conn.execute(
                "UPDATE facts SET created_at = datetime('now', '-120 seconds') WHERE fact_id = ?",
                (fid1,),
            )
            store._conn.commit()

        # Second add — content_hash dedup hits regardless of age
        fid2 = store.add_fact("Old content dedup", project="old-proj")
        assert fid2 == fid1
        fact = store.get_fact(fid1)
        assert fact["duplicate_count"] >= 1, (
            f"Expected duplicate_count >= 1 (lifetime dedup), "
            f"got {fact['duplicate_count']}"
        )


class TestTopicUpsertRevisionCount:
    """Topic upsert behavior — safety net + extras."""

    def test_topic_upsert_increments_revision_count(self, store):
        """Topic upsert increments revision_count on existing fact."""
        fid1 = store.add_fact(
            "Original topic content",
            topic_key="topic:my-topic",
            project="rev-proj",
        )
        fid2 = store.add_fact(
            "Updated topic content",
            topic_key="topic:my-topic",
            project="rev-proj",
        )
        assert fid2 == fid1, "Topic upsert should reuse same fact_id"
        fact = store.get_fact(fid2)
        assert fact["revision_count"] >= 1, (
            f"Expected revision_count >= 1, got {fact['revision_count']}"
        )

    def test_topic_upsert_not_affected_by_content_dedup(self, store):
        """Topic upsert with different content but same key works independently."""
        fid1 = store.add_fact("Topic first version", topic_key="topic:v2")
        fid2 = store.add_fact(
            "Topic second version (different)",
            topic_key="topic:v2",
        )
        assert fid2 == fid1  # upsert
        fact = store.get_fact(fid1)
        assert fact["revision_count"] >= 1


class TestConflictSurfacing:
    """Task 2.2 — Conflict surfacing via return_metadata."""

    def test_conflict_detected_on_very_similar_content(self, store):
        """Very similar content returns conflicts_with via return_metadata."""
        fid1 = store.add_fact(
            "Python is a dynamically typed programming language",
            project="conflict-proj",
        )
        result = store.add_fact(
            "Python is a dynamically typed programming language used widely",
            project="conflict-proj",
            return_metadata=True,
        )
        assert isinstance(result, dict), "return_metadata=True should return dict"
        assert result["id"] > 0
        # Should detect conflict either by BM25 or topic_key match
        assert "conflicts_with" in result, (
            f"Expected conflicts_with in result, got keys: {result.keys()}"
        )
        assert any(c["id"] == fid1 for c in result["conflicts_with"]), (
            "conflicts_with should include the original fact"
        )

    def test_no_conflict_on_unique_content(self, store):
        """Unique content returns no conflicts_with."""
        store.add_fact("Something about databases", project="no-conflict")
        result = store.add_fact(
            "Quantum physics is fascinating",
            project="no-conflict",
            return_metadata=True,
        )
        assert isinstance(result, dict)
        conflicts = result.get("conflicts_with", [])
        assert len(conflicts) == 0, (
            f"Expected no conflicts, got {len(conflicts)}"
        )

    def test_conflict_by_topic_key(self, store):
        """Same topic_key triggers conflict surfacing."""
        store.add_fact(
            "Dark mode is preferred at night",
            project="topic-conflict",
            topic_key="topic:theme",
        )
        result = store.add_fact(
            "Dark mode helps with eye strain",
            project="topic-conflict",
            topic_key="topic:theme",
            return_metadata=True,
        )
        assert isinstance(result, dict)
        # Both have same topic_key — should detect conflict
        assert "conflicts_with" in result, (
            "Same topic_key should trigger conflicts_with"
        )

    def test_return_metadata_backward_compat(self, store):
        """return_metadata=False returns int (backward compat)."""
        fid = store.add_fact("Backward compat test")
        assert isinstance(fid, int), "Default return should be int"

    def test_return_metadata_default_is_false(self, store):
        """Default return_metadata=False preserves backward compat."""
        fid = store.add_fact("Default return test", project="default-ret")
        assert isinstance(fid, int)

    def test_status_created_on_new_fact(self, store):
        """return_metadata returns status='created' for new facts."""
        result = store.add_fact(
            "Brand new unique fact for status test",
            project="status-test",
            return_metadata=True,
        )
        assert result["status"] == "created"

    def test_status_created_on_unique_insert(self, store):
        """Unique insert returns status='created' and no conflicts."""
        result = store.add_fact(
            "Totally unique content for this test run",
            project="unique-status",
            return_metadata=True,
        )
        assert result["status"] == "created"
        assert len(result.get("conflicts_with", [])) == 0


class TestContentHashIndex:
    """Index exists for fast lookups."""

    def test_content_hash_index_exists(self, store):
        """Verify the content_hash index is present."""
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_facts_content_hash'"
        ).fetchall()
        assert len(indexes) == 1, "idx_facts_content_hash index should exist"

    def test_content_hash_and_duplicate_count_columns_exist(self, store):
        """Verify content_hash and duplicate_count columns exist."""
        cols = {
            r["name"]
            for r in store._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        assert "content_hash" in cols, "content_hash column should exist"
        assert "duplicate_count" in cols, "duplicate_count column should exist"
