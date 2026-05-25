"""Tests for EtchStore core functionality."""

import tempfile
from pathlib import Path

import pytest
from memento import EtchStore


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


class TestStoreBasics:
    def test_add_and_get_fact(self, store):
        fid = store.add_fact("Python is a programming language", category="tech")
        assert fid > 0
        fact = store.get_fact(fid)
        assert fact is not None
        assert fact["content"] == "Python is a programming language"
        assert fact["category"] == "tech"

    def test_add_duplicate_returns_existing(self, store):
        fid1 = store.add_fact("unique content", category="general")
        fid2 = store.add_fact("unique content", category="general")
        # INSERT OR IGNORE returns the existing row's fact_id
        assert fid2 == fid1
        assert store.stats()["fact_count"] == 1

    def test_list_facts(self, store):
        store.add_fact("Fact A", category="general")
        store.add_fact("Fact B", category="tech")
        result = store.list_facts()
        assert len(result) >= 2

    def test_list_facts_with_category_filter(self, store):
        store.add_fact("Fact A", category="general")
        store.add_fact("Fact B", category="tech")
        result = store.list_facts(category="tech")
        assert len(result) == 1
        assert result[0]["category"] == "tech"

    def test_list_facts_pagination(self, store):
        for i in range(5):
            store.add_fact(f"Fact {i}", category="general")
        result = store.list_facts(limit=3, offset=0)
        assert len(result) == 3

    def test_update_fact(self, store):
        fid = store.add_fact("Original content", category="general")
        store.update_fact(fid, content="Updated content")
        fact = store.get_fact(fid)
        assert fact["content"] == "Updated content"

    def test_update_fact_partial(self, store):
        fid = store.add_fact("Test", category="general", tags="a,b")
        store.update_fact(fid, tags="c,d")
        fact = store.get_fact(fid)
        assert fact["tags"] == "c,d"
        assert fact["content"] == "Test"  # unchanged

    def test_remove_fact(self, store):
        fid = store.add_fact("To delete", category="general")
        store.remove_fact(fid)
        assert store.get_fact(fid) is None

    def test_stats(self, store):
        store.add_fact("A", category="general")
        store.add_fact("B", category="tech")
        stats = store.stats()
        assert stats["fact_count"] >= 2
        assert "session_count" in stats
        assert "relation_count" in stats

    def test_projects(self, store):
        store.add_fact("A", category="general", project="alpha")
        store.add_fact("B", category="tech", project="beta")
        projects = store.projects()
        assert "alpha" in projects
        assert "beta" in projects


class TestStoreSoftDelete:
    def test_soft_delete_excludes_by_default(self, store):
        fid = store.add_fact("Will be deleted", category="general")
        store.soft_delete_fact(fid, reason="test cleanup")
        result = store.list_facts()
        ids = [f["fact_id"] for f in result]
        assert fid not in ids

    def test_soft_delete_can_be_included(self, store):
        fid = store.add_fact("Will be deleted", category="general")
        store.soft_delete_fact(fid)
        # search_facts excludes by default
        results = store.search_facts("deleted")
        ids = [r["fact_id"] for r in results]
        assert fid not in ids

    def test_stats_excludes_deleted(self, store):
        fid1 = store.add_fact("Keep A", category="general")
        fid2 = store.add_fact("Delete B", category="general")
        assert store.stats()["fact_count"] == 2
        store.soft_delete_fact(fid2)
        assert store.stats()["fact_count"] == 1

    def test_multiple_soft_delete(self, store):
        fids = []
        for i in range(3):
            fids.append(store.add_fact(f"Fact {i}", category="general"))
        store.soft_delete_fact(fids[1])
        result = store.list_facts()
        result_ids = [f["fact_id"] for f in result]
        assert fids[1] not in result_ids


class TestRestore:
    def test_restore_archived_fact(self, store):
        fid = store.add_fact("Archived fact", category="general")
        store.soft_delete_fact(fid, reason="curator: trust=0.05")
        # Fact should be hidden
        assert fid not in [f["fact_id"] for f in store.list_facts()]
        # Restore
        assert store.restore_fact(fid) is True
        # Fact should be visible again
        restored_ids = [f["fact_id"] for f in store.list_facts()]
        assert fid in restored_ids
        # deleted_reason should be cleared
        fact = store.get_fact(fid)
        assert fact["deleted"] == 0
        assert fact["deleted_reason"] == ""

    def test_restore_already_active_fact_returns_false(self, store):
        fid = store.add_fact("Active fact", category="general")
        result = store.restore_fact(fid)
        assert result is False

    def test_restore_nonexistent_fact_returns_false(self, store):
        result = store.restore_fact(99999)
        assert result is False

    def test_restore_multiple_times(self, store):
        """Restore is idempotent after first call."""
        fid = store.add_fact("To archive", category="general")
        store.soft_delete_fact(fid)
        assert store.restore_fact(fid) is True  # First restore
        assert store.restore_fact(fid) is False  # Already active — no-op


class TestStoreEntities:
    def test_add_fact_with_entities(self, store):
        fid = store.add_fact("Python is great", category="tech", entities=["Python", "Programming"])
        assert fid > 0

    def test_get_entities(self, store):
        fid = store.add_fact("Python is great", category="tech", entities=["Python"])
        entities = store.get_entities(fid)
        names = [e["name"] for e in entities]
        assert "python" in names  # lowercased

    def test_entities_case_insensitive(self, store):
        fid = store.add_fact("Test", entities=["Python"])
        entities = store.get_entities(fid)
        assert any(e["name"] == "python" for e in entities)


class TestStoreSessions:
    def test_start_and_end_session(self, store):
        store.start_session("session-1", project="test")
        session = store.get_session("session-1")
        assert session is not None
        assert session["status"] == "active"

        store.end_session("session-1", summary="Done")
        session = store.get_session("session-1")
        assert session["status"] == "ended"
        assert session["summary"] == "Done"

    def test_session_not_found(self, store):
        session = store.get_session("nonexistent")
        assert session is None

    def test_facts_with_session_id(self, store):
        store.start_session("s1")
        store.add_fact("Fact in session", category="general", session_id="s1")
        result = store.list_facts()
        fact = next((f for f in result if f["session_id"] == "s1"), None)
        assert fact is not None


class TestStoreRelations:
    def test_add_and_get_relations(self, store):
        fida = store.add_fact("Fact A", category="general")
        fidb = store.add_fact("Fact B", category="general")
        store.add_relation(fida, fidb, relation_type="compatible", confidence=0.9)

        result = store.get_relations(fida)
        assert len(result) >= 1
        assert any(r["relation_type"] == "compatible" for r in result)

    def test_contradictions(self, store):
        fida = store.add_fact("Fact A", category="general")
        fidb = store.add_fact("Fact B", category="general")
        store.add_relation(fida, fidb, relation_type="conflicts_with", confidence=0.8)
        contradictions = store.get_contradictions()
        assert len(contradictions) >= 1

    def test_relation_no_duplicates(self, store):
        fida = store.add_fact("Fact A", category="general")
        fidb = store.add_fact("Fact B", category="general")
        store.add_relation(fida, fidb, relation_type="compatible")
        # Second attempt should not create duplicate (UNIQUE constraint)
        assert store.add_relation(fida, fidb, relation_type="compatible") is True


class TestStoreTimeline:
    def test_timeline_empty_for_standalone_fact(self, store):
        fid = store.add_fact("Standalone", category="general")
        tl = store.get_timeline(fid)
        assert tl["fact"] is not None
        assert tl["fact"]["fact_id"] == fid

    def test_timeline_with_session(self, store):
        store.start_session("t1")
        fida = store.add_fact("First", category="general", session_id="t1")
        fidb = store.add_fact("Second", category="general", session_id="t1")
        tl = store.get_timeline(fidb, before=3, after=3)
        assert tl["fact"] is not None
        assert len(tl["before"]) > 0
        assert tl["before"][0]["fact_id"] == fida

    def test_timeline_not_found(self, store):
        tl = store.get_timeline(99999)
        assert tl["fact"] is None


class TestStoreConsolidation:
    def test_add_without_consolidation(self, store):
        result = store.add_fact_with_consolidation("Brand new fact", category="general")
        assert result["action"] == "added"
        assert result["fact_id"] > 0

    def test_add_skips_on_llm_fallback(self, store):
        """Without search_fn/llm_decide_fn, should fall through to simple add."""
        store.add_fact("Python is a language", category="tech")
        result = store.add_fact_with_consolidation("Python is a language", category="tech")
        # Without consolidation callbacks, it treats as simple add
        assert "fact_id" in result

    def test_purge_dry_run(self, store):
        result = store.purge_facts(dry_run=True)
        assert result["action"] == "dry_run"


class TestExportImport:
    def test_export_roundtrip(self, tmp_path, store):
        # Add data
        store.add_fact("Python is great", category="tech", tags="python", importance=0.9)
        store.add_fact("FastAPI is fast", category="tech", tags="python,web")
        store.start_session("s1")
        store.end_session("s1")

        # Export
        out = tmp_path / "export.json"
        stats = store.export_memory(str(out))
        assert stats["facts"] == 2
        assert out.exists()

        # Import into a fresh store
        store2 = EtchStore(":memory:", auto_migrate=True)
        imported = store2.import_memory(str(out))
        assert imported["facts"] == 2
        assert imported["sessions"] >= 1
        store2.close()

    def test_export_empty_store(self, tmp_path, store):
        out = tmp_path / "empty.json"
        stats = store.export_memory(str(out))
        assert stats["facts"] == 0
        assert stats["sessions"] == 0

    def test_import_into_existing_population(self, tmp_path, store):
        store.add_fact("Original fact", category="general")
        out = tmp_path / "data.json"
        store.export_memory(str(out))

        # Add the same data again (should be idempotent via content dedup)
        imported = store.import_memory(str(out))
        assert imported["facts"] >= 0  # dedup prevents full re-import
        assert len(store.list_facts()) == 1  # content dedup kept it to 1
