"""Tests for Typed Facts / Schema system."""

import json
import tempfile
from pathlib import Path

import pytest
from memento import EtchStore


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# TestSchemaRegistry
# ---------------------------------------------------------------------------

class TestSchemaRegistry:
    def test_register_and_get_schema(self, store):
        """Register a schema, get it back, verify all fields."""
        schema = store.register_schema(
            "bug_report",
            description="A software bug report",
            required_fields=["what", "where"],
            optional_fields=["why", "learned"],
        )
        assert schema["fact_type"] == "bug_report"
        assert schema["description"] == "A software bug report"
        assert schema["required_fields"] == ["what", "where"]
        assert schema["optional_fields"] == ["why", "learned"]
        assert "created_at" in schema
        assert "updated_at" in schema

        # Get it back fresh
        fetched = store.get_schema("bug_report")
        assert fetched is not None
        assert fetched["fact_type"] == "bug_report"
        assert fetched["required_fields"] == ["what", "where"]
        assert fetched["optional_fields"] == ["why", "learned"]

    def test_register_duplicate_updates(self, store):
        """Register same type twice, verify upsert."""
        store.register_schema("feature", description="v1", required_fields=["what"])
        store.register_schema(
            "feature", description="v2", required_fields=["what", "why"]
        )
        fetched = store.get_schema("feature")
        assert fetched["description"] == "v2"
        assert fetched["required_fields"] == ["what", "why"]

    def test_get_schema_not_found_returns_none(self, store):
        """get_schema for unregistered type returns None."""
        assert store.get_schema("nonexistent") is None

    def test_list_schemas(self, store):
        """Register 2+ schemas, list returns all."""
        store.register_schema("alpha", description="First")
        store.register_schema("beta", description="Second")
        schemas = store.list_schemas()
        assert len(schemas) >= 2
        types = [s["fact_type"] for s in schemas]
        assert "alpha" in types
        assert "beta" in types

    def test_delete_schema(self, store):
        """Delete returns True, get after delete returns None."""
        store.register_schema("temp", description="To be deleted")
        assert store.get_schema("temp") is not None
        deleted = store.delete_schema("temp")
        assert deleted is True
        assert store.get_schema("temp") is None

    def test_delete_schema_not_found_returns_false(self, store):
        """Delete non-existent schema returns False."""
        assert store.delete_schema("does_not_exist") is False

    def test_required_fields_stored_as_json_list(self, store):
        """Verify JSON round-trip for required_fields/optional_fields."""
        store.register_schema(
            "test_type",
            required_fields=["what", "why"],
            optional_fields=["learned"],
        )
        with store._lock:
            row = store._conn.execute(
                "SELECT required_fields, optional_fields FROM fact_schemas WHERE fact_type = ?",
                ("test_type",),
            ).fetchone()
        assert json.loads(row["required_fields"]) == ["what", "why"]
        assert json.loads(row["optional_fields"]) == ["learned"]


# ---------------------------------------------------------------------------
# TestTypedFactCreation
# ---------------------------------------------------------------------------

class TestTypedFactCreation:
    def test_add_fact_with_type(self, store):
        """Add fact with fact_type, verify it's stored and returned."""
        store.register_schema("note", required_fields=[])
        fid = store.add_fact("A typed note", fact_type="note")
        assert fid > 0
        fact = store.get_fact(fid)
        assert fact["fact_type"] == "note"

    def test_add_fact_without_type(self, store):
        """Backward compat — no type works."""
        fid = store.add_fact("Plain fact without type")
        assert fid > 0
        fact = store.get_fact(fid)
        assert fact["fact_type"] == ""

    def test_add_fact_validates_required_fields(self, store):
        """Register schema with required=['what'], add fact without what -> ValueError."""
        store.register_schema("strict", required_fields=["what"])
        with pytest.raises(ValueError, match="requires field 'what'"):
            store.add_fact("Missing what field", fact_type="strict")

    def test_add_fact_all_required_fields_present(self, store):
        """Required fields satisfied -> success."""
        store.register_schema("complete", required_fields=["what", "why"])
        fid = store.add_fact(
            "A complete fact",
            fact_type="complete",
            what="Completed task",
            why="Testing required fields",
        )
        assert fid > 0
        fact = store.get_fact(fid)
        assert fact["fact_type"] == "complete"
        assert fact["what"] == "Completed task"
        assert fact["why"] == "Testing required fields"

    def test_add_fact_invalid_type_raises(self, store):
        """Fact type that doesn't exist in registry -> ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            store.add_fact("Orphan fact", fact_type="unregistered_type")

    def test_add_fact_required_why_field(self, store):
        """Required 'why' field is validated."""
        store.register_schema("req_why", required_fields=["why"])
        with pytest.raises(ValueError, match="requires field 'why'"):
            store.add_fact("Missing why", fact_type="req_why")

    def test_add_fact_required_where_field(self, store):
        """Required 'where' field is validated."""
        store.register_schema("req_where", required_fields=["where"])
        with pytest.raises(ValueError, match="requires field 'where'"):
            store.add_fact("Missing where", fact_type="req_where")

    def test_add_fact_required_learned_field(self, store):
        """Required 'learned' field is validated."""
        store.register_schema("req_learned", required_fields=["learned"])
        with pytest.raises(ValueError, match="requires field 'learned'"):
            store.add_fact("Missing learned", fact_type="req_learned")


# ---------------------------------------------------------------------------
# TestTypedFactSearch
# ---------------------------------------------------------------------------

class TestTypedFactSearch:
    def test_search_facts_by_type(self, store):
        """search_facts with fact_type filter."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        store.add_fact("Login crash on submit", fact_type="bug")
        store.add_fact("Add dark mode toggle", fact_type="feature")
        results = store.search_facts("Login", fact_type="bug")
        assert len(results) >= 1
        assert all(r["fact_type"] == "bug" for r in results)

    def test_search_facts_by_type_no_match(self, store):
        """search_facts with fact_type that has no matches."""
        store.register_schema("bug", required_fields=[])
        store.add_fact("Login crash", fact_type="bug")
        results = store.search_facts("Login", fact_type="feature")
        assert len(results) == 0

    def test_search_facts_excludes_type_by_default(self, store):
        """search_facts without fact_type returns all types."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        store.add_fact("Crash bug", fact_type="bug")
        store.add_fact("New feature", fact_type="feature")
        results = store.search_facts("Crash")
        # should find it regardless of type filter
        assert len(results) >= 1

    def test_list_facts_by_type(self, store):
        """list_facts with fact_type filter."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        store.add_fact("Bug one", fact_type="bug")
        store.add_fact("Bug two", fact_type="bug")
        store.add_fact("Feature one", fact_type="feature")
        results = store.list_facts(fact_type="bug")
        assert len(results) == 2
        assert all(r["fact_type"] == "bug" for r in results)

    def test_list_facts_by_type_no_match(self, store):
        """list_facts with fact_type that has no facts."""
        store.register_schema("bug", required_fields=[])
        store.add_fact("Bug one", fact_type="bug")
        results = store.list_facts(fact_type="feature")
        assert len(results) == 0

    def test_search_by_metadata_with_type(self, store):
        """search_by_metadata with fact_type filter."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        store.add_fact("Fix crash", what="Auth bug", fact_type="bug")
        store.add_fact("Add login", what="Auth feature", fact_type="feature")
        results = store.search_by_metadata(what="Auth", fact_type="bug")
        assert len(results) == 1
        assert results[0]["fact_type"] == "bug"


# ---------------------------------------------------------------------------
# TestTypedFactUpsert
# ---------------------------------------------------------------------------

class TestTypedFactUpsert:
    def test_topic_upsert_preserves_type(self, store):
        """Topic upsert keeps fact_type from the first fact if second has no type."""
        store.register_schema("note", required_fields=[])
        fid1 = store.add_fact(
            "Original text",
            topic_key="topic:test",
            fact_type="note",
        )
        fid2 = store.add_fact(
            "Updated text",
            topic_key="topic:test",
            # no fact_type — should preserve original
        )
        assert fid2 == fid1
        fact = store.get_fact(fid2)
        assert fact["fact_type"] == "note"

    def test_topic_upsert_updates_type(self, store):
        """Topic upsert with new fact_type updates it."""
        store.register_schema("note", required_fields=[])
        store.register_schema("important", required_fields=[])
        fid1 = store.add_fact(
            "Original text",
            topic_key="topic:test",
            fact_type="note",
        )
        fid2 = store.add_fact(
            "Updated text",
            topic_key="topic:test",
            fact_type="important",
        )
        assert fid2 == fid1
        fact = store.get_fact(fid2)
        assert fact["fact_type"] == "important"

    def test_topic_upsert_without_type_stays_empty(self, store):
        """Topic upsert without fact_type leaves it empty."""
        fid1 = store.add_fact("First", topic_key="topic:no_type")
        assert store.get_fact(fid1)["fact_type"] == ""
        fid2 = store.add_fact("Second", topic_key="topic:no_type")
        assert store.get_fact(fid2)["fact_type"] == ""

    def test_topic_upsert_validates_new_type(self, store):
        """Topic upsert with unregistered fact_type raises ValueError."""
        store.register_schema("valid_type", required_fields=[])
        fid1 = store.add_fact("First", topic_key="topic:validate", fact_type="valid_type")
        with pytest.raises(ValueError, match="not registered"):
            store.add_fact("Second", topic_key="topic:validate", fact_type="invalid_type")


# ---------------------------------------------------------------------------
# TestTypedFactExportImport
# ---------------------------------------------------------------------------

class TestTypedFactExportImport:
    def test_export_import_preserves_type(self, store):
        """Round-trip export/import preserves fact_type."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        fid1 = store.add_fact("Login crash", fact_type="bug")
        fid2 = store.add_fact("Dark mode", fact_type="feature")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            export_path = f.name

        try:
            store.export_memory(export_path)

            # Import into a fresh store
            store2 = EtchStore(":memory:", auto_migrate=True)
            store2.register_schema("bug", required_fields=[])
            store2.register_schema("feature", required_fields=[])
            store2.import_memory(export_path)

            # Verify facts have fact_type
            f1 = store2.get_fact(fid1)
            f2 = store2.get_fact(fid2)
            # Note: after import, fact_ids may differ since they auto-increment
            # Search by content instead
            results = store2.search_facts("Login crash")
            assert len(results) >= 1
            assert results[0]["fact_type"] == "bug"

            results = store2.search_facts("Dark mode")
            assert len(results) >= 1
            assert results[0]["fact_type"] == "feature"

            store2.close()
        finally:
            Path(export_path).unlink(missing_ok=True)

    def test_export_includes_fact_type_in_json(self, store):
        """Exported JSON includes fact_type field."""
        store.register_schema("note", required_fields=[])
        store.add_fact("Exported note", fact_type="note")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            export_path = f.name

        try:
            store.export_memory(export_path)
            with open(export_path, encoding="utf-8") as f:
                data = json.load(f)
            assert len(data["facts"]) >= 1
            exported_fact = data["facts"][0]
            assert "fact_type" in exported_fact
            assert exported_fact["fact_type"] == "note"
        finally:
            Path(export_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestSchemaEventLog
# ---------------------------------------------------------------------------

class TestSchemaEventLog:
    def test_register_logs_event(self, store):
        """Register logs schema_registered event."""
        store.register_schema("test_event", description="Event test")
        events = store.get_event_log(event_type="schema_registered")
        assert len(events) >= 1
        metadata = events[0]["metadata"]
        assert isinstance(metadata, dict)
        assert metadata.get("fact_type") == "test_event"

    def test_delete_logs_event(self, store):
        """Delete logs schema_deleted event."""
        store.register_schema("to_delete", description="Will be deleted")
        store.delete_schema("to_delete")
        events = store.get_event_log(event_type="schema_deleted")
        assert len(events) >= 1
        metadata = events[0]["metadata"]
        assert isinstance(metadata, dict)
        assert metadata.get("fact_type") == "to_delete"


# ---------------------------------------------------------------------------
# TestUpdateFactWithType
# ---------------------------------------------------------------------------

class TestUpdateFactWithType:
    def test_update_fact_type(self, store):
        """Update fact_type via update_fact."""
        store.register_schema("bug", required_fields=[])
        store.register_schema("feature", required_fields=[])
        fid = store.add_fact("Some issue", fact_type="bug")
        store.update_fact(fid, fact_type="feature")
        fact = store.get_fact(fid)
        assert fact["fact_type"] == "feature"

    def test_update_fact_type_backward_compat(self, store):
        """update_fact without fact_type keeps it unchanged."""
        store.register_schema("bug", required_fields=[])
        fid = store.add_fact("Some issue", fact_type="bug")
        store.update_fact(fid, category="tech")
        fact = store.get_fact(fid)
        assert fact["fact_type"] == "bug"
        assert fact["category"] == "tech"
