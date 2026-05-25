"""Tests for the Project Registry (Workspaces) feature.

The ``workspaces`` table upgrades the existing project TEXT column into
a first-class entity. Workspaces auto-vivify when facts or sessions are
created with a project name.
"""

import json

import pytest

from memory_etch import EtchStore


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------

class TestCreateWorkspace:
    def test_creates_and_returns(self, store):
        ws = store.create_workspace("myproject", description="Test project")
        assert ws["name"] == "myproject"
        assert ws["description"] == "Test project"
        assert ws["tags"] == []
        assert ws["settings"] == {}
        assert ws["metadata"] == {}
        assert ws.get("workspace_id") is not None
        assert ws.get("deleted") == 0

    def test_duplicate_returns_existing(self, store):
        ws1 = store.create_workspace("dup")
        ws2 = store.create_workspace("dup", description="ignored")
        assert ws2["workspace_id"] == ws1["workspace_id"]
        assert ws2["description"] == ""  # not updated

    def test_all_optional_fields_default(self, store):
        ws = store.create_workspace("minimal")
        assert ws["tags"] == []
        assert ws["settings"] == {}
        assert ws["metadata"] == {}

    def test_stores_json_fields_as_parsed_objects(self, store):
        ws = store.create_workspace(
            "jsonfields",
            tags=["alpha", "beta"],
            settings={"theme": "dark"},
            metadata={"version": 2},
        )
        assert ws["tags"] == ["alpha", "beta"]
        assert ws["settings"] == {"theme": "dark"}
        assert ws["metadata"] == {"version": 2}


class TestGetWorkspace:
    def test_returns_none_for_missing(self, store):
        assert store.get_workspace("nonexistent") is None

    def test_returns_correct_data(self, store):
        store.create_workspace("gettest", description="Get me", tags=["a"])
        ws = store.get_workspace("gettest")
        assert ws is not None
        assert ws["name"] == "gettest"
        assert ws["description"] == "Get me"
        assert ws["tags"] == ["a"]

    def test_parsed_json_fields(self, store):
        store.create_workspace("parsed", settings={"key": "val"})
        ws = store.get_workspace("parsed")
        assert isinstance(ws["settings"], dict)
        assert ws["settings"]["key"] == "val"

    def test_returns_none_after_soft_delete(self, store):
        store.create_workspace("willdelete")
        store.delete_workspace("willdelete")
        assert store.get_workspace("willdelete") is None


class TestUpdateWorkspace:
    def test_updates_description(self, store):
        store.create_workspace("upd")
        assert store.update_workspace("upd", description="new desc")
        ws = store.get_workspace("upd")
        assert ws["description"] == "new desc"

    def test_updates_tags(self, store):
        store.create_workspace("updtags")
        store.update_workspace("updtags", tags=["x", "y"])
        ws = store.get_workspace("updtags")
        assert ws["tags"] == ["x", "y"]

    def test_updates_settings(self, store):
        store.create_workspace("updsett")
        store.update_workspace("updsett", settings={"k": "v"})
        ws = store.get_workspace("updsett")
        assert ws["settings"] == {"k": "v"}

    def test_updates_metadata(self, store):
        store.create_workspace("updmeta")
        store.update_workspace("updmeta", metadata={"count": 5})
        ws = store.get_workspace("updmeta")
        assert ws["metadata"] == {"count": 5}

    def test_updates_updated_at(self, store):
        store.create_workspace("tscheck")
        original = store.get_workspace("tscheck")
        store.update_workspace("tscheck", description="changed")
        updated = store.get_workspace("tscheck")
        assert updated["updated_at"] >= original["updated_at"]

    def test_returns_false_for_nonexistent(self, store):
        assert store.update_workspace("nope", description="x") is False

    def test_rejects_unknown_field(self, store):
        store.create_workspace("badfield")
        with pytest.raises(ValueError, match="Unknown workspace field"):
            store.update_workspace("badfield", invalid="value")

    def test_noop_on_empty_kwargs(self, store):
        store.create_workspace("noop")
        assert store.update_workspace("noop") is False


class TestDeleteWorkspace:
    def test_soft_deletes(self, store):
        store.create_workspace("todelete")
        assert store.delete_workspace("todelete") is True
        # Should be excluded from normal list
        assert store.get_workspace("todelete") is None

    def test_returns_false_for_missing(self, store):
        assert store.delete_workspace("missing") is False

    def test_returns_false_if_already_deleted(self, store):
        store.create_workspace("twice")
        store.delete_workspace("twice")
        assert store.delete_workspace("twice") is False


class TestListWorkspaces:
    def test_excludes_deleted_by_default(self, store):
        store.create_workspace("keep")
        store.create_workspace("also_keep")
        store.delete_workspace("also_keep")
        names = [ws["name"] for ws in store.list_workspaces()]
        assert "keep" in names
        assert "also_keep" not in names

    def test_include_deleted_includes_all(self, store):
        store.create_workspace("one")
        store.create_workspace("two")
        store.delete_workspace("two")
        names = [ws["name"] for ws in store.list_workspaces(include_deleted=True)]
        assert "one" in names
        assert "two" in names

    def test_empty_list_when_no_workspaces(self, store):
        assert store.list_workspaces() == []

    def test_parsed_json_fields_in_list(self, store):
        store.create_workspace("jsonlist", tags=["a", "b"], settings={"x": 1})
        ws_list = store.list_workspaces()
        ws = next(w for w in ws_list if w["name"] == "jsonlist")
        assert ws["tags"] == ["a", "b"]
        assert ws["settings"] == {"x": 1}


class TestWorkspaceStats:
    def test_counts_facts(self, store):
        store.add_fact("Fact 1", project="statstest")
        store.add_fact("Fact 2", project="statstest")
        stats = store.workspace_stats("statstest")
        assert stats["fact_count"] == 2
        assert stats["name"] == "statstest"

    def test_counts_sessions(self, store):
        store.start_session("sess1", project="sessproj")
        store.start_session("sess2", project="sessproj")
        stats = store.workspace_stats("sessproj")
        assert stats["session_count"] == 2

    def test_workspace_zero_stats_when_empty(self, store):
        store.create_workspace("emptyws")
        stats = store.workspace_stats("emptyws")
        assert stats["fact_count"] == 0
        assert stats["session_count"] == 0
        assert stats["last_active"] is None

    def test_fact_count_excludes_deleted_facts(self, store):
        store.add_fact("Active fact", project="counttest")
        fid = store.add_fact("To delete", project="counttest")
        store.soft_delete_fact(fid)
        stats = store.workspace_stats("counttest")
        assert stats["fact_count"] == 1

    def test_list_workspaces_with_stats(self, store):
        store.add_fact("Testing stats", project="statsws")
        store.start_session("s1", project="statsws")
        ws_list = store.list_workspaces(include_stats=True)
        ws = next(w for w in ws_list if w["name"] == "statsws")
        assert ws["fact_count"] == 1
        assert ws["session_count"] == 1


# ---------------------------------------------------------------------------
# Auto-vivify tests
# ---------------------------------------------------------------------------

class TestAutoVivify:
    def test_add_fact_auto_creates_workspace(self, store):
        store.add_fact("New project fact", project="autoproj")
        ws = store.get_workspace("autoproj")
        assert ws is not None
        assert ws["name"] == "autoproj"

    def test_start_session_auto_creates_workspace(self, store):
        store.start_session("sess", project="autosess")
        ws = store.get_workspace("autosess")
        assert ws is not None
        assert ws["name"] == "autosess"

    def test_empty_project_does_not_vivify(self, store):
        store.add_fact("No project")
        store.start_session("sess_empty")
        # If no workspace was created, list is empty
        assert len(store.list_workspaces()) == 0

    def test_ensure_workspace_is_idempotent(self, store):
        store.add_fact("First", project="idempotent")
        store.add_fact("Second", project="idempotent")
        store.add_fact("Third", project="idempotent")
        ws = store.get_workspace("idempotent")
        assert ws is not None
        # Only one workspace row
        assert len(store.list_workspaces()) == 1


# ---------------------------------------------------------------------------
# fact_count and last_active cache tests
# ---------------------------------------------------------------------------

class TestFactCountCache:
    def test_increments_on_add_fact(self, store):
        store.add_fact("Countable", project="cachetest")
        ws = store.get_workspace("cachetest")
        assert ws["fact_count"] >= 1

    def test_decrements_on_remove_fact(self, store):
        fid = store.add_fact("Remove me", project="cachetest")
        store.remove_fact(fid)
        ws = store.get_workspace("cachetest")
        # After decrement, should be 0 (use MAX(0) guard)
        assert ws["fact_count"] == 0

    def test_decrements_on_soft_delete(self, store):
        fid = store.add_fact("Soft delete me", project="cachetest")
        store.soft_delete_fact(fid, reason="test")
        ws = store.get_workspace("cachetest")
        assert ws["fact_count"] == 0

    def test_increments_on_restore(self, store):
        fid = store.add_fact("Restore me", project="cachetest")
        store.soft_delete_fact(fid, reason="gone")
        ws_after_delete = store.get_workspace("cachetest")
        assert ws_after_delete["fact_count"] == 0

        store.restore_fact(fid)
        ws_after_restore = store.get_workspace("cachetest")
        assert ws_after_restore["fact_count"] == 1

    def test_last_active_updates_on_add(self, store):
        store.add_fact("Active 1", project="lastact")
        ws1 = store.get_workspace("lastact")
        store.add_fact("Active 2", project="lastact")
        ws2 = store.get_workspace("lastact")
        assert ws2["last_active"] >= ws1["last_active"]


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_projects_method_unchanged(self, store):
        store.add_fact("A", project="proja")
        store.add_fact("B", project="projb")
        store.add_fact("C", project="proja")
        projs = store.projects()
        # Projects from facts (de-duped)
        assert "proja" in projs
        assert "projb" in projs

    def test_filter_by_project_still_works(self, store):
        store.add_fact("Fact A", project="alpha")
        store.add_fact("Fact B", project="beta")
        facts_alpha = store.list_facts(project="alpha")
        assert len(facts_alpha) == 1
        assert facts_alpha[0]["project"] == "alpha"

    def test_workspace_independent_of_projects_query(self, store):
        """Creating a workspace without facts doesn't appear in projects()."""
        store.create_workspace("orphan")
        assert "orphan" not in store.projects()
