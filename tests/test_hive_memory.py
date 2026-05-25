"""Tests for Hive Memory v1 — Core Store Governance (PR 1).

Strict TDD: tests written BEFORE implementation.
"""

import sqlite3
import pytest
from memento import EtchStore


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# =========================================================================
# Task 1.1 — Schema migration columns
# =========================================================================

class TestMigrationColumns:
    """source_harness, source_agent, source_kind, scope are added by migration."""

    def test_migration_adds_source_columns(self, store):
        """Migration adds source_harness, source_agent, source_kind, scope columns."""
        cols = {
            r["name"]
            for r in store._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        assert "source_harness" in cols, "source_harness column missing"
        assert "source_agent" in cols, "source_agent column missing"
        assert "source_kind" in cols, "source_kind column missing"
        assert "scope" in cols, "scope column missing"

    def test_migration_defaults(self, store):
        """New columns have expected defaults."""
        fid = store.add_fact("plain fact without scope")
        row = store._conn.execute(
            "SELECT source_harness, source_agent, source_kind, scope FROM facts WHERE fact_id = ?",
            (fid,),
        ).fetchone()
        assert row["source_harness"] == ""
        assert row["source_agent"] == ""
        assert row["source_kind"] == ""
        assert row["scope"] == "canonical"

    def test_migration_on_pre_hive_database(self):
        """A DB created before hive columns still gets them after migration."""
        # Create a store WITHOUT running migration
        pre = EtchStore(":memory:", auto_migrate=False)
        pre._ensure_schema()
        # Verify columns don't exist yet (pre-hive state)
        cols_pre = {
            r["name"]
            for r in pre._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        for col in ("source_harness", "source_agent", "source_kind", "scope"):
            assert col not in cols_pre, f"{col} should not exist pre-migration"
        # Add a fact row directly (raw SQL since add_fact needs migration columns)
        pre._conn.execute(
            "INSERT INTO facts (content) VALUES (?)", ("pre-hive fact",)
        )
        pre._conn.commit()
        fid = pre._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?", ("pre-hive fact",)
        ).fetchone()["fact_id"]
        # Now run migration
        pre._migrate_schema()
        # Verify columns exist after migration
        cols_post = {
            r["name"]
            for r in pre._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        for col in ("source_harness", "source_agent", "source_kind", "scope"):
            assert col in cols_post, f"{col} missing after migration"
        # Pre-hive fact remains readable and gets defaults
        row = pre._conn.execute(
            "SELECT source_harness, source_agent, source_kind, scope FROM facts WHERE fact_id = ?",
            (fid,),
        ).fetchone()
        assert row["source_harness"] == ""
        assert row["source_agent"] == ""
        assert row["source_kind"] == ""
        assert row["scope"] == "canonical"
        pre.close()


# =========================================================================
# Task 1.2 — Scope validation helper
# =========================================================================

class TestScopeValidation:
    """Only canonical, inbox, personal, ephemeral are valid scopes."""

    def test_invalid_scope_raises_value_error(self, store):
        """Invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="scope"):
            store.add_fact("fact with invalid scope", scope="admin")

    def test_invalid_scope_does_not_write(self, store):
        """Invalid scope does NOT persist any data."""
        with pytest.raises(ValueError):
            store.add_fact("fact with admin scope", scope="admin")
        count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert count == 0, "No fact should be stored with invalid scope"

    def test_valid_scopes_accepted(self, store):
        """Each valid scope is accepted: canonical, inbox, personal, ephemeral."""
        for scope in ("canonical", "inbox", "personal", "ephemeral"):
            fid = store.add_fact(f"fact with scope {scope}", scope=scope)
            assert fid > 0
            row = store._conn.execute(
                "SELECT scope FROM facts WHERE fact_id = ?", (fid,)
            ).fetchone()
            assert row["scope"] == scope

    def test_multiple_invalid_scopes(self, store):
        """Various invalid scope values all raise ValueError."""
        for bad in ("admin", "system", "private", "public", "", "INBOX"):
            with pytest.raises(ValueError, match="scope"):
                store.add_fact(f"bad scope test {bad}", scope=bad)

    def test_empty_string_scope_raises(self, store):
        """Empty string is not a valid scope."""
        with pytest.raises(ValueError, match="scope"):
            store.add_fact("fact with empty scope", scope="")


# =========================================================================
# Task 1.3 — Backward-compatible add_fact with provenance metadata
# =========================================================================

class TestProvenanceMetadata:
    """add_fact accepts optional provenance args while preserving backward compat."""

    def test_backward_compat_defaults(self, store):
        """add_fact(content) without provenance args stores canonical with empty source."""
        fid = store.add_fact("backward compat fact")
        row = store._conn.execute(
            """SELECT source_harness, source_agent, source_kind, scope
               FROM facts WHERE fact_id = ?""",
            (fid,),
        ).fetchone()
        assert row["scope"] == "canonical"
        assert row["source_harness"] == ""
        assert row["source_agent"] == ""
        assert row["source_kind"] == ""

    def test_provenance_persists(self, store):
        """Provenance metadata persists through add_fact and is readable."""
        fid = store.add_fact(
            "fact with provenance",
            source_harness="opencode",
            source_agent="worker-1",
            source_kind="manual",
            scope="inbox",
        )
        row = store._conn.execute(
            """SELECT source_harness, source_agent, source_kind, scope
               FROM facts WHERE fact_id = ?""",
            (fid,),
        ).fetchone()
        assert row["source_harness"] == "opencode"
        assert row["source_agent"] == "worker-1"
        assert row["source_kind"] == "manual"
        assert row["scope"] == "inbox"

    def test_provenance_in_get_fact(self, store):
        """get_fact() returns provenance metadata."""
        fid = store.add_fact(
            "provenance roundtrip",
            source_harness="test-harness",
            source_agent="test-agent",
            source_kind="test",
            scope="inbox",
        )
        fact = store.get_fact(fid)
        assert fact is not None
        assert fact["source_harness"] == "test-harness"
        assert fact["source_agent"] == "test-agent"
        assert fact["source_kind"] == "test"
        assert fact["scope"] == "inbox"

    def test_provenance_partial(self, store):
        """Partial provenance: only provided fields are set; others default."""
        fid = store.add_fact(
            "partial provenance",
            source_harness="harness-only",
            scope="inbox",
        )
        row = store._conn.execute(
            """SELECT source_harness, source_agent, source_kind, scope
               FROM facts WHERE fact_id = ?""",
            (fid,),
        ).fetchone()
        assert row["source_harness"] == "harness-only"
        assert row["source_agent"] == ""
        assert row["source_kind"] == ""
        assert row["scope"] == "inbox"

    def test_provenance_on_topic_upsert(self, store):
        """Topic upsert preserves provenance metadata."""
        fid1 = store.add_fact(
            "provenance upsert v1",
            topic_key="topic:provenance-test",
            source_harness="v1-harness",
            scope="inbox",
        )
        fid2 = store.add_fact(
            "provenance upsert v2",
            topic_key="topic:provenance-test",
            source_harness="v2-harness",
            scope="canonical",
        )
        assert fid2 == fid1
        row = store._conn.execute(
            """SELECT source_harness, source_agent, source_kind, scope
               FROM facts WHERE fact_id = ?""",
            (fid1,),
        ).fetchone()
        assert row["source_harness"] == "v2-harness"
        assert row["scope"] == "canonical"


# =========================================================================
# Task 1.4 — Inbox lifecycle methods
# =========================================================================

class TestInboxLifecycle:
    """list_inbox, promote_fact, reject_fact."""

    def test_list_inbox_empty(self, store):
        """list_inbox returns empty list when no inbox facts exist."""
        assert store.list_inbox() == []

    def test_list_inbox_returns_only_inbox_facts(self, store):
        """list_inbox returns only facts where scope='inbox'."""
        store.add_fact("canonical fact A", scope="canonical")
        store.add_fact("inbox fact B", scope="inbox")
        store.add_fact("inbox fact C", scope="inbox")
        inbox = store.list_inbox()
        assert len(inbox) == 2
        for f in inbox:
            assert f["scope"] == "inbox"
            assert f["deleted"] == 0 or f["deleted"] is None

    def test_list_inbox_newest_first(self, store):
        """list_inbox returns newest facts first."""
        f1 = store.add_fact("inbox old", scope="inbox")
        f2 = store.add_fact("inbox new", scope="inbox")
        inbox = store.list_inbox()
        assert inbox[0]["fact_id"] == f2  # newest first

    def test_list_inbox_filters_by_project(self, store):
        """list_inbox filters by project."""
        store.add_fact("project a inbox", scope="inbox", project="proj-a")
        store.add_fact("project b inbox", scope="inbox", project="proj-b")
        inbox_a = store.list_inbox(project="proj-a")
        assert len(inbox_a) == 1
        assert inbox_a[0]["project"] == "proj-a"

    def test_list_inbox_filters_by_source_harness(self, store):
        """list_inbox filters by source_harness."""
        store.add_fact("opencode inbox", scope="inbox", source_harness="opencode")
        store.add_fact("hermes inbox", scope="inbox", source_harness="hermes")
        inbox_open = store.list_inbox(source_harness="opencode")
        assert len(inbox_open) == 1
        assert inbox_open[0]["source_harness"] == "opencode"

    def test_list_inbox_combined_filters(self, store):
        """list_inbox combines project AND source_harness filter."""
        store.add_fact("match", scope="inbox", project="proj-a", source_harness="hermes")
        store.add_fact("wrong project", scope="inbox", project="proj-b", source_harness="hermes")
        store.add_fact("wrong harness", scope="inbox", project="proj-a", source_harness="opencode")
        inbox = store.list_inbox(project="proj-a", source_harness="hermes")
        assert len(inbox) == 1
        assert inbox[0]["content"] == "match"

    def test_list_inbox_respects_limit(self, store):
        """list_inbox respects limit parameter."""
        for i in range(10):
            store.add_fact(f"inbox fact {i}", scope="inbox")
        inbox = store.list_inbox(limit=3)
        assert len(inbox) == 3

    def test_promote_fact_changes_scope(self, store):
        """promote_fact changes scope from inbox to canonical."""
        fid = store.add_fact("promotable fact", scope="inbox")
        result = store.promote_fact(fid)
        assert result is True
        row = store._conn.execute(
            "SELECT scope FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["scope"] == "canonical"

    def test_promote_fact_updates_timestamp(self, store):
        """promote_fact updates the updated_at timestamp."""
        fid = store.add_fact("timestamp check", scope="inbox")
        original_updated = store._conn.execute(
            "SELECT updated_at FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()["updated_at"]
        store.promote_fact(fid)
        new_updated = store._conn.execute(
            "SELECT updated_at FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()["updated_at"]
        assert new_updated > original_updated or new_updated >= original_updated

    def test_promote_nonexistent_fact_returns_false(self, store):
        """promote_fact on nonexistent fact_id returns False."""
        result = store.promote_fact(99999)
        assert result is False

    def test_promote_already_canonical_returns_false(self, store):
        """promote_fact on already canonical fact returns False."""
        fid = store.add_fact("already canonical", scope="canonical")
        result = store.promote_fact(fid)
        assert result is False

    def test_reject_fact_soft_deletes(self, store):
        """reject_fact soft-deletes the inbox fact."""
        fid = store.add_fact("rejectable fact", scope="inbox")
        result = store.reject_fact(fid, reason="spam")
        assert result is True
        row = store._conn.execute(
            "SELECT deleted, deleted_reason, scope FROM facts WHERE fact_id = ?",
            (fid,),
        ).fetchone()
        assert row["deleted"] == 1
        assert row["deleted_reason"] == "spam"
        assert row["scope"] == "inbox"

    def test_reject_fact_default_reason(self, store):
        """reject_fact with no reason stores empty string."""
        fid = store.add_fact("rejected no reason", scope="inbox")
        store.reject_fact(fid)
        row = store._conn.execute(
            "SELECT deleted, deleted_reason FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["deleted"] == 1
        assert row["deleted_reason"] == ""

    def test_reject_nonexistent_fact_returns_false(self, store):
        """reject_fact on nonexistent fact_id returns False."""
        result = store.reject_fact(99999)
        assert result is False

    def test_reject_already_deleted_returns_false(self, store):
        """reject_fact on already deleted fact returns False."""
        fid = store.add_fact("already deleted", scope="inbox")
        store.reject_fact(fid, reason="first time")
        result = store.reject_fact(fid, reason="second time")
        assert result is False

    def test_rejected_fact_not_in_list_inbox(self, store):
        """Rejected facts do NOT appear in list_inbox."""
        fid = store.add_fact("will be rejected", scope="inbox")
        store.add_fact("stays in inbox", scope="inbox")
        store.reject_fact(fid, reason="test")
        inbox = store.list_inbox()
        ids = [f["fact_id"] for f in inbox]
        assert fid not in ids


# =========================================================================
# Task 2.1 — Store search filters (PR 2)
# =========================================================================

class TestStoreSearchFilters:
    """Default search returns only canonical; explicit scope/source filters work."""

    def test_default_search_excludes_non_canonical(self, store):
        """Default search (no scope arg) returns only canonical facts."""
        store.add_fact("canonical alpha", scope="canonical")
        store.add_fact("canonical beta", scope="canonical")
        store.add_fact("inbox gamma", scope="inbox")
        store.add_fact("personal delta", scope="personal")
        store.add_fact("ephemeral epsilon", scope="ephemeral")
        results = store.search("canonical")
        contents = [r["content"] for r in results]
        assert "canonical alpha" in contents
        assert "canonical beta" in contents
        assert "inbox gamma" not in contents
        assert "personal delta" not in contents
        assert "ephemeral epsilon" not in contents

    def test_explicit_scope_inbox(self, store):
        """Explicit scope='inbox' returns inbox facts."""
        store.add_fact("canonical one", scope="canonical")
        store.add_fact("inbox one", scope="inbox")
        results = store.search("one", scope="inbox")
        contents = [r["content"] for r in results]
        assert "inbox one" in contents
        assert "canonical one" not in contents

    def test_explicit_scope_personal(self, store):
        """Explicit scope='personal' returns personal facts."""
        store.add_fact("canonical one", scope="canonical")
        store.add_fact("personal one", scope="personal")
        results = store.search("one", scope="personal")
        contents = [r["content"] for r in results]
        assert "personal one" in contents
        assert "canonical one" not in contents

    def test_explicit_scope_ephemeral(self, store):
        """Explicit scope='ephemeral' returns ephemeral facts."""
        store.add_fact("canonical one", scope="canonical")
        store.add_fact("ephemeral one", scope="ephemeral")
        results = store.search("one", scope="ephemeral")
        contents = [r["content"] for r in results]
        assert "ephemeral one" in contents
        assert "canonical one" not in contents

    def test_empty_search_default_canonical_only(self, store):
        """Default search excludes non-canonical even when no canonical exists."""
        store.add_fact("inbox one", scope="inbox")
        results = store.search("one")
        assert len(results) == 0

    def test_source_harness_filter(self, store):
        """Search filters by source_harness when provided."""
        store.add_fact("harness a alpha", scope="canonical", source_harness="harness_a")
        store.add_fact("harness b beta", scope="canonical", source_harness="harness_b")
        results = store.search("alpha", source_harness="harness_a")
        contents = [r["content"] for r in results]
        assert "harness a alpha" in contents
        assert "harness b beta" not in contents

    def test_source_agent_filter(self, store):
        """Search filters by source_agent when provided."""
        store.add_fact("agent x ray", scope="canonical", source_agent="agent_x")
        store.add_fact("agent y ray", scope="canonical", source_agent="agent_y")
        results = store.search("ray", source_agent="agent_x")
        contents = [r["content"] for r in results]
        assert "agent x ray" in contents
        assert "agent y ray" not in contents

    def test_source_kind_filter(self, store):
        """Search filters by source_kind when provided."""
        store.add_fact("kind manual one", scope="canonical", source_kind="manual")
        store.add_fact("kind auto one", scope="canonical", source_kind="auto")
        results = store.search("one", source_kind="manual")
        contents = [r["content"] for r in results]
        assert "kind manual one" in contents
        assert "kind auto one" not in contents

    def test_search_facts_scope_default(self, store):
        """search_facts defaults to canonical only (via search)."""
        store.add_fact("canonical one", scope="canonical")
        store.add_fact("inbox one", scope="inbox")
        results = store.search_facts("one")
        contents = [r["content"] for r in results]
        assert "canonical one" in contents
        assert "inbox one" not in contents

    def test_search_facts_explicit_scope(self, store):
        """search_facts with explicit scope returns only that scope."""
        store.add_fact("canonical one", scope="canonical")
        store.add_fact("inbox one", scope="inbox")
        results = store.search_facts("one", scope="inbox")
        contents = [r["content"] for r in results]
        assert "inbox one" in contents
        assert "canonical one" not in contents


# =========================================================================
# Task 2.1 — Store list_facts / search_by_metadata / search_by_vector filters
# =========================================================================

class TestStoreListFactsScope:
    """list_facts defaults to canonical only and accepts explicit scope."""

    def test_list_facts_default_canonical_only(self, store):
        """list_facts without scope returns only canonical facts."""
        store.add_fact("canonical list one", scope="canonical")
        store.add_fact("inbox list one", scope="inbox")
        results = store.list_facts()
        contents = [r["content"] for r in results]
        assert "canonical list one" in contents
        assert "inbox list one" not in contents

    def test_list_facts_explicit_scope(self, store):
        """list_facts with explicit scope returns only that scope."""
        store.add_fact("canonical list one", scope="canonical")
        store.add_fact("inbox list one", scope="inbox")
        results = store.list_facts(scope="inbox")
        contents = [r["content"] for r in results]
        assert "inbox list one" in contents


class TestStoreSearchByMetadataScope:
    """search_by_metadata defaults to canonical only and accepts scope."""

    def test_search_by_metadata_default_canonical_only(self, store):
        """search_by_metadata without scope returns only canonical."""
        store.add_fact("canonical meta one", scope="canonical", what="test")
        store.add_fact("inbox meta one", scope="inbox", what="test")
        results = store.search_by_metadata(what="test")
        contents = [r["content"] for r in results]
        assert "canonical meta one" in contents
        assert "inbox meta one" not in contents

    def test_search_by_metadata_explicit_scope(self, store):
        """search_by_metadata with explicit scope works."""
        store.add_fact("canonical meta one", scope="canonical", what="test")
        store.add_fact("inbox meta one", scope="inbox", what="test")
        results = store.search_by_metadata(what="test", scope="inbox")
        contents = [r["content"] for r in results]
        assert "inbox meta one" in contents

    def test_search_by_metadata_returns_provenance_fields(self, store):
        """search_by_metadata returns source metadata for roundtrip checks."""
        store.add_fact(
            "canonical meta provenance",
            scope="canonical",
            what="test",
            source_harness="opencode",
            source_agent="worker-a",
            source_kind="manual",
        )
        results = store.search_by_metadata(what="test")
        assert results[0]["source_harness"] == "opencode"
        assert results[0]["source_agent"] == "worker-a"
        assert results[0]["source_kind"] == "manual"
        assert results[0]["scope"] == "canonical"


class TestRetrieverScopeFilters:
    """Retriever search defaults to canonical only; explicit scope/source filters work."""

    @pytest.fixture
    def retriever(self, store):
        from memento import EtchRetriever
        return EtchRetriever(store, hrr_dim=256)

    def test_default_search_excludes_non_canonical(self, retriever):
        """Default retriever search excludes inbox/ephemeral."""
        retriever._store.add_fact("alpha canonical one", scope="canonical")
        retriever._store.add_fact("beta inbox one", scope="inbox")
        retriever._store.add_fact("gamma ephemeral one", scope="ephemeral")
        results = retriever.search("one")
        contents = [r["content"] for r in results]
        assert "alpha canonical one" in contents
        assert "beta inbox one" not in contents
        assert "gamma ephemeral one" not in contents

    def test_explicit_scope_inbox(self, retriever):
        """Retriever search with explicit scope='inbox' returns inbox facts."""
        retriever._store.add_fact("alpha canonical one", scope="canonical")
        retriever._store.add_fact("beta inbox one", scope="inbox")
        results = retriever.search("one", scope="inbox")
        contents = [r["content"] for r in results]
        assert "beta inbox one" in contents
        assert "alpha canonical one" not in contents

    def test_explicit_scope_personal(self, retriever):
        """Retriever search with explicit scope='personal' returns personal facts."""
        retriever._store.add_fact("alpha canonical one", scope="canonical")
        retriever._store.add_fact("delta personal one", scope="personal")
        results = retriever.search("one", scope="personal")
        contents = [r["content"] for r in results]
        assert "delta personal one" in contents

    def test_retriever_source_harness_filter(self, retriever):
        """Retriever search filters by source_harness."""
        retriever._store.add_fact("echo harness one", scope="canonical", source_harness="xbox")
        retriever._store.add_fact("foxtrot harness one", scope="canonical", source_harness="ybox")
        results = retriever.search("one", source_harness="xbox")
        contents = [r["content"] for r in results]
        assert "echo harness one" in contents
        assert "foxtrot harness one" not in contents

    def test_retriever_probe_default_canonical_only(self, retriever):
        """Retriever.probe defaults to canonical only."""
        retriever._store.add_fact("golf probe canonical", scope="canonical", tags="probe_group")
        retriever._store.add_fact("hotel probe inbox", scope="inbox", tags="probe_group")
        results = retriever.probe("probe")
        contents = [r["content"] for r in results]
        assert "golf probe canonical" in contents
        assert "hotel probe inbox" not in contents


# =========================================================================
# Task 3.3 — Hermes provider metadata passthrough
# =========================================================================

class TestHermesProviderMetadata:
    """Hermes EtchMemoryProvider passes provenance metadata."""

    def test_hermes_add_fact_default_harness(self, store):
        """Hermes add action defaults source_harness='hermes' via tool dispatch."""
        from memento.etch import EtchMemoryProvider
        import json

        p = EtchMemoryProvider({"db_path": ":memory:"})
        p.initialize("test-hermes-default")
        # Override the store to use our test store
        p._store = store
        p._session_id = "test-hermes-default"

        result = p.handle_tool_call("fact_store", {
            "action": "add",
            "content": "hermes default harness fact",
        })
        data = json.loads(result)
        assert "fact_id" in data

        row = store._conn.execute(
            "SELECT source_harness, source_agent, source_kind, scope FROM facts WHERE fact_id = ?",
            (data["fact_id"],),
        ).fetchone()
        assert row["source_harness"] == "hermes", f"Expected 'hermes', got '{row['source_harness']}'"
        assert row["source_kind"] == "provider", f"Expected 'provider', got '{row['source_kind']}'"
        assert row["scope"] == "canonical"

    def test_hermes_add_fact_passthrough_source(self, store):
        """Hermes add action passes explicit provenance args to add_fact."""
        from memento.etch import EtchMemoryProvider
        import json

        p = EtchMemoryProvider({"db_path": ":memory:"})
        p.initialize("test-hermes-passthrough")
        p._store = store
        p._session_id = "test-hermes-passthrough"

        result = p.handle_tool_call("fact_store", {
            "action": "add",
            "content": "hermes explicit provenance fact",
            "source_harness": "my-harness",
            "source_agent": "my-agent",
            "source_kind": "manual",
        })
        data = json.loads(result)
        assert "fact_id" in data

        row = store._conn.execute(
            "SELECT source_harness, source_agent, source_kind FROM facts WHERE fact_id = ?",
            (data["fact_id"],),
        ).fetchone()
        assert row["source_harness"] == "my-harness"
        assert row["source_agent"] == "my-agent"
        assert row["source_kind"] == "manual"

    def test_hermes_add_fact_with_scope(self, store):
        """Hermes add action passes scope when provided."""
        from memento.etch import EtchMemoryProvider
        import json

        p = EtchMemoryProvider({"db_path": ":memory:"})
        p.initialize("test-hermes-scope")
        p._store = store
        p._session_id = "test-hermes-scope"

        result = p.handle_tool_call("fact_store", {
            "action": "add",
            "content": "hermes inbox fact",
            "scope": "inbox",
        })
        data = json.loads(result)
        assert "fact_id" in data

        row = store._conn.execute(
            "SELECT scope FROM facts WHERE fact_id = ?",
            (data["fact_id"],),
        ).fetchone()
        assert row["scope"] == "inbox"


# =========================================================================
# Tasks 3.1 & 3.2 — MCP metadata passthrough and inbox tools
# =========================================================================

try:
    from memento.mcp.server import add_fact as mcp_add_fact
    from memento.mcp.server import search_facts as mcp_search_facts
    from memento.mcp.server import list_inbox as mcp_list_inbox
    from memento.mcp.server import promote_fact as mcp_promote_fact
    from memento.mcp.server import reject_fact as mcp_reject_fact
    HAS_MCP = True
except (ImportError, ModuleNotFoundError):
    HAS_MCP = False


@pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")
class TestMCPMetadataPassthrough:
    """MCP tools pass provenance metadata to store methods."""

    def test_mcp_add_accepts_provenance_args(self):
        """MCP add_fact tool accepts optional provenance args."""
        # Can only test signature/import; integration requires mcp runtime
        from memento.mcp.server import add_fact
        import inspect
        sig = inspect.signature(add_fact)
        assert "source_harness" in sig.parameters
        assert "source_agent" in sig.parameters
        assert "source_kind" in sig.parameters
        assert "scope" in sig.parameters

    def test_mcp_search_accepts_filters(self):
        """MCP search_facts tool accepts optional filter args."""
        from memento.mcp.server import search_facts
        import inspect
        sig = inspect.signature(search_facts)
        assert "scope" in sig.parameters
        assert "source_harness" in sig.parameters
        assert "source_agent" in sig.parameters
        assert "source_kind" in sig.parameters


@pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")
class TestMCPInboxTools:
    """MCP inbox review tools."""

    def test_mcp_list_inbox_exists(self):
        """MCP list_inbox tool is callable and returns JSON."""
        from memento.mcp.server import list_inbox
        import inspect
        sig = inspect.signature(list_inbox)
        assert "project" in sig.parameters
        assert "source_harness" in sig.parameters
        assert "limit" in sig.parameters

    def test_mcp_promote_fact_exists(self):
        """MCP promote_fact tool accepts fact_id."""
        from memento.mcp.server import promote_fact
        import inspect
        sig = inspect.signature(promote_fact)
        assert "fact_id" in sig.parameters

    def test_mcp_reject_fact_exists(self):
        """MCP reject_fact tool accepts fact_id and reason."""
        from memento.mcp.server import reject_fact
        import inspect
        sig = inspect.signature(reject_fact)
        assert "fact_id" in sig.parameters
        assert "reason" in sig.parameters
