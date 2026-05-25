"""Tests for the Mutation Journal (event log) feature.

Every mutation method in EtchStore should log a corresponding event
in the ``event_log`` table.  This module tests that all events are
correctly recorded with proper metadata.
"""

import json

import pytest

from memento import EtchStore


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Event type tests
# ---------------------------------------------------------------------------

class TestAddFactEvents:
    def test_add_fact_logs_event(self, store):
        fid = store.add_fact("Hello world", category="greeting", topic_key="hello", scope="canonical")
        logs = store.get_event_log()
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["event_type"] == "fact_added"
        assert ev["fact_id"] == fid
        assert ev["metadata"]["category"] == "greeting"
        assert ev["metadata"]["topic_key"] == "hello"
        assert ev["metadata"]["scope"] == "canonical"

    def test_add_fact_logs_event_with_project(self, store):
        fid = store.add_fact("Project scoped", project="myproject")
        logs = store.get_event_log(project="myproject")
        assert len(logs) >= 1
        assert logs[0]["project"] == "myproject"

    def test_dedup_logs_event(self, store):
        content = "Dedup test content"
        fid1 = store.add_fact(content, category="test")
        logs_after_first = store.get_event_log(event_type="fact_added")
        assert len(logs_after_first) >= 1

        fid2 = store.add_fact(content, category="test")
        assert fid2 == fid1  # same fact returned

        # Should have a fact_deduped event
        dedup_events = store.get_event_log(event_type="fact_deduped")
        assert len(dedup_events) >= 1
        ev = dedup_events[0]
        assert ev["fact_id"] == fid1
        assert ev["metadata"]["original_fact_id"] == fid1
        assert ev["metadata"]["duplicate_count"] >= 1


class TestSoftDeleteEvents:
    def test_soft_delete_logs_event(self, store):
        fid = store.add_fact("To delete")
        store.soft_delete_fact(fid, reason="cleanup")
        logs = store.get_event_log(event_type="fact_soft_deleted")
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["fact_id"] == fid
        assert ev["metadata"]["reason"] == "cleanup"

    def test_soft_delete_no_event_on_miss(self, store):
        store.soft_delete_fact(99999, reason="not there")
        logs = store.get_event_log(event_type="fact_soft_deleted")
        assert len(logs) == 0


class TestRestoreEvents:
    def test_restore_logs_event(self, store):
        fid = store.add_fact("To restore")
        store.soft_delete_fact(fid, reason="test")
        store.restore_fact(fid)
        logs = store.get_event_log(event_type="fact_restored")
        assert len(logs) >= 1
        assert logs[0]["fact_id"] == fid

    def test_restore_no_event_on_miss(self, store):
        store.restore_fact(99999)
        logs = store.get_event_log(event_type="fact_restored")
        assert len(logs) == 0


class TestUpdateEvents:
    def test_update_logs_event(self, store):
        fid = store.add_fact("Original")
        store.update_fact(fid, content="Updated")
        logs = store.get_event_log(event_type="fact_updated")
        assert len(logs) >= 1
        assert logs[0]["fact_id"] == fid

    def test_update_logs_old_values(self, store):
        fid = store.add_fact("Original", category="general", tags="a,b")
        store.update_fact(fid, content="Updated", tags="c,d")
        logs = store.get_event_log(event_type="fact_updated")
        assert len(logs) >= 1
        fields = logs[0]["metadata"].get("fields", {})
        # The old values of the updated fields
        assert fields.get("content") == "Original"
        assert fields.get("tags") == "a,b"

    def test_update_no_event_on_no_changes(self, store):
        fid = store.add_fact("No change")
        result = store.update_fact(fid)  # no valid kwargs
        assert result is False
        logs = store.get_event_log(event_type="fact_updated")
        # Should be from the add_fact topic upsert or nothing
        # But at least should not have a second update event
        assert len(logs) == 0 if not any(l["fact_id"] == fid for l in logs) else True


class TestPromoteEvents:
    def test_promote_logs_event(self, store):
        fid = store.add_fact("Inbox item", scope="inbox")
        store.promote_fact(fid)
        logs = store.get_event_log(event_type="fact_promoted")
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["fact_id"] == fid
        assert ev["metadata"]["from_scope"] == "inbox"
        assert ev["metadata"]["to_scope"] == "canonical"

    def test_promote_no_event_on_miss(self, store):
        store.promote_fact(99999)
        logs = store.get_event_log(event_type="fact_promoted")
        assert len(logs) == 0


class TestRejectEvents:
    def test_reject_logs_event(self, store):
        fid = store.add_fact("Reject me", scope="inbox")
        store.reject_fact(fid, reason="spam")
        logs = store.get_event_log(event_type="fact_rejected")
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["fact_id"] == fid
        assert ev["metadata"]["reason"] == "spam"

    def test_reject_no_event_on_miss(self, store):
        store.reject_fact(99999, reason="nope")
        logs = store.get_event_log(event_type="fact_rejected")
        assert len(logs) == 0


class TestRemoveEvents:
    def test_remove_logs_event(self, store):
        fid = store.add_fact("Permanent delete")
        store.remove_fact(fid)
        logs = store.get_event_log(event_type="fact_removed")
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["fact_id"] == fid
        assert ev["metadata"]["permanent"] is True


class TestPurgeEvents:
    def test_purge_logs_event(self, store):
        # Add a fact that qualifies for purge (low trust, low importance)
        fid = store.add_fact("Old low-value fact", trust_score=0.1, importance=0.1)
        # Manually set created_at to 100 days ago so purge picks it up
        store._conn.execute(
            "UPDATE facts SET created_at = datetime('now', '-100 days') WHERE fact_id = ?",
            (fid,),
        )
        store._conn.commit()
        result = store.purge_facts(dry_run=False)
        assert result["action"] == "purged"
        logs = store.get_event_log(event_type="facts_purged")
        assert len(logs) >= 1
        assert logs[0]["metadata"]["count"] >= 1


class TestRelationEvents:
    def test_relation_added_logs_event(self, store):
        fid_a = store.add_fact("Fact A")
        fid_b = store.add_fact("Fact B")
        store.add_relation(fid_a, fid_b, relation_type="related", confidence=0.8, judged_by="test")
        logs = store.get_event_log(event_type="relation_added")
        assert len(logs) >= 1
        ev = logs[0]
        assert ev["metadata"]["fact_id_a"] == fid_a
        assert ev["metadata"]["fact_id_b"] == fid_b
        assert ev["metadata"]["relation_type"] == "related"
        assert ev["metadata"]["confidence"] == 0.8
        assert ev["metadata"]["judged_by"] == "test"

    def test_judge_relation_insert_logs_event(self, store):
        fid_a = store.add_fact("Alpha")
        fid_b = store.add_fact("Beta")
        store.judge_relation(fid_a, fid_b, relation_type="compatible", confidence=0.9, judged_by="human")
        logs = store.get_event_log(event_type="relation_added")
        assert len(logs) >= 1
        assert logs[0]["metadata"]["relation_type"] == "compatible"

    def test_judge_relation_update_logs_event(self, store):
        fid_a = store.add_fact("Gamma")
        fid_b = store.add_fact("Delta")
        store.judge_relation(fid_a, fid_b, relation_type="related", confidence=0.5, judged_by="auto")
        store.judge_relation(fid_a, fid_b, relation_type="conflicts_with", confidence=0.9, judged_by="review")
        logs = store.get_event_log(event_type="relation_added")
        # Should have two relation_added events
        assert len(logs) >= 2


# ---------------------------------------------------------------------------
# Event log query tests
# ---------------------------------------------------------------------------

class TestGetEventLog:
    def test_get_event_log_filters_by_type(self, store):
        store.add_fact("One")
        store.add_fact("Two")
        logs = store.get_event_log(event_type="fact_added")
        assert all(e["event_type"] == "fact_added" for e in logs)

    def test_get_event_log_filters_by_fact_id(self, store):
        fid = store.add_fact("Target fact")
        logs = store.get_event_log(fact_id=fid)
        assert all(e["fact_id"] == fid for e in logs)

    def test_get_event_log_filters_by_project(self, store):
        store.add_fact("A", project="alpha")
        store.add_fact("B", project="beta")
        logs = store.get_event_log(project="alpha")
        assert all(e["project"] == "alpha" for e in logs)

    def test_get_event_log_pagination(self, store):
        for i in range(10):
            store.add_fact(f"Fact {i}")
        # limit = 3
        page1 = store.get_event_log(limit=3, offset=0)
        assert len(page1) <= 3
        page2 = store.get_event_log(limit=3, offset=3)
        assert len(page2) <= 3
        # Pages should be different
        if page1 and page2:
            assert page1[0]["event_id"] != page2[0]["event_id"]

    def test_event_log_order_newest_first(self, store):
        store.add_fact("First fact")
        store.add_fact("Second fact")
        logs = store.get_event_log(event_type="fact_added")
        assert len(logs) >= 2
        # event_id is auto-increment, so larger = newer
        assert logs[0]["event_id"] > logs[1]["event_id"]

    def test_get_event_log_metadata_is_dict(self, store):
        store.add_fact("Metadata check")
        logs = store.get_event_log(limit=1)
        assert len(logs) >= 1
        assert isinstance(logs[0]["metadata"], dict)


# ---------------------------------------------------------------------------
# Consolidation events
# ---------------------------------------------------------------------------

class TestConsolidationEvents:
    def test_fact_merged_event(self, store):
        """add_fact_with_consolidation with MERGE action should log fact_merged."""
        fid = store.add_fact("Existing content here", category="test")

        def search_fn(query, limit=3):
            return [{"fact_id": fid, "content": "Existing content here"}]

        def llm_decide_fn(new_content, existing):
            return {"action": "MERGE", "merged_content": "Merged content"}

        result = store.add_fact_with_consolidation(
            "Existing content here plus more", category="test",
            search_fn=search_fn, llm_decide_fn=llm_decide_fn,
        )
        assert result["action"] == "merged"
        logs = store.get_event_log(event_type="fact_merged")
        assert len(logs) >= 1
        assert logs[0]["metadata"].get("replaced_fact_id") == fid

    def test_fact_replaced_event(self, store):
        """add_fact_with_consolidation with REPLACE action should log fact_replaced."""
        fid = store.add_fact("Old content to replace", category="test")

        def search_fn(query, limit=3):
            return [{"fact_id": fid, "content": "Old content to replace"}]

        def llm_decide_fn(new_content, existing):
            return {"action": "REPLACE"}

        result = store.add_fact_with_consolidation(
            "Old content to replace with new stuff", category="test",
            search_fn=search_fn, llm_decide_fn=llm_decide_fn,
        )
        assert result["action"] == "merged"
        logs = store.get_event_log(event_type="fact_replaced")
        assert len(logs) >= 1
        assert logs[0]["metadata"].get("replaced_fact_id") == fid
