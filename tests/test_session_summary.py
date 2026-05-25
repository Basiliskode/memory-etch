"""Tests for generate_session_summary — best-effort aggregation of session facts."""

from pathlib import Path

import pytest

from memento.store import EtchStore


@pytest.fixture
def db_path(tmp_path):
    return Path(tmp_path) / "test_summary.db"


@pytest.fixture
def store(db_path):
    s = EtchStore(str(db_path), auto_migrate=True)
    yield s
    s.close()


class TestGenerateSessionSummary:
    """generate_session_summary should aggregate facts from a session."""

    def test_returns_dict_with_expected_keys(self, store):
        """Even for empty sessions, all keys should be present."""
        summary = store.generate_session_summary("nonexistent")
        assert isinstance(summary, dict)
        assert "goal" in summary
        assert "discoveries" in summary
        assert "accomplished" in summary
        assert "next_steps" in summary

    def test_goal_from_goal_fact(self, store):
        """Fact starting with '## Goal' becomes the goal."""
        store.add_fact("## Goal\nBuild a memory system", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert summary["goal"] == "## Goal\nBuild a memory system"

    def test_goal_case_insensitive(self, store):
        """'## GOAL' or '## goal' also works."""
        store.add_fact("## GOAL\nBuild a memory system", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert "Build a memory system" in summary["goal"]

    def test_discoveries_contains_bugfix_and_discovery_facts(self, store):
        """Facts with category 'discovery' or 'bugfix' go into discoveries."""
        store.add_fact(
            "Found a race condition",
            session_id="s1",
            category="discovery",
        )
        store.add_fact(
            "Fixed the race condition",
            session_id="s1",
            category="bugfix",
        )
        # A general fact should NOT be in discoveries
        store.add_fact(
            "General note about project",
            session_id="s1",
            category="general",
        )
        summary = store.generate_session_summary("s1")
        assert len(summary["discoveries"]) >= 2
        discovery_contents = [d for d in summary["discoveries"]]
        assert any("race condition" in d for d in discovery_contents)

    def test_accomplished_includes_all_session_facts(self, store):
        """All facts in the session are listed in accomplished."""
        store.add_fact("First fact", session_id="s1")
        store.add_fact("Second fact", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert len(summary["accomplished"]) >= 2

    def test_next_steps_from_fact(self, store):
        """Fact starting with '## Next Steps' populates next_steps."""
        store.add_fact("## Next Steps\nDeploy to production", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert "Deploy to production" in summary["next_steps"]

    def test_next_steps_alt_format(self, store):
        """Fact starting with 'Next Steps:' also works."""
        store.add_fact("Next Steps: Deploy to production", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert "Next Steps: Deploy to production" in summary["next_steps"]

    def test_missing_sections_default_to_empty(self, store):
        """If no discovery/bugfix facts, discoveries is empty list."""
        store.add_fact("Just a note", session_id="s1")
        summary = store.generate_session_summary("s1")
        assert summary["goal"] == ""
        assert summary["discoveries"] == []
        assert summary["next_steps"] == ""

    def test_only_session_facts_included(self, store):
        """Facts from other sessions are excluded."""
        store.add_fact("Session 1 fact", session_id="s1")
        store.add_fact("Session 2 fact", session_id="s2")
        summary = store.generate_session_summary("s1")
        accomplished_contents = [a for a in summary["accomplished"]]
        assert any("Session 1" in a for a in accomplished_contents)
        assert not any("Session 2" in a for a in accomplished_contents)
