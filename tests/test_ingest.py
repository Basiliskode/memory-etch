"""Tests for the ingest pipeline — parsers and ``EtchStore.ingest()``."""

import json
from pathlib import Path

import pytest

from memory_etch import EtchStore
from memory_etch.ingest import (
    detect_format,
    parse_csv,
    parse_json,
    parse_markdown,
    parse_text,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# ======================================================================
# Markdown parsing
# ======================================================================


class TestParseMarkdown:
    def test_markdown_sections(self):
        text = """## Architecture
The system uses a hexagonal architecture.

## Database
We chose SQLite for persistence.

## Testing
Pytest is the test runner.
"""
        results = list(parse_markdown(text))
        assert len(results) == 3
        assert results[0] == (
            "The system uses a hexagonal architecture.",
            {"heading": "Architecture"},
        )
        assert results[1] == (
            "We chose SQLite for persistence.",
            {"heading": "Database"},
        )
        assert results[2] == (
            "Pytest is the test runner.",
            {"heading": "Testing"},
        )

    def test_markdown_no_headings(self):
        text = "Just a plain paragraph of text with no headings at all."
        results = list(parse_markdown(text))
        assert len(results) == 1
        assert results[0] == (text, {})

    def test_markdown_heading_in_metadata(self):
        text = """## Design
The design is minimal and clean.
"""
        results = list(parse_markdown(text))
        assert len(results) == 1
        content, meta = results[0]
        assert content == "The design is minimal and clean."
        assert meta["heading"] == "Design"

    def test_markdown_front_matter_skipped(self):
        text = """This is front matter that should be skipped.

## Section One
Body of section one.

## Section Two
Body of section two.
"""
        results = list(parse_markdown(text))
        assert len(results) == 2
        assert results[0][1]["heading"] == "Section One"
        assert results[1][1]["heading"] == "Section Two"

    def test_markdown_empty_text(self):
        results = list(parse_markdown(""))
        assert len(results) == 0

    def test_markdown_heading_only_no_body(self):
        text = "## Empty Section\n"
        results = list(parse_markdown(text))
        assert len(results) == 0


# ======================================================================
# Text parsing
# ======================================================================


class TestParseText:
    def test_text_paragraphs(self):
        text = """First paragraph with some content.

Second paragraph here.

Third and final paragraph.
"""
        results = list(parse_text(text, delimiter="paragraph"))
        assert len(results) == 3
        assert results[0][0] == "First paragraph with some content."
        assert results[1][0] == "Second paragraph here."
        assert results[2][0] == "Third and final paragraph."

    def test_text_lines(self):
        text = "line one\nline two\nline three\n"
        results = list(parse_text(text, delimiter="line"))
        assert len(results) == 3
        assert results[0][0] == "line one"
        assert results[1][0] == "line two"
        assert results[2][0] == "line three"

    def test_text_chunk_size(self):
        text = "hello world foo bar baz qux"
        # chunk_size=10 should split into ~10-char chunks
        results = list(parse_text(text, delimiter=10))
        assert len(results) >= 2
        # All chunks should be non-empty
        for content, meta in results:
            assert content
            assert "chunk_index" in meta
            assert "total_chunks" in meta
        # Check that total_chunks is consistent
        total = results[0][1]["total_chunks"]
        for _, meta in results:
            assert meta["total_chunks"] == total
        # Verify chunk_index is sequential
        indices = [meta["chunk_index"] for _, meta in results]
        assert indices == list(range(len(indices)))

    def test_text_empty_text(self):
        results = list(parse_text("", delimiter="paragraph"))
        assert len(results) == 0
        results = list(parse_text("   ", delimiter="line"))
        assert len(results) == 0

    def test_text_single_paragraph(self):
        text = "Just one block of text with no blank lines."
        results = list(parse_text(text, delimiter="paragraph"))
        assert len(results) == 1
        assert results[0][0] == text


# ======================================================================
# JSON parsing
# ======================================================================


class TestParseJson:
    def test_json_list_of_strings(self):
        data = '["Fact one", "Fact two", "Fact three"]'
        results = list(parse_json(data))
        assert len(results) == 3
        assert results[0][0] == "Fact one"
        assert results[1][0] == "Fact two"
        assert results[2][0] == "Fact three"

    def test_json_list_of_dicts_no_content_key(self):
        data = [{"title": "Alpha", "body": "Content A"}, {"title": "Beta", "body": "Content B"}]
        results = list(parse_json(data))
        assert len(results) == 2
        # Without content_key each dict becomes str(item)
        assert "Alpha" in results[0][0]
        assert "Beta" in results[1][0]

    def test_json_list_of_dicts_with_content_key(self):
        data = [{"title": "Alpha", "body": "Content A"}, {"title": "Beta", "body": "Content B"}]
        results = list(parse_json(data, content_key="body"))
        assert len(results) == 2
        assert results[0][0] == "Content A"
        assert results[1][0] == "Content B"

    def test_json_object_values(self):
        data = '{"fact1": "Python is great", "fact2": "SQLite is fast", "_internal": "skip me"}'
        results = list(parse_json(data))
        assert len(results) == 2  # _internal is skipped
        contents = {r[0] for r in results}
        assert "Python is great" in contents
        assert "SQLite is fast" in contents

    def test_json_empty_list(self):
        results = list(parse_json([]))
        assert len(results) == 0

    def test_json_mixed_types(self):
        data = ["string fact", 42, True, None]
        results = list(parse_json(data))
        assert len(results) == 3  # None is skipped
        contents = [r[0] for r in results]
        assert "string fact" in contents
        assert "42" in contents  # int becomes str
        assert "True" in contents  # bool becomes str


# ======================================================================
# CSV parsing
# ======================================================================


class TestParseCsv:
    def test_csv_basic(self):
        text = "name,role,project\nAlice,Engineer,Alpha\nBob,Designer,Beta\n"
        results = list(parse_csv(text))
        assert len(results) == 2
        # First row
        content0, meta0 = results[0]
        assert "name: Alice" in content0
        assert "role: Engineer" in content0
        assert "project: Alpha" in content0
        assert meta0["row_index"] == 0
        assert meta0["headers"] == ["name", "role", "project"]
        assert meta0["columns"]["name"] == "Alice"

    def test_csv_empty(self):
        text = "header1,header2\n"
        results = list(parse_csv(text))
        assert len(results) == 0

    def test_csv_single_row(self):
        text = "col1,col2\nval1,val2\n"
        results = list(parse_csv(text))
        assert len(results) == 1
        assert results[0][0] == "col1: val1, col2: val2"

    def test_csv_empty_cells(self):
        text = "a,b,c\n1,,3\n"
        results = list(parse_csv(text))
        assert len(results) == 1
        # Empty cells produce no "col: " pair
        assert "a: 1" in results[0][0]
        assert "c: 3" in results[0][0]
        assert "b:" not in results[0][0]


# ======================================================================
# Format detection
# ======================================================================


class TestDetectFormat:
    def test_detect_markdown(self):
        assert detect_format("notes.md") == "markdown"
        assert detect_format("README.markdown") == "markdown"

    def test_detect_json(self):
        assert detect_format("data.json") == "json"

    def test_detect_csv(self):
        assert detect_format("spreadsheet.csv") == "csv"

    def test_detect_text(self):
        assert detect_format("readme.txt") == "text"

    def test_detect_json_from_content(self):
        assert detect_format("unknown", text='{"key": "value"}') == "json"
        assert detect_format("unknown", text='[1, 2, 3]') == "json"

    def test_detect_fallback_to_text(self):
        assert detect_format("readme", text="just some text") == "text"


# ======================================================================
# Store integration — ingest method
# ======================================================================


class TestStoreIngest:
    def test_ingest_format_auto_file(self, store, tmp_path):
        """Auto-detect from file extension."""
        md_file = tmp_path / "notes.md"
        md_file.write_text(
            "## Title\nContent here.\n## Another\nMore content.\n",
            encoding="utf-8",
        )
        stats = store.ingest(str(md_file))
        assert stats["total"] == 2
        assert stats["created"] == 2

    def test_ingest_dedup(self, store):
        """Repeated ingest of same content increments duplicate_count."""
        text = "## Section\nThis is a unique test fact for dedup checking.\n"
        stats1 = store.ingest(text, format="markdown")
        assert stats1["total"] == 1
        assert stats1["created"] == 1

        stats2 = store.ingest(text, format="markdown")
        assert stats2["total"] == 1
        assert stats2["deduped"] == 1

    def test_ingest_project(self, store):
        """Facts are tagged with the specified project."""
        text = "## Work\nImportant project fact.\n"
        stats = store.ingest(text, format="markdown", project="ingest-test")
        assert stats["total"] == 1
        assert stats["created"] == 1

        facts = store.list_facts(project="ingest-test")
        assert len(facts) >= 1
        assert facts[0]["project"] == "ingest-test"

    def test_ingest_stats_returned(self, store):
        """Stats dict has correct structure."""
        text = "## A\nFact A.\n## B\nFact B.\n## C\nFact C.\n"
        stats = store.ingest(text, format="markdown")
        assert isinstance(stats, dict)
        assert set(stats.keys()) == {"total", "created", "deduped", "errors"}
        assert stats["total"] == 3
        assert stats["created"] == 3
        assert stats["deduped"] == 0
        assert stats["errors"] == 0

    def test_ingest_errors_reported(self, store):
        """Malformed input is handled without crashing."""
        # An empty string — should produce 0 facts
        stats = store.ingest("", format="text")
        assert stats["total"] == 0
        assert stats["errors"] == 0

    def test_ingest_batch_commit(self, store):
        """batch_size is respected by inserting enough facts to cross a boundary."""
        lines = "\n".join([f"Line {i}" for i in range(10)])
        stats = store.ingest(lines, format="text", delimiter="line", batch_size=3)
        # Expect 10 facts
        assert stats["total"] == 10
        assert stats["created"] == 10

        # Verify facts were actually stored
        facts = store.list_facts()
        assert len(facts) >= 10

    def test_ingest_json_list(self, store):
        """JSON list of strings is ingested correctly."""
        data = json.dumps(["alpha fact", "beta fact", "gamma fact"])
        stats = store.ingest(data, format="json")
        assert stats["total"] == 3
        assert stats["created"] == 3

    def test_ingest_json_with_content_key(self, store):
        """JSON list of dicts with content_key extracts the right field."""
        data = json.dumps([
            {"text": "First item", "extra": "ignored"},
            {"text": "Second item"},
        ])
        stats = store.ingest(data, format="json", content_key="text")
        assert stats["total"] == 2
        assert stats["created"] == 2

    def test_ingest_csv(self, store):
        """CSV input is parsed and ingested."""
        csv_text = "title,priority\nFix login,high\nUpdate docs,medium\n"
        stats = store.ingest(csv_text, format="csv")
        assert stats["total"] == 2
        assert stats["created"] == 2

    def test_ingest_text_lines(self, store):
        """Text line-by-line ingestion."""
        text = "one\ntwo\nthree\n"
        stats = store.ingest(text, format="text", delimiter="line")
        assert stats["total"] == 3
        assert stats["created"] == 3

    def test_ingest_markdown_from_string(self, store):
        """Markdown string (not file) is parsed correctly."""
        text = "## Intro\nHello world.\n## Conclusion\nGoodbye.\n"
        stats = store.ingest(text, format="markdown")
        assert stats["total"] == 2
        assert stats["created"] == 2

    def test_ingest_with_tags_and_category(self, store):
        """Tags and category are propagated to ingested facts."""
        text = "## A\nImportant fact.\n"
        stats = store.ingest(
            text, format="markdown",
            category="reference", tags="urgent,important",
        )
        assert stats["created"] == 1

        facts = store.list_facts(category="reference")
        matching = [f for f in facts if f["tags"] == "urgent,important"]
        assert len(matching) >= 1

    def test_ingest_logs_event(self, store):
        """Ingest completion is logged in the event log."""
        text = "## A\nFact A.\n## B\nFact B.\n"
        store.ingest(text, format="markdown", project="evt-test")

        events = store.get_event_log(event_type="ingest_completed")
        assert len(events) >= 1
        ev = events[0]
        assert ev["project"] == "evt-test"
        assert ev["metadata"]["total"] == 2
        assert ev["metadata"]["created"] == 2
        assert ev["metadata"]["format"] == "markdown"

    def test_ingest_with_path_object(self, store, tmp_path):
        """Path object is accepted as source."""
        p = tmp_path / "data.csv"
        p.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        stats = store.ingest(p, format="csv")
        assert stats["total"] == 2
        assert stats["created"] == 2

    def test_ingest_empty_file(self, store, tmp_path):
        """File with only whitespace returns zero stats."""
        p = tmp_path / "empty.txt"
        p.write_text("   \n\n  ", encoding="utf-8")
        stats = store.ingest(p)
        assert stats["total"] == 0

    def test_ingest_source_harness_is_ingest(self, store):
        """Ingested facts have source_harness='ingest' and source_kind set."""
        text = "## A\nCheck source field.\n"
        store.ingest(text, format="markdown")
        facts = store.list_facts()
        matching = [f for f in facts if f["source_harness"] == "ingest"]
        assert len(matching) >= 1
        assert matching[0]["source_kind"] == "markdown"
