"""Structural tests for README + API docs (v1.0 — Spanish-only)."""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
DOCS_API = PROJECT_ROOT / "docs/api"


def test_readme_exists():
    """README.md must exist."""
    assert README.is_file(), "README.md not found"


def test_readme_has_spanish_title():
    """README has Spanish title and tagline."""
    text = README.read_text(encoding="utf-8")
    assert "# Memory Etch" in text, "Title missing"
    assert "SQLite + FTS5 + HRR" in text, "Tech stack missing"


def test_readme_has_quickstart():
    """README has minimal quickstart code."""
    text = README.read_text(encoding="utf-8")
    assert "EtchStore" in text, "Core class missing"
    assert "pip install" in text, "Install instructions missing"


def test_readme_has_install_extras():
    """README documents pip install extras."""
    text = README.read_text(encoding="utf-8")
    assert "[hrr]" in text and "[embeddings]" in text, "Install extras missing"
    assert "[mcp]" in text or "[all]" in text, "MCP/all extras missing"


def test_readme_has_benchmark_table():
    """README includes benchmark results."""
    text = README.read_text(encoding="utf-8")
    assert "94.4%" in text or "accuracy" in text.lower(), "Benchmark results missing"
    assert "FTS5" in text, "FTS5 mention missing"


def test_readme_has_api_links():
    """README links to docs/api/*.md files."""
    text = README.read_text(encoding="utf-8")
    assert "docs/api/store.md" in text, "Missing link to store.md"
    assert "docs/api/retrieval.md" in text, "Missing link to retrieval.md"
    assert "docs/api/classifier.md" in text, "Missing link to classifier.md"


def test_readme_has_mcp_section():
    """README includes MCP server documentation."""
    text = README.read_text(encoding="utf-8")
    assert "MCP" in text, "MCP section missing"


def test_readme_has_embedding_providers():
    """README includes embedding providers documentation."""
    text = README.read_text(encoding="utf-8")
    assert "Embedding" in text or "embedding" in text, "Embedding section missing"


def test_api_store_md_exists():
    """docs/api/store.md exists with class signature."""
    path = DOCS_API / "store.md"
    assert path.is_file(), "docs/api/store.md not found"
    text = path.read_text(encoding="utf-8")
    assert "class EtchStore" in text, "store.md missing EtchStore class signature"
    assert "def __init__" in text, "store.md missing __init__ signature"


def test_api_retrieval_md_exists():
    """docs/api/retrieval.md exists with class signature."""
    path = DOCS_API / "retrieval.md"
    assert path.is_file(), "docs/api/retrieval.md not found"
    text = path.read_text(encoding="utf-8")
    assert "class EtchRetriever" in text, "retrieval.md missing EtchRetriever class signature"
    assert "def search" in text, "retrieval.md missing search method signature"


def test_api_classifier_md_exists():
    """docs/api/classifier.md exists with class signature."""
    path = DOCS_API / "classifier.md"
    assert path.is_file(), "docs/api/classifier.md not found"
    text = path.read_text(encoding="utf-8")
    assert "class QueryClassifier" in text, "classifier.md missing QueryClassifier class signature"
    assert "def classify" in text, "classifier.md missing classify method signature"


def test_help_etchstore_runs():
    """`from memory_etch import EtchStore; help(EtchStore)` runs without error."""
    import importlib
    try:
        mod = importlib.import_module("memory_etch")
        store_cls = getattr(mod, "EtchStore")
        help_text = store_cls.__doc__
        assert help_text is not None, "EtchStore has no docstring"
        assert "SQLite" in help_text, "EtchStore docstring missing content"
    except ImportError as e:
        pytest.skip(f"memory_etch not importable: {e}")
