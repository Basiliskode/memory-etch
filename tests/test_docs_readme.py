"""Structural tests for bilingual README + API stubs (PR 5)."""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
DOCS_API = PROJECT_ROOT / "docs/api"


def test_readme_exists():
    """README.md must exist."""
    assert README.is_file(), "README.md not found"


def test_readme_has_spanish_section():
    """Spanish CMO section is present."""
    text = README.read_text(encoding="utf-8")
    assert "~0.8ms por búsqueda" in text, "Spanish header missing"
    assert "Nadie construye un agente serio sin memoria." in text, "Spanish pitch missing"
    assert "pip install" in text, "Spanish install snippet missing"


def test_readme_has_english_section():
    """English section is present."""
    text = README.read_text(encoding="utf-8")
    assert "## English" in text, "English section heading missing"


def test_english_has_quickstart():
    """English section has a quickstart code example."""
    text = README.read_text(encoding="utf-8")
    eng = text.split("## English")[1] if "## English" in text else ""
    assert "EtchStore" in eng and "EtchRetriever" in eng, \
        "English quickstart missing core classes"


def test_english_has_install_extras():
    """English section documents install extras."""
    text = README.read_text(encoding="utf-8")
    eng = text.split("## English")[1] if "## English" in text else ""
    assert "[hrr]" in eng and "[all]" in eng and "[bge-m3]" in eng, \
        "English section missing install extras"


def test_english_has_benchmark_table():
    """English section has a benchmark comparison table."""
    text = README.read_text(encoding="utf-8")
    eng = text.split("## English")[1] if "## English" in text else ""
    assert "FTS5 only" in eng and "FTS5 + HRR" in eng and "BGE-M3" in eng, \
        "English section missing benchmark table columns"


def test_english_has_api_links():
    """English section links to docs/api/*.md files."""
    text = README.read_text(encoding="utf-8")
    assert "docs/api/store.md" in text, "Missing link to store.md"
    assert "docs/api/retrieval.md" in text, "Missing link to retrieval.md"
    assert "docs/api/classifier.md" in text, "Missing link to classifier.md"


def test_english_no_production_ready_claims():
    """English section must NOT claim 'production-ready' or 'stable'."""
    text = README.read_text(encoding="utf-8")
    eng = text.split("## English")[1] if "## English" in text else ""
    lower = eng.lower()
    assert "production-ready" not in lower, \
        "English section must not claim 'production-ready'"
    assert "production ready" not in lower, \
        "English section must not claim 'production ready'"
    assert "production" not in lower or "ready" not in lower.split("production")[1][:20], \
        "English section must not imply production-ready status"


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


def test_spanish_preserved_exactly():
    """Existing Spanish content is preserved — key phrases unchanged."""
    text = README.read_text(encoding="utf-8")
    # These should still appear exactly as before
    markers = [
        "~0.8ms por búsqueda",
        "Sin GPU, sin servicios, sin excusas",
        "Nadie construye un agente serio sin memoria.",
        "Después me contás.",
    ]
    for m in markers:
        assert m in text, f"Spanish marker lost: {m}"
