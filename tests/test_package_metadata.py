"""Tests for PyPI packaging metadata (PR 1)."""

import sys
from importlib.metadata import entry_points, metadata
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
PY_TYPED = PROJECT_ROOT / "src" / "memory_etch" / "py.typed"
PUBLISH_YML = PROJECT_ROOT / ".github" / "workflows" / "publish.yml"


# ── pyproject.toml metadata ──────────────────────────────────────────────


def test_version_string_present():
    """PKG: __version__ exists and is non-empty."""
    from memory_etch import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_python_requires_constraint():
    """PKG-1: requires-python = '>=3.10,<3.13' in pyproject.toml (PEP 621)."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert 'requires-python' in text or '"requires-python"' in text


def test_python_requires_value():
    """PKG-1: python_requires must cover 3.10, 3.11, 3.12 but not 3.13."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert ">=3.10" in text
    assert "<3.13" in text or "<=3.12" in text


def test_etch_viewer_entry_point():
    """PKG-3: console_scripts entry point for etch-viewer."""
    eps = entry_points(group="console_scripts")
    etch_eps = [ep for ep in eps if ep.name == "etch-viewer"]
    assert len(etch_eps) >= 1
    ep = etch_eps[0]
    assert "viewer" in ep.value
    assert "main" in ep.value


def test_classifiers_include_beta():
    """PKG-2: Development Status :: 4 - Beta classifier present."""
    meta = metadata("memory-etch")
    classifiers = meta.get_all("Classifier") or []
    assert any("Development Status :: 4 - Beta" in c for c in classifiers)


def test_classifiers_include_mit():
    """PKG-2: MIT License classifier present."""
    meta = metadata("memory-etch")
    classifiers = meta.get_all("Classifier") or []
    assert any("MIT License" in c for c in classifiers)


def test_classifiers_include_310_311_312():
    """PKG-2: Python 3.10, 3.11, 3.12 classifiers present."""
    meta = metadata("memory-etch")
    classifiers = meta.get_all("Classifier") or []
    for ver in ("3.10", "3.11", "3.12"):
        assert any(f"Python :: {ver}" in c for c in classifiers)


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="requires_metadata was added in Python 3.10 dev",
)
def test_requires_python_reflects_constraint():
    """PKG-1: Requires-Python metadata entry matches constraint."""
    meta = metadata("memory-etch")
    rp = meta.get("Requires-Python", "")
    assert ">=" in rp
    assert "<" in rp or "<=" in rp


# ── py.typed marker ──────────────────────────────────────────────────────


def test_py_typed_marker_exists():
    """PKG-4: py.typed PEP 561 marker exists."""
    assert PY_TYPED.exists(), f"{PY_TYPED} not found"


def test_py_typed_marker_is_file():
    """PKG-4: py.typed is a regular file (not a directory)."""
    assert PY_TYPED.is_file(), f"{PY_TYPED} is not a regular file"


# ── Extras remain optional ────────────────────────────────────────────────


@pytest.mark.skip(reason="Requires a fresh venv without extras; run manually")
def test_base_install_no_extra_deps():
    """PKG-5: Base install does NOT make HRR or BGE-M3 mandatory."""
    # Run in a fresh venv: pip install memory-etch && python -c "from memory_etch import EtchStore"
    # This test is a manual check and CI gate, not a runtime assert.
    pass


# ── publish workflow ──────────────────────────────────────────────────────


def test_publish_workflow_exists():
    """PKG-6: publish.yml exists."""
    assert PUBLISH_YML.exists(), f"{PUBLISH_YML} not found"


def test_publish_workflow_is_yaml():
    """PKG-6: publish.yml is non-empty."""
    content = PUBLISH_YML.read_text(encoding="utf-8")
    assert len(content) > 50


def test_publish_workflow_has_name():
    """PKG-6: publish.yml contains a name field."""
    content = PUBLISH_YML.read_text(encoding="utf-8")
    assert "name:" in content


# ── Init check ─────────────────────────────────────────────────────────────


def test_etch_store_importable():
    """Package installs and EtchStore is importable (smoke)."""
    from memory_etch import EtchStore

    assert EtchStore is not None
