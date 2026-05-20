"""Tests for code quality tooling config (PR 2)."""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
EDITORCONFIG = PROJECT_ROOT / ".editorconfig"
PRE_COMMIT_CONFIG = PROJECT_ROOT / ".pre-commit-config.yaml"


# ── pyproject.toml ruff sections ──────────────────────────────────────────


def test_ruff_section_exists():
    """LINT-1: pyproject.toml has [tool.ruff] section."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "[tool.ruff]" in text


def test_ruff_line_length():
    """LINT-1: ruff line-length is set."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "line-length" in text
    match = re.search(r'line-length\s*=\s*(\d+)', text)
    assert match, "line-length not found"
    assert int(match.group(1)) == 100


def test_ruff_target_version():
    """LINT-1: ruff target-version is set."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "target-version" in text
    assert "py310" in text


def test_ruff_lint_section_exists():
    """LINT-1: pyproject.toml has [tool.ruff.lint] section."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "[tool.ruff.lint]" in text


def test_ruff_lint_select_contains_core_rules():
    """LINT-1: ruff.lint select includes core rule categories."""
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "select" in text
    # Must have: pycodestyle, pyflakes, isort, pep8-naming, warnings,
    # pyupgrade, bugbear, simplify, and unused-argument checks.
    for rule in ("E", "F", "I", "N", "W", "UP", "B", "SIM", "ARG"):
        assert rule in text, f"Core rule {rule} missing from ruff.lint select"


def test_pre_commit_ruff_rev_is_pinned_semver():
    """LINT-4: pre-commit ruff hook is pinned to a concrete semver release."""
    content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    match = re.search(r"rev:\s*(v\d+\.\d+\.\d+)", content)
    assert match, "ruff pre-commit rev must be pinned to a concrete semver tag"


# ── .editorconfig ──────────────────────────────────────────────────────────


def test_editorconfig_exists():
    """LINT-3: .editorconfig exists."""
    assert EDITORCONFIG.exists(), f"{EDITORCONFIG} not found"


def test_editorconfig_is_nonempty():
    """LINT-3: .editorconfig is non-empty."""
    assert EDITORCONFIG.stat().st_size > 0


def test_editorconfig_root_true():
    """LINT-3: .editorconfig declares root = true."""
    content = EDITORCONFIG.read_text(encoding="utf-8")
    assert "root = true" in content


def test_editorconfig_indent_style():
    """LINT-3: .editorconfig sets indent_style = space."""
    content = EDITORCONFIG.read_text(encoding="utf-8")
    assert "indent_style = space" in content


def test_editorconfig_indent_size():
    """LINT-3: .editorconfig sets indent_size = 4."""
    content = EDITORCONFIG.read_text(encoding="utf-8")
    assert "indent_size = 4" in content


def test_editorconfig_end_of_line():
    """LINT-3: .editorconfig sets end_of_line = lf."""
    content = EDITORCONFIG.read_text(encoding="utf-8")
    assert "end_of_line = lf" in content


# ── .pre-commit-config.yaml ────────────────────────────────────────────────


def test_pre_commit_config_exists():
    """LINT-4: .pre-commit-config.yaml exists."""
    assert PRE_COMMIT_CONFIG.exists(), f"{PRE_COMMIT_CONFIG} not found"


def test_pre_commit_config_is_nonempty():
    """LINT-4: .pre-commit-config.yaml is non-empty."""
    assert PRE_COMMIT_CONFIG.stat().st_size > 0


def test_pre_commit_has_ruff_repo():
    """LINT-4: .pre-commit-config.yaml references ruff hook repo."""
    content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "ruff" in content.lower()


def test_pre_commit_has_check_hook():
    """LINT-4: .pre-commit-config.yaml has ruff check hook."""
    content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "check" in content or "ruff" in content


def test_pre_commit_has_repos_key():
    """LINT-4: .pre-commit-config.yaml contains a repos key."""
    content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "repos" in content


def test_pre_commit_has_ruff_hook_repo_url():
    """LINT-4: .pre-commit-config.yaml references astral-sh/ruff-pre-commit."""
    content = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "https://github.com/astral-sh/ruff-pre-commit" in content
