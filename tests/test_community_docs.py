"""Structural tests for community docs and GitHub templates (PR 4)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def assert_has(path: str, snippets: tuple[str, ...]) -> None:
    content = read(path)
    missing = [snippet for snippet in snippets if snippet not in content]
    assert not missing, f"{path} missing {missing}"
def test_contributing_guides_setup_tests_linting_and_pr_flow():
    assert_has(
        "CONTRIBUTING.md",
        (
            "# Contributing to Memory Etch",
            "## Quick start",
            'pip install -e ".[dev]"',
            "python -m pytest tests/",
            "python -m ruff check",
            "python -m ruff format",
            "## Pull request checklist",
            "## Commit convention",
            "## Changelog guidance",
            "- [ ]",
        ),
    )
def test_security_and_code_of_conduct_define_safe_reporting():
    assert_has(
        "SECURITY.md",
        (
            "# Security Policy",
            "## Reporting a vulnerability",
            "Please do not open a public issue",
            "GitHub's private vulnerability reporting",
            "## What to include",
            "## Response expectations",
        ),
    )
    assert_has(
        "CODE_OF_CONDUCT.md",
        (
            "# Contributor Covenant Code of Conduct",
            "## Our Pledge",
            "## Our Standards",
            "## Enforcement Responsibilities",
            "## Enforcement Guidelines",
            "## Reporting",
            "Contributor Covenant, version 2.1",
        ),
    )
def test_changelog_uses_keep_a_changelog_shape():
    assert_has(
        "CHANGELOG.md",
        (
            "# Changelog",
            "All notable changes to this project will be documented in this file.",
            "Keep a Changelog",
            "Semantic Versioning",
            "## [Unreleased]",
            "## [0.2.0]",
            "## [0.1.0]",
        ),
    )
def test_issue_templates_guide_bug_and_feature_reports():
    bug = read(".github/ISSUE_TEMPLATE/bug_report.md")
    feature = read(".github/ISSUE_TEMPLATE/feature_request.md")
    assert bug.startswith("---\n")
    assert feature.startswith("---\n")
    for snippet in ("name: Bug report", "## Steps to reproduce", "## Logs"):
        assert snippet in bug
    for snippet in ("name: Feature request", "## Local-first impact", "## Scope check"):
        assert snippet in feature
def test_pr_template_and_gitattributes_support_review_flow():
    assert_has(
        ".github/PULL_REQUEST_TEMPLATE.md",
        (
            "## Summary",
            "Closes #",
            "CONTRIBUTING.md",
            "## Checklist",
            "- [ ] Tests added or updated",
            "- [ ] `python -m pytest tests/` passes",
            "- [ ] `python -m ruff check",
            "- [ ] Changelog updated or not needed",
        ),
    )
    assert_has(".gitattributes", ("* text=auto", "*.py text diff=python", "*.md text"))
