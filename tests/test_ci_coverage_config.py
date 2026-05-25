"""Tests for CI hardening and coverage config (PR 3)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
COVERAGERC = PROJECT_ROOT / ".coveragerc"
DEPENDABOT = PROJECT_ROOT / ".github" / "dependabot.yml"


def test_ci_workflow_targets_main_for_push_and_pull_request():
    """CI-1: CI runs for pushes and pull requests targeting main."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "push:" in content
    assert "pull_request:" in content
    assert content.count("branches: [main]") >= 2


def test_ci_workflow_uses_supported_python_matrix():
    """CI-1: CI matrix covers Python 3.10, 3.11, and 3.12."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in content


def test_ci_workflow_installs_quality_and_coverage_tools():
    """CI-2: CI installs test, coverage, and lint tooling explicitly."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "pip install" in content
    for package in ("pytest", "pytest-cov", "ruff"):
        assert package in content


def test_ci_workflow_uses_pip_cache_keyed_by_pyproject():
    """CI-1: setup-python enables pip caching keyed by pyproject.toml."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "cache: pip" in content
    assert "cache-dependency-path: pyproject.toml" in content


def test_ci_workflow_runs_ruff_check_and_format_check():
    """CI-2: CI runs Ruff check and Ruff format in check mode."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "ruff check" in content
    assert "--no-fix" in content
    assert "ruff format" in content
    assert "--check" in content


def test_ci_workflow_documents_temporary_ruff_baseline_scope():
    """CI-2: CI documents the temporary Ruff scope until legacy source is fixed."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "Temporary baseline" in content
    assert "translator" in content
    assert "python -m ruff check tests/test_ci_coverage_config.py --no-fix" in content
    assert "python -m ruff format tests/test_ci_coverage_config.py --check" in content


def test_ci_workflow_runs_pytest_with_coverage_threshold():
    """CI-2/CI-3: CI runs pytest-cov with an 80 percent threshold."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python -m pytest tests/" in content
    assert "--cov=memento" in content
    assert "--cov-fail-under=80" in content


def test_ci_workflow_uploads_coverage_artifacts():
    """CI-4: CI uploads coverage reports as an artifact."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "actions/upload-artifact@v7" in content
    assert "coverage" in content
    assert "htmlcov" in content


def test_coveragerc_sets_source_and_omits_non_core_paths():
    """CI-3: coverage is measured from src and omits tests/plugins/translator/benchmark."""
    content = COVERAGERC.read_text(encoding="utf-8")

    assert "[run]" in content
    assert "source =" in content
    assert "src" in content
    for omitted in (
        "*/tests/*",
        "*/plugins/*",
        "*/translator_deprecated/*",
        "src/memento/benchmark/*",
    ):
        assert omitted in content


def test_coveragerc_sets_report_threshold_and_exclusions():
    """CI-3: coverage report sets fail_under=80 and excludes defensive lines."""
    content = COVERAGERC.read_text(encoding="utf-8")

    assert "[report]" in content
    assert "fail_under = 80" in content
    assert "pragma: no cover" in content
    assert "def _call_llm_extract" in content


def test_dependabot_configures_weekly_pip_updates():
    """CI-6: Dependabot checks Python packaging dependencies weekly."""
    content = DEPENDABOT.read_text(encoding="utf-8")

    assert 'package-ecosystem: "pip"' in content
    assert 'directory: "/"' in content
    assert 'interval: "weekly"' in content


def test_dependabot_configures_weekly_github_actions_updates():
    """CI-6: Dependabot checks GitHub Actions dependencies weekly."""
    content = DEPENDABOT.read_text(encoding="utf-8")

    assert 'package-ecosystem: "github-actions"' in content
    assert 'directory: "/"' in content
    assert 'interval: "weekly"' in content
