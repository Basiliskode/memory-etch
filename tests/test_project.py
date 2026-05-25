"""Tests for project detection functionality."""

import os
import tempfile
from pathlib import Path

import pytest
from memento import EtchStore
from memento.project import detect_project


class TestDetectProject:
    def test_git_repo_remote(self, tmp_path):
        """detect_project returns repo name from git remote."""
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        # Init git repo
        _run_git(repo_dir, "init")
        _run_git(repo_dir, "remote", "add", "origin", "git@github.com:user/my-repo.git")
        name = detect_project(str(repo_dir))
        assert name == "my-repo"

    def test_git_repo_https_remote(self, tmp_path):
        """detect_project handles HTTPS remotes."""
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        _run_git(repo_dir, "init")
        _run_git(repo_dir, "remote", "add", "origin", "https://github.com/user/my-repo.git")
        name = detect_project(str(repo_dir))
        assert name == "my-repo"

    def test_git_repo_without_remote(self, tmp_path):
        """detect_project falls back to basename when no remote."""
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        _run_git(repo_dir, "init")
        name = detect_project(str(repo_dir))
        assert name == "myrepo"

    def test_non_git_dir_returns_basename(self, tmp_path):
        """detect_project returns basename for non-git dir."""
        d = tmp_path / "some-project"
        d.mkdir()
        name = detect_project(str(d))
        assert name == "some-project"

    def test_nonexistent_dir_returns_none(self):
        """detect_project returns None for nonexistent dir."""
        name = detect_project("/nonexistent/path/xyz123")
        assert name is None

    def test_result_is_lowercased(self, tmp_path):
        """detect_project lowercases the result."""
        repo_dir = tmp_path / "MyProject"
        repo_dir.mkdir()
        _run_git(repo_dir, "init")
        _run_git(repo_dir, "remote", "add", "origin", "git@github.com:user/MyProject.git")
        name = detect_project(str(repo_dir))
        assert name == "myproject"

    def test_detect_project_cwd_default(self):
        """detect_project works with default (cwd)."""
        name = detect_project()
        # Should be the basename of cwd, lowercased
        cwd_basename = os.path.basename(os.getcwd()).lower()
        assert name == cwd_basename


class TestEtchStoreProjectAuto:
    def test_project_auto_detects_from_cwd(self):
        """EtchStore(project='auto') detects project from cwd."""
        store = EtchStore(":memory:", project="auto", auto_migrate=True)
        try:
            assert store._project is not None
            assert isinstance(store._project, str)
            assert len(store._project) > 0
        finally:
            store.close()

    def test_project_none_maintains_backward_compat(self):
        """EtchStore with default project=None doesn't break."""
        store = EtchStore(":memory:", auto_migrate=True)
        try:
            assert store._project is None
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_git(workdir: Path, *args: str) -> None:
    """Run a git command in the given working directory."""
    import subprocess
    result = subprocess.run(
        ["git", *args],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
