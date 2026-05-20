"""Project detection utilities for Memory Etch.

Provides ``detect_project()`` to extract a project name from the current
working directory — either from the git remote URL or the directory basename.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_project(directory: Optional[str] = None) -> Optional[str]:
    """Detect the project name from a directory.

    Strategy:
    1. If ``directory`` is a git repo, try ``git remote get-url origin``
       and extract the repo name from the URL (strip ``.git``, take last path
       component).
    2. If no git remote exists, fall back to the basename of the directory.
    3. Return ``None`` only if the directory does not exist.

    The result is lowercased.

    Args:
        directory: Path to detect the project from. Defaults to the current
            working directory.

    Returns:
        Lowercased project name string, or ``None`` if the directory doesn't
        exist.
    """
    target = Path(directory or os.getcwd()).resolve()

    if not target.exists():
        return None

    # Try git remote first
    name = _detect_from_git_remote(target)
    if name:
        return name.lower()

    # Fallback: directory basename
    return target.name.lower()


def _detect_from_git_remote(directory: Path) -> Optional[str]:
    """Extract the repo name from ``git remote get-url origin``.

    Tries to run ``git remote get-url origin`` in the given directory.
    Returns the repo name (last path component, stripped of ``.git``),
    or ``None`` on any failure.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        url = result.stdout.strip()
        if not url:
            return None

        # Parse repo name from URL — take last path component, strip .git
        # Handles both SSH (git@github.com:user/repo.git) and HTTPS URLs
        url = url.replace("\\", "/")
        # Normalize SSH-style colon separator to slash
        if "://" not in url:
            # e.g. git@github.com:user/repo.git → split on ':'
            url = url.replace(":", "/", 1) if ":" in url else url

        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name if name else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("Git remote detection failed: %s", exc)
        return None
