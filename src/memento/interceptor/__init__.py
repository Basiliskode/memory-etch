"""Interceptor pattern — monkey-patches LLM SDKs to auto-capture facts.

Usage:

    from memento import EtchStore
    from memento.interceptor import intercept, teardown_all

    store = EtchStore(":memory:")
    handles = intercept(store)
    # ... use OpenAI / Anthropic as normal ...
    teardown_all(handles)

Each call to the LLM stores a "user" and "assistant" fact automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InterceptorHandle
# ---------------------------------------------------------------------------


@dataclass
class InterceptorHandle:
    """A handle returned by :func:`intercept` for a single wrapped SDK.

    Attributes:
        name: Target name (e.g. ``"openai"``, ``"anthropic"``).
        original: The original unpatched callable.
        wrapped: The replacement wrapper.
        teardown: Callable that restores the original.
    """

    name: str
    original: Any
    wrapped: Any
    teardown: Callable[[], None]


# ---------------------------------------------------------------------------
# Known target registry
# ---------------------------------------------------------------------------

_KNOWN_TARGETS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
}


def _normalize_targets(
    targets: Optional[list[str] | str] = None,
) -> list[str]:
    """Normalize the ``targets`` argument into a list of target names."""
    if targets is None:
        return list(_KNOWN_TARGETS.keys())
    if isinstance(targets, str):
        targets = [targets]
    return list(targets)


def _validate_targets(targets: list[str]) -> None:
    """Raise ``ValueError`` for any unknown target name."""
    for name in targets:
        if name not in _KNOWN_TARGETS:
            raise ValueError(f"Unknown target: {name}")


def _import_target(name: str) -> None:
    """Lazy-import the SDK for *name*; raises ``ImportError`` if missing."""
    _import_map = {
        "openai": "openai",
        "anthropic": "anthropic",
    }
    sdk_name = _import_map[name]
    try:
        __import__(sdk_name)
    except ImportError:
        raise ImportError(
            f"SDK '{sdk_name}' is required for interceptor target '{name}'. "
            f"Install: pip install memento[{name}]"
        )


def _install_target(
    store: "EtchStore",
    name: str,
    conversation_id: Optional[str] = None,
) -> InterceptorHandle:
    """Install a single target wrapper and return its handle."""
    if name == "openai":
        from .openai import install_openai  # type: ignore[import-untyped]

        return install_openai(store, conversation_id)
    elif name == "anthropic":
        from .anthropic import install_anthropic  # type: ignore[import-untyped]

        return install_anthropic(store, conversation_id)
    else:
        raise ValueError(f"Unknown target: {name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def intercept(
    store: "EtchStore",
    targets: Optional[list[str] | str] = None,
    conversation_id: Optional[str] = None,
) -> list[InterceptorHandle]:
    """Install interceptors for one or more LLM SDKs.

    Args:
        store: An initialized ``EtchStore``.
        targets: Which wrappers to install. ``None`` (default) tries all known
            SDKs but silently skips any whose package is not installed.
            A single string or list of strings selects specific targets.
        conversation_id: Optional conversation ID. Auto-generated UUIDv4 if
            omitted.

    Returns:
        List of :class:`InterceptorHandle` objects, one per installed wrapper.

    Raises:
        ValueError: If a target name is not recognised.
        ImportError: If a requested target's SDK is not installed.
    """
    names = _normalize_targets(targets)
    _validate_targets(names)
    is_auto_detect = targets is None

    handles: list[InterceptorHandle] = []
    for name in names:
        if is_auto_detect:
            try:
                _import_target(name)
            except ImportError:
                logger.info("SDK not installed for target '%s' — skipping", name)
                continue
        else:
            _import_target(name)
        handle = _install_target(store, name, conversation_id)
        handles.append(handle)
    return handles


def teardown_all(handles: list[InterceptorHandle]) -> None:
    """Restore all original SDK functions that were patched by *handles*.

    Idempotent — calling multiple times is safe.
    """
    for handle in handles:
        try:
            handle.teardown()
        except Exception:
            logger.exception("Interceptor teardown failed for %s", handle.name)
