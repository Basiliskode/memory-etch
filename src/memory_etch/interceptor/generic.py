"""Generic callable interceptor — wrap any LLM-like function to auto-capture facts.

Usage:

    from memory_etch.interceptor.generic import GenericInterceptor

    def my_llm(messages, **kwargs):
        return "response text"

    interceptor = GenericInterceptor(store, my_llm)
    wrapped = interceptor.wrap()
    reply = wrapped(messages=[{"role": "user", "content": "Hello"}])

Or as a context manager::

    with GenericInterceptor(store, my_llm) as wrapped:
        reply = wrapped(messages=[{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional

from memory_etch import EtchStore

logger = logging.getLogger(__name__)

MessagesT = list[dict[str, Any]]


class GenericInterceptor:
    """Wraps a callable ``fn(messages, **kwargs) -> str`` to auto-capture facts.

    Each invocation stores two facts — one for the user messages and one for
    the assistant response — with ``category="conversation"``.
    """

    def __init__(
        self,
        store: EtchStore,
        fn: Callable[..., str],
        conversation_id: Optional[str] = None,
    ) -> None:
        self._store = store
        self._original_fn = fn
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._teardown_called = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wrap(self) -> Callable[..., str]:
        """Return a wrapped version of the original callable.

        The wrapped callable has the same signature as the original but
        also stores conversation facts in the EtchStore.
        """

        original = self._original_fn
        store = self._store
        conv_id = self._conversation_id
        _turn_counter = 0

        def wrapped(messages: MessagesT, **kwargs: Any) -> str:
            nonlocal _turn_counter
            _turn_counter += 1

            # 1. Capture user turn (if not torn down)
            if not self._teardown_called:
                user_text = _format_messages(messages)
                store.add_fact(
                    content=f"user: {user_text}",
                    category="conversation",
                    tags="interceptor,role:user",
                    topic_key=f"conversation/{conv_id}/t{_turn_counter:04d}/user",
                    trust_score=0.9,
                    importance=0.5,
                )

            # 2. Call original
            result = original(messages, **kwargs)

            # 3. Capture assistant turn (if not torn down)
            if not self._teardown_called:
                store.add_fact(
                    content=f"assistant: {result}",
                    category="conversation",
                    tags="interceptor,role:assistant",
                    topic_key=f"conversation/{conv_id}/t{_turn_counter:04d}/assistant",
                    trust_score=0.9,
                    importance=0.5,
                )

            return result

        return wrapped

    def teardown(self) -> None:
        """Restore the original callable (stop capturing facts).

        After calling this, previously-wrapped references still call the
        original function but no longer store facts.
        """
        self._teardown_called = True

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Callable[..., str]:
        return self.wrap()

    def __exit__(self, *args: Any) -> None:
        self.teardown()


def _format_messages(messages: MessagesT) -> str:
    """Join message roles and content into a single string.

    For simple turns (single user message), returns just the text.
    For multi-message lists, returns a newline-separated block.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            if len(messages) == 1:
                return content
            parts.append(f"{role}: {content}")
    if not parts:
        return str(messages)
    return "\n".join(parts)
