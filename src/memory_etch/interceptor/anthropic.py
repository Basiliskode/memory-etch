"""Anthropic Messages interceptor — captured as memory-etch facts.

Monkey-patches ``anthropic.Anthropic.messages.create`` at the resource
level to auto-capture every request/response turn.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from memory_etch import EtchStore
    from memory_etch.interceptor import InterceptorHandle

logger = logging.getLogger(__name__)


def install_anthropic(
    store: "EtchStore",
    conversation_id: Optional[str] = None,
) -> "InterceptorHandle":
    """Monkey-patch ``anthropic.Anthropic.messages.create``.

    Stores each user message and assistant response as facts with
    ``category="conversation"``.

    Args:
        store: An initialized ``EtchStore``.
        conversation_id: Optional conversation ID. Auto-generated UUIDv4 if
            omitted.

    Returns:
        An ``InterceptorHandle`` whose ``.teardown()`` restores the original.

    Raises:
        NotImplementedError: If ``stream=True`` is passed.
    """
    from memory_etch.interceptor import InterceptorHandle

    import anthropic  # lazy import — only imported when this target is selected

    conv_id = conversation_id or str(uuid.uuid4())

    # Target: anthropic.Anthropic.messages.create
    target = anthropic.Anthropic.messages
    original_create = target.create
    _turn_counter = 0

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _turn_counter
        _turn_counter += 1

        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not supported in memory-etch interceptor v1. "
                "Set stream=False or use a non-streaming call."
            )

        messages: list[dict] = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # 1. Store user turn(s)
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                store.add_fact(
                    content=f"{role}: {content}",
                    category="conversation",
                    tags=f"interceptor,role:{role},model:{model}",
                    topic_key=f"conversation/{conv_id}/t{_turn_counter:04d}/{role}",
                    trust_score=0.9,
                    importance=0.5,
                    source_harness="anthropic",
                    source_kind="conversation",
                    source_agent=model,
                )

        # 2. Call original
        result = original_create(*args, **kwargs)

        # 3. Store assistant turn
        try:
            # Anthropic v0.30+: result.content is a list of ContentBlock objects
            assistant_text = " ".join(
                block.text for block in result.content
                if hasattr(block, "text") and block.text
            )
        except (AttributeError, TypeError):
            try:
                assistant_text = str(result.content)
            except (AttributeError, TypeError):
                assistant_text = str(result)

        store.add_fact(
            content=f"assistant: {assistant_text}",
            category="conversation",
            tags=f"interceptor,role:assistant,model:{model}",
            topic_key=f"conversation/{conv_id}/t{_turn_counter:04d}/assistant",
            trust_score=0.9,
            importance=0.5,
            source_harness="anthropic",
            source_kind="conversation",
            source_agent=model,
        )

        return result

    target.create = _wrapper  # type: ignore[method-assign]

    def _teardown() -> None:
        target.create = original_create  # type: ignore[method-assign]

    return InterceptorHandle(
        name="anthropic",
        original=original_create,
        wrapped=_wrapper,
        teardown=_teardown,
    )
