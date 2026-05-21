"""OpenAI ChatCompletion interceptor — captured as memory-etch facts.

Monkey-patches ``openai.OpenAI.chat.completions.create`` at the class level
of the resource to auto-capture every request/response turn.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from memory_etch import EtchStore
    from memory_etch.interceptor import InterceptorHandle

logger = logging.getLogger(__name__)


def install_openai(
    store: "EtchStore",
    conversation_id: Optional[str] = None,
) -> "InterceptorHandle":
    """Monkey-patch the OpenAI ChatCompletion create method.

    Patches ``openai.OpenAI.chat.completions.create`` at the resource class
    level so all client instances are captured.

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

    import openai  # lazy import — only imported when this target is selected

    conv_id = conversation_id or str(uuid.uuid4())

    # The target method lives on the resource class.
    # In openai v1+: openai.resources.chat.completions.Completions.create
    target = openai.OpenAI.chat.completions
    original_create = target.create

    _turn_counter = 0

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _turn_counter
        _turn_counter += 1

        # Handle both instance-bound calls (self passed) and direct attribute calls.
        # In bound-method context, self is the Completions resource instance;
        # in direct mock-chain context, there is no self.
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not supported in memory-etch interceptor v1. "
                "Set stream=False or use a non-streaming call."
            )

        # Extract messages from kwargs (works in both bound and unbound contexts)
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
                )

        # 2. Call original (pass all args/kwargs as-is, preserving self if bound)
        result = original_create(*args, **kwargs)

        # 3. Store assistant turn
        try:
            assistant_text = result.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            assistant_text = str(result)

        store.add_fact(
            content=f"assistant: {assistant_text}",
            category="conversation",
            tags=f"interceptor,role:assistant,model:{model}",
            topic_key=f"conversation/{conv_id}/t{_turn_counter:04d}/assistant",
            trust_score=0.9,
            importance=0.5,
        )

        return result

    target.create = _wrapper  # type: ignore[method-assign]

    def _teardown() -> None:
        target.create = original_create  # type: ignore[method-assign]

    return InterceptorHandle(
        name="openai",
        original=original_create,
        wrapped=_wrapper,
        teardown=_teardown,
    )
