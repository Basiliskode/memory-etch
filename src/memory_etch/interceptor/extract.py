"""Conversation extraction — run LLM extraction on stored conversation turns.

The user provides a callback ``llm_extract_fn(conversation_text: str) -> list[str]``
that receives the full conversation text and returns a list of extracted fact
strings. Each returned string is stored as a fact with ``category="extracted"``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from memory_etch import EtchStore

logger = logging.getLogger(__name__)

# A user-supplied extraction function.
# Receives full conversation text, returns list of extracted fact strings.
ExtractFn = Callable[[str], list[str]]


def extract_conversation(
    conversation: list[dict[str, Any]],
    store: EtchStore,
    llm_extract_fn: ExtractFn,
) -> int:
    """Run an LLM extraction callback on conversation turns and store results.

    Args:
        conversation: List of fact dicts. Each dict should have at least
            ``content`` and optionally ``role`` (extracted from tags).
        store: An initialized ``EtchStore``.
        llm_extract_fn: Callback ``(conversation_text: str) -> list[str]``
            that returns extracted fact strings.

    Returns:
        Number of facts extracted and stored.
    """
    # Build full conversation text from turns
    lines: list[str] = []
    for turn in conversation:
        content = turn.get("content", "")
        role = turn.get("role", "")
        if role:
            lines.append(f"{role}: {content}" if not content.startswith(f"{role}: ") else content)
        else:
            lines.append(content)

    conversation_text = "\n".join(lines)

    # Run extraction
    extracted: list[str] = llm_extract_fn(conversation_text)

    # Store each extracted fact
    for fact_str in extracted:
        store.add_fact(
            content=fact_str,
            category="extracted",
            tags="interceptor,extracted",
            trust_score=0.9,
            importance=0.5,
            source_harness="interceptor",
            source_kind="conversation",
        )

    return len(extracted)
