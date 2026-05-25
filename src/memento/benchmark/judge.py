"""Gemini-based judge for evaluating memory retrieval quality.

The judge receives a question, the retrieved context, and the expected
answer, then determines whether the context is sufficient to answer
the question correctly.
"""

import os
from typing import Optional


class GeminiJudge:
    """LLM-based judge that evaluates retrieval quality.

    Args:
        model: Gemini model name (default: gemini-2.5-flash-lite).
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.

    Example::

        judge = GeminiJudge()
        verdict, meta = judge.judge(
            question="What is Bob's profession?",
            retrieved_context="Bob Martinez works as an architect...",
            gold_answer="architect",
        )
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def judge(self, question: str, retrieved_context: str, gold_answer: str) -> tuple[Optional[bool], dict]:
        """Judge if the retrieved context supports the gold answer.

        Returns:
            tuple[bool | None, dict]: (verdict, metadata) where verdict is
            True/False or None if the API call failed.
        """
        if not self._api_key:
            return None, {"error": "No GEMINI_API_KEY set"}

        try:
            from google import genai
            client = genai.Client(api_key=self._api_key)

            prompt = f"""You are a strict judge evaluating a memory retrieval system.

Given a QUESTION, the RETRIEVED CONTEXT, and the EXPECTED ANSWER, determine if the
retrieved context contains enough information to answer the question correctly.

QUESTION: {question}

RETRIEVED CONTEXT:
{retrieved_context}

EXPECTED ANSWER: {gold_answer}

RESPOND WITH ONLY: YES or NO
YES = The retrieved context contains enough information to answer the question correctly
NO = The retrieved context does NOT contain enough information

Your verdict:"""

            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 10},
            )

            prompt_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            verdict = response.text.strip().upper().startswith("YES")
            return verdict, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}

        except Exception as exc:
            return None, {"error": str(exc)}
