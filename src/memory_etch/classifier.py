"""Lightweight classifier for re-routing queries to the right retrieval strategy.

The classifier assigns a query to a category based on keyword patterns.
This is used by the retriever to decide whether to do hybrid search,
entity probe, relation reasoning, or fall back to raw FTS5.
"""

import re
from typing import Optional


# Pattern maps — ordered by specificity (first match wins)
_INTENT_PATTERNS: list[tuple[str, str, list[str]]] = [
    ("entity", "what do i know about", ["entity", "about"]),
    ("entity", "tell me about", ["entity"]),
    ("entity", "what is", ["entity"]),
    ("entity", "who is", ["entity"]),
    ("probe", "probe", ["probe"]),
    ("project", "project:", ["project"]),
    ("project", r"^project ", ["project"]),
    ("relation", "relation", ["relation"]),
    ("relation", "between", ["relation"]),
    ("timeline", "timeline", ["timeline"]),
    ("timeline", "history of", ["timeline"]),
    ("search", "search", ["search"]),
    ("contradict", "contradict", ["contradict"]),
    ("contradict", "conflict", ["contradict"]),
]

_EMPTY_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^(hi|hello|hey|thanks|ok|okay|yes|no|sure|dale)$", re.IGNORECASE),
]


class QueryClassifier:
    """Simple rule-based classifier for memory queries."""

    def classify(self, query: str) -> dict:
        """Classify a query into an intent and extract entities.

        Uses keyword pattern matching to determine the query intent.
        Intent can be one of: ``search``, ``entity``, ``probe``,
        ``project``, ``relation``, ``timeline``, ``contradict``, ``empty``.

        Args:
            query: The user's query string.

        Returns:
            Dict with keys ``intent`` (str), ``entities`` (list[str]),
            and ``keywords`` (list[str]).
        """
        if not query or not query.strip():
            return {"intent": "empty", "entities": [], "keywords": []}

        q_lower = query.strip().lower()

        # Empty/greeting patterns
        for pat in _EMPTY_PATTERNS:
            if pat.match(q_lower):
                return {"intent": "empty", "entities": [], "keywords": []}

        # Intent from patterns
        for intent, pattern, _ in _INTENT_PATTERNS:
            if re.search(pattern, q_lower):
                return self._build(intent, q_lower)

        # Default: general search
        return self._build("search", q_lower)

    def _build(self, intent: str, q: str) -> dict:
        return {
            "intent": intent,
            "entities": self._extract_entities(q),
            "keywords": self._extract_keywords(q),
        }

    @staticmethod
    def _extract_entities(q: str) -> list[str]:
        """Extract capitalized phrases (potential entity names)."""
        return list(set(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", q)))

    @staticmethod
    def _extract_keywords(q: str) -> list[str]:
        """Extract meaningful keywords (nouns, no stopwords)."""
        stopwords = {
            "what", "when", "where", "why", "how", "who", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "a", "an", "the", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "about", "tell", "me",
            "know", "search", "find", "show", "list", "get",
            "i", "you", "we", "they", "he", "she", "it",
            "this", "that", "these", "those",
        }
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", q.lower())
        return [t for t in tokens if t not in stopwords]
