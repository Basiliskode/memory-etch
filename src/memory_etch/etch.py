"""Hermes Memory Provider for Memory Etch — bridges EtchStore into the agent lifecycle."""
import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .store import EtchStore

logger = logging.getLogger(__name__)

# Default extraction config
_DEFAULT_CONFIG = {
    "auto_extract_llm": True,
    "extract_interval": 5,
    "extract_min_meaningful": 3,
    "extract_min_buffer": 5,
    "extract_max_batch": 20,
    "db_path": "",
}

# Patterns to detect meaningful content (heuristic)
_MEANINGFUL_MIN_LENGTH = 15

# System prompt for LLM extraction
_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a conversation turn, extract factual statements that should be remembered long-term.

Return JSON with this exact structure:
{
    "facts": [
        {"content": "fact statement", "category": "project|user_pref|tool|general", "importance": "critical|important|useful|trivial", "tags": "comma,separated,tags"}
    ],
    "contradicts": []
}

Rules:
- Only extract facts, not conversational filler
- Use present tense
- Be specific — include names, versions, decisions
- Default category is "general"
"""


def _extractor_get_provider_config() -> tuple:
    """Detect which LLM provider is available for extraction.

    Priority: MINIMAX_API_KEY > OPENROUTER_API_KEY > ...

    Returns (provider_name, api_key, base_url, model_name).
    """
    if os.environ.get("MINIMAX_API_KEY"):
        return (
            "minimax",
            os.environ["MINIMAX_API_KEY"],
            "https://api.minimax.io",
            "MiniMax-M2.7",
        )
    elif os.environ.get("OPENROUTER_API_KEY"):
        return (
            "openrouter",
            os.environ["OPENROUTER_API_KEY"],
            "https://openrouter.ai/api/v1",
            "openrouter/auto",
        )
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "anthropic",
            os.environ["ANTHROPIC_API_KEY"],
            "https://api.anthropic.com",
            "claude-sonnet-4-20250514",
        )
    elif os.environ.get("OPENAI_API_KEY"):
        return (
            "openai",
            os.environ["OPENAI_API_KEY"],
            "https://api.openai.com/v1",
            "gpt-4o",
        )
    # Fallback
    return ("", "", "", "")


class EtchMemoryProvider:
    """Hermes-compatible memory provider using Memory Etch as the backend.

    Routes the Hermes memory lifecycle (buffer -> extract -> consolidate -> retrieve)
    through EtchStore + optional LLM extraction.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the Hermes-compatible memory provider.

        Args:
            config: Optional dict overriding default extraction settings.
                Supported keys: ``auto_extract_llm``, ``extract_interval``,
                ``extract_min_meaningful``, ``extract_min_buffer``,
                ``extract_max_batch``, ``db_path``.
        """
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self._store: Optional[EtchStore] = None
        self._session_id: str = ""
        self._extractor_enabled: bool = False
        self._paused_until: float = 0.0
        self._consecutive_failures: int = 0
        self._total_extractions: int = 0
        self._lock = threading.Lock()

    def initialize(self, session_id: str) -> None:
        """Initialize the provider with a session.

        Creates or opens the EtchStore and prepares the extractor.

        Args:
            session_id: Unique session identifier for this provider instance.

        Raises:
            sqlite3.Error: If the underlying store cannot be initialized.
        """
        self._session_id = session_id
        db_path = self.config.get("db_path", f"memory_etch_{session_id}.db")
        self._store = EtchStore(db_path=db_path)
        self._extractor_enabled = self.config.get("auto_extract_llm", False)

        logger.info(
            "EtchMemoryProvider initialized | session=%s | extractor=%s",
            session_id, self._extractor_enabled,
        )

    def shutdown(self) -> None:
        """Close the provider and flush any pending data.

        Closes the underlying EtchStore and releases database resources.
        Safe to call multiple times.
        """
        if self._store:
            self._store.close()
            self._store = None
        logger.info("EtchMemoryProvider shut down")

    # ------------------------------------------------------------------
    # Turn buffer
    # ------------------------------------------------------------------

    def _buffer_turn(self, session_id: str, role: str, content: str) -> None:
        """Buffer a conversation turn for later extraction."""
        if not self._store:
            return
        meaningful = 1 if len(content.strip()) >= _MEANINGFUL_MIN_LENGTH else 0
        with self._store._lock:
            self._store._conn.execute(
                """INSERT INTO turn_buffer (session_id, role, content, meaningful)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, meaningful),
            )
            self._store._conn.commit()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_from_buffer(self, session_id: str) -> None:
        """Run extraction on buffered turns."""
        if not self._store or not self._extractor_enabled:
            return
        if self._circuit_breaker_active():
            logger.warning("Circuit breaker active — skipping extraction")
            return

        # Read buffered turns
        with self._store._lock:
            rows = self._store._conn.execute(
                "SELECT turn_id, role, content FROM turn_buffer "
                "WHERE session_id = ? ORDER BY turn_id LIMIT ?",
                (session_id, self.config.get("extract_max_batch", 20)),
            ).fetchall()
            if not rows:
                return
            turn_ids = [r["turn_id"] for r in rows]

        # Build prompt from turns
        conversation_text = "\n".join(
            f"{r['role']}: {r['content']}" for r in rows
        )
        user_prompt = f"Extract facts from this conversation:\n\n{conversation_text}"

        try:
            t0 = time.time()
            raw = self._call_llm_extract(_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            elapsed = int((time.time() - t0) * 1000)

            parsed = self._parse_llm_response(raw)
            if parsed is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._paused_until = time.time() + 300
                return

            facts_added = self._store_facts_from_llm(parsed, session_id)
            self._consecutive_failures = 0
            self._total_extractions += 1

            # Log extraction
            facts_list = parsed.get("facts", [])
            with self._store._lock:
                self._store._conn.execute(
                    """INSERT INTO extractions
                       (session_id, facts_found, facts_extracted, facts_added, duration_ms)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, len(facts_list), len(facts_list),
                     facts_added, elapsed),
                )
                # Clear buffered turns
                placeholders = ",".join("?" * len(turn_ids))
                self._store._conn.execute(
                    f"DELETE FROM turn_buffer WHERE turn_id IN ({placeholders})",
                    turn_ids,
                )
                self._store._conn.commit()

        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            self._consecutive_failures += 1
            with self._store._lock:
                self._store._conn.execute(
                    """INSERT INTO failed_buffers (session_id, turn_count, error)
                       VALUES (?, ?, ?)""",
                    (session_id, len(turn_ids), str(exc)[:500]),
                )
                self._store._conn.commit()
                if self._consecutive_failures >= 3:
                    self._paused_until = time.time() + 300

    def _call_llm_extract(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM for extraction. This is mocked in E2E tests.

        In production this would call the configured API.
        """
        raise NotImplementedError(
            "_call_llm_extract must be patched or overridden in production"
        )

    def _parse_llm_response(self, raw: str) -> Optional[dict]:
        """Parse LLM response to extract JSON fact list.

        Handles code fence blocks and plain JSON.
        """
        text = raw.strip()
        # Try to extract from code fence
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "facts" in data:
                return data
            return None
        except (json.JSONDecodeError, ValueError):
            return None

    def _store_facts_from_llm(self, parsed: dict, session_id: str) -> int:
        """Store extracted facts into EtchStore, handling dedup/reinforce."""
        if not self._store:
            return 0
        added = 0
        for fact_data in parsed.get("facts", []):
            content = fact_data.get("content", "").strip()
            if not content:
                continue
            category = fact_data.get("category", "general")
            importance_str = fact_data.get("importance", "useful")
            importance_map = {
                "critical": 1.0,
                "important": 0.75,
                "useful": 0.5,
                "trivial": 0.2,
            }
            importance = importance_map.get(importance_str, 0.5)
            tags = fact_data.get("tags", "")

            # Dedup: check if similar content exists
            existing = self._find_similar_fact(content, session_id)
            if existing:
                # Reinforce: increment reinforcement_count
                with self._store._lock:
                    self._store._conn.execute(
                        "UPDATE facts SET reinforcement_count = COALESCE(reinforcement_count, 0) + 1, "
                        "trust_score = MIN(1.0, trust_score + 0.05), "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE fact_id = ?",
                        (existing["fact_id"],),
                    )
                    self._store._conn.commit()
            else:
                self._store.add_fact(
                    content=content,
                    category=category,
                    tags=tags,
                    importance=importance,
                    session_id=session_id,
                )
                added += 1
        return added

    def _find_similar_fact(self, content: str, session_id: str) -> Optional[dict]:
        """Find an existing fact with the same content (exact match dedup)."""
        if not self._store:
            return None
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT fact_id, content FROM facts "
                "WHERE content = ? AND (deleted IS NULL OR deleted = 0) LIMIT 1",
                (content,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: dict) -> str:
        """Handle Hermes tool dispatch for fact_store operations.

        Supported actions: ``add``, ``search``, ``feedback``.

        Args:
            tool_name: Must be ``"fact_store"``.
            args: Tool arguments dict with ``action`` and action-specific keys.

        Returns:
            JSON string with the action result.

        Raises:
            ValueError: If required arguments are missing.
        """
        if tool_name != "fact_store":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        action = args.get("action", "")

        if action == "add":
            if not self._store:
                return json.dumps({"error": "Provider not initialized"})
            fid = self._store.add_fact(
                content=args.get("content", ""),
                category=args.get("category", "general"),
                tags=args.get("tags", ""),
                trust_score=args.get("trust_score"),
                importance=args.get("importance"),
                project=args.get("project", ""),
                session_id=args.get("session_id", self._session_id),
                topic_key=args.get("topic_key", ""),
            )
            return json.dumps({"fact_id": fid, "action": "added"})

        elif action == "search":
            if not self._store:
                return json.dumps({"error": "Provider not initialized", "count": 0, "results": []})
            query = args.get("query", "")
            if not query:
                return json.dumps({"count": 0, "results": []})
            from .retrieval import EtchRetriever
            retriever = EtchRetriever(self._store)
            results = retriever.search(
                query,
                limit=args.get("limit", 10),
                project=args.get("project", ""),
            )
            return json.dumps({"count": len(results), "results": results}, default=str)

        elif action == "feedback":
            if not self._store:
                return json.dumps({"error": "Provider not initialized"})
            fid = args.get("fact_id")
            helpful = args.get("helpful", False)
            if fid is not None:
                with self._store._lock:
                    self._store._conn.execute(
                        "UPDATE facts SET helpful_count = COALESCE(helpful_count, 0) + 1, "
                        "trust_score = MIN(1.0, trust_score + ?) WHERE fact_id = ?",
                        (0.1 if helpful else -0.1, fid),
                    )
                    self._store._conn.commit()
            return json.dumps({"fact_id": fid, "feedback": "recorded"})

        return json.dumps({"error": f"Unknown action: {action}"})

    def _handle_extractor_status(self) -> str:
        """Return current extraction status as JSON."""
        buffer_turns = 0
        if self._store:
            row = self._store._conn.execute(
                "SELECT COUNT(*) as c FROM turn_buffer"
            ).fetchone()
            buffer_turns = row["c"] if row else 0

        return json.dumps({
            "enabled": self._extractor_enabled,
            "buffer_turns": buffer_turns,
            "total_extractions": self._total_extractions,
            "consecutive_failures": self._consecutive_failures,
            "paused": self._circuit_breaker_active(),
            "paused_until": self._paused_until,
            "config": {
                "interval": self.config.get("extract_interval", 5),
                "min_buffer": self.config.get("extract_min_buffer", 5),
                "max_batch": self.config.get("extract_max_batch", 20),
            },
        })

    def _circuit_breaker_active(self) -> bool:
        """Check if the circuit breaker is open (paused)."""
        return time.time() < self._paused_until
