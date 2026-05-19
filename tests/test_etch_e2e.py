"""End-to-end integration tests for the Memory Etch system.

Tests the full pipeline: provider init → buffer turns → LLM extraction →
dedup/reinforce → fact storage → retrieval — using mocked LLM calls
but real store/retrieval layer.

These are NOT unit tests — they test how the pieces fit together.
"""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import sys
import os

# Mock Hermes modules before importing EtchMemoryProvider
import types
for m in ["agent", "agent.memory_provider", "agent.memory_manager",
          "agent.portal_tags", "tools", "tools.registry",
          "hermes_cli", "hermes_cli.config", "hermes_constants",
          "hermes_state"]:
    sys.modules[m] = types.ModuleType(m)


class MockMemoryProvider:
    @property
    def name(self): return "mock"
    def is_available(self): return True
    def initialize(self, *a, **kw): pass
    def get_tool_schemas(self): return []
    def handle_tool_call(self, *a): return ""
    def system_prompt_block(self): return ""
    def prefetch(self, *a, **kw): return ""
    def sync_turn(self, *a, **kw): pass
    def shutdown(self): pass


sys.modules["agent.memory_provider"].MemoryProvider = MockMemoryProvider
sys.modules["agent.memory_manager"].sanitize_context = lambda x, **kw: x
sys.modules["tools.registry"].tool_error = lambda m: json.dumps({"error": m})
sys.modules["hermes_cli.config"].cfg_get = lambda d, *k, **kw: d.get(k[0], {}) if k else {}
sys.modules["hermes_constants"].get_hermes_home = lambda: Path(tempfile.mkdtemp())
sys.modules["hermes_constants"].display_hermes_home = lambda: str(Path(tempfile.mkdtemp()))
sys.modules["hermes_state"].apply_wal_with_fallback = lambda conn, *a, **kw: conn.execute("PRAGMA journal_mode=WAL")

from memory_etch.__init__ import EtchMemoryProvider, _extractor_get_provider_config


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Return a temporary db path."""
    return Path(tempfile.mkdtemp()) / "test_e2e.db"


@pytest.fixture
def provider(tmp_db):
    """Create a EtchMemoryProvider with a fresh temp DB and MiniMax key."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-cp-test"}):
        p = EtchMemoryProvider({
            "auto_extract_llm": True,
            "extract_interval": 3,
            "extract_min_meaningful": 1,
            "extract_min_buffer": 2,
            "extract_max_batch": 10,
            "db_path": str(tmp_db),
        })
        p.initialize("test-e2e")
    yield p
    p.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderDetection:
    """Test _extractor_get_provider_config under various env setups."""

    def test_detects_minimax(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-cp-test"}, clear=True):
            p, k, b, m = _extractor_get_provider_config()
        assert p == "minimax"
        assert "MiniMax-M2.7" in m
        assert "api.minimax.io" in b

    def test_minimax_priority_over_openrouter(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mx", "OPENROUTER_API_KEY": "or"}, clear=True):
            p2, _, _, _ = _extractor_get_provider_config()
        assert p2 == "minimax"


class TestProviderInit:
    """Test provider initialization and schema correctness."""

    def test_provider_created(self, provider):
        assert provider is not None
        assert provider._extractor_enabled

    def test_all_tables_exist(self, provider):
        conn = provider._store._conn
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for tbl in ("turn_buffer", "facts", "extractions", "failed_buffers"):
            assert tbl in tables, f"Missing table: {tbl}"

    def test_facts_columns(self, provider):
        conn = provider._store._conn
        cols = {r[1] for r in conn.execute("PRAGMA table_info(facts)")}
        for col in ("importance", "reinforcement_count", "consolidated",
                    "trust_score", "tags", "category", "content"):
            assert col in cols, f"Missing column: {col}"


class TestBufferTurns:
    """Test the turn buffer — conversations are captured correctly."""

    CONVERSATION = [
        ("user", "Hola, cómo estás?"),
        ("assistant", "Bien, gracias! En qué te ayudo?"),
        ("user", "Necesito migrar de SQLite a PostgreSQL por JSONB"),
        ("assistant", "Buena decisión. Te recomiendo asyncpg."),
        ("user", "También Redis para caché de sesiones con TTL 1h"),
        ("assistant", "Perfecto, redis-py con connection pool."),
        ("user", "Y FastAPI en vez de Flask para la API"),
        ("assistant", "FastAPI es superior: async nativo, OpenAPI, Pydantic v2."),
    ]

    def test_buffers_8_turns(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        cnt = provider._store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE session_id='test-e2e'"
        ).fetchone()[0]
        assert cnt == 8

    def test_all_turns_meaningful(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        meaningful = provider._store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE session_id='test-e2e' AND meaningful=1"
        ).fetchone()[0]
        assert meaningful == 8


class TestExtractionPipeline:
    """Test full extraction: buffer → LLM → parse → store → clear."""

    LLM_RESPONSE = json.dumps({
        "facts": [
            {"content": "Database migrated from SQLite to PostgreSQL for JSONB support and concurrent performance",
             "category": "project", "importance": "important", "tags": "postgresql,jsonb"},
            {"content": "Redis used as session cache with 1 hour TTL via redis-py with connection pool",
             "category": "project", "importance": "important", "tags": "redis,cache"},
            {"content": "API framework switched from Flask to FastAPI for async support and Pydantic v2",
             "category": "project", "importance": "important", "tags": "fastapi,api"},
            {"content": "User prefers asyncpg for PostgreSQL async connections",
             "category": "tool", "importance": "useful", "tags": "asyncpg,postgresql"},
        ],
        "contradicts": []
    })

    CONVERSATION = [
        ("user", "Necesito migrar de SQLite a PostgreSQL por JSONB"),
        ("assistant", "Buena decisión. Te recomiendo asyncpg."),
        ("user", "Redis para caché de sesiones con TTL 1h"),
        ("assistant", "Perfecto, redis-py con connection pool."),
        ("user", "FastAPI en vez de Flask"),
        ("assistant", "FastAPI es superior."),
    ]

    def test_extracts_and_stores_4_facts(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_RESPONSE):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        facts = provider._store._conn.execute(
            "SELECT fact_id, content, category FROM facts ORDER BY fact_id"
        ).fetchall()
        assert len(facts) == 4

    def test_buffer_cleared_after_extraction(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_RESPONSE):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        left = provider._store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE session_id='test-e2e'"
        ).fetchone()[0]
        assert left == 0

    def test_extractions_logged(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_RESPONSE):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        extractions = provider._store._conn.execute(
            "SELECT facts_extracted FROM extractions"
        ).fetchall()
        assert len(extractions) >= 1
        if extractions:
            assert extractions[0][0] == 4

    def test_all_specific_facts_present(self, provider):
        for role, text in self.CONVERSATION:
            provider._buffer_turn("test-e2e", role, text)
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_RESPONSE):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        facts = provider._store._conn.execute(
            "SELECT content, category FROM facts"
        ).fetchall()
        contents = [f[0] for f in facts]
        categories = [f[1] for f in facts]
        assert any("PostgreSQL" in c for c in contents)
        assert any("Redis" in c for c in contents)
        assert any("FastAPI" in c for c in contents)
        assert "project" in categories
        assert "tool" in categories


class TestDedupAndReinforce:
    """Test that near-duplicate facts are deduped and reinforced."""

    LLM_ORIGINAL = json.dumps({
        "facts": [
            {"content": "Database migrated from SQLite to PostgreSQL for JSONB support and concurrent performance",
             "category": "project", "importance": "important", "tags": "postgresql,jsonb"},
        ],
        "contradicts": []
    })

    LLM_DUP = json.dumps({
        "facts": [
            {"content": "Database migrated from SQLite to PostgreSQL for JSONB support and concurrent performance",
             "category": "project", "importance": "critical", "tags": "postgresql"},
            {"content": "New decision: deploy via Docker Compose with health checks",
             "category": "project", "importance": "important", "tags": "docker,deploy"},
        ],
        "contradicts": []
    })

    def test_duplicate_gets_reinforced_not_duplicated(self, provider):
        # First extraction
        provider._buffer_turn("test-e2e", "user", "migrate to PostgreSQL")
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_ORIGINAL):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        ref_before = provider._store._conn.execute(
            "SELECT reinforcement_count FROM facts WHERE content LIKE '%PostgreSQL%'"
        ).fetchone()
        assert ref_before is not None
        before_count = ref_before[0]

        # Second extraction with duplicate
        provider._buffer_turn("test-e2e", "user", "Docker deploy")
        provider._buffer_turn("test-e2e", "assistant", "OK")
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_DUP):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)

        facts = provider._store._conn.execute(
            "SELECT fact_id, content FROM facts ORDER BY fact_id"
        ).fetchall()
        ref_after = provider._store._conn.execute(
            "SELECT reinforcement_count FROM facts WHERE content LIKE '%PostgreSQL%'"
        ).fetchone()

        assert len(facts) == 2  # original 1 + new docker = 2 (not 3)
        assert ref_after is not None
        assert ref_after[0] >= before_count + 1
        assert any("Docker" in f[1] for f in facts)

    def test_buffer_cleared_after_dedup(self, provider):
        provider._buffer_turn("test-e2e-dup", "user", "test")
        with patch.object(provider, '_call_llm_extract', return_value=self.LLM_ORIGINAL):
            provider._extract_from_buffer("test-e2e-dup")
        time.sleep(0.3)
        left = provider._store._conn.execute(
            "SELECT COUNT(*) FROM turn_buffer WHERE session_id='test-e2e-dup'"
        ).fetchone()[0]
        assert left == 0


class TestExtractorStatus:
    """Test the extractor_status endpoint."""

    def test_status_after_extraction(self, provider):
        provider._buffer_turn("test-e2e", "user", "hello")
        with patch.object(provider, '_call_llm_extract',
                          return_value=json.dumps({"facts": [], "contradicts": []})):
            provider._extract_from_buffer("test-e2e")
        time.sleep(0.3)
        status = json.loads(provider._handle_extractor_status())
        assert status["enabled"] is True
        assert "buffer_turns" in status
        assert "total_extractions" in status
        assert status["consecutive_failures"] == 0
        assert status.get("paused") is False
        assert "config" in status
        assert status["config"]["interval"] == 3


class TestCircuitBreaker:
    """Test the circuit breaker / pause mechanism."""

    def test_normal_closed(self, provider):
        assert not provider._circuit_breaker_active()

    def test_paused_open(self, provider):
        import time
        provider._paused_until = time.time() + 3600
        assert provider._circuit_breaker_active()

    def test_recovered_closed(self, provider):
        import time
        provider._paused_until = time.time() - 1
        assert not provider._circuit_breaker_active()


class TestParsingAndPrivacy:
    """Test LLM response parsing."""

    def test_code_fence_parsed(self, provider):
        fence = "Here:\n\n```json\n{\"facts\": [{\"content\": \"Test fact\", \"category\": \"project\", \"importance\": \"useful\", \"tags\": \"test\"}], \"contradicts\": []}\n```\n\nDone."
        parsed = provider._parse_llm_response(fence)
        assert parsed is not None
        assert len(parsed["facts"]) == 1


class TestStoreThroughToolDispatch:
    """Test that fact_store tool dispatch works end-to-end."""

    def test_add_and_search_fact_via_tool(self, provider):
        add_result = provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "E2E test fact — tool dispatch works",
            "category": "tool",
            "tags": "e2e,test",
        })
        data = json.loads(add_result)
        assert "fact_id" in data

        search_result = provider.handle_tool_call("fact_store", {
            "action": "search",
            "query": "tool dispatch",
        })
        data2 = json.loads(search_result)
        assert data2["count"] >= 1
        assert any("tool dispatch" in f["content"] for f in data2["results"])

    def test_feedback_updates_trust(self, provider):
        add_result = provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "Fact for trust testing E2E",
            "category": "general",
        })
        fid = json.loads(add_result)["fact_id"]

        provider.handle_tool_call("fact_store", {
            "action": "feedback",
            "fact_id": fid,
            "helpful": True,
        })

        search_data = json.loads(provider.handle_tool_call("fact_store", {
            "action": "search",
            "query": "trust testing",
        }))
        matched = [f for f in search_data.get("results", []) if f["fact_id"] == fid]
        assert len(matched) == 1
        assert matched[0]["trust_score"] > 0.0
