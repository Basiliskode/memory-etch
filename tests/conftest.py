"""Shared test fixtures for Memory Etch."""

from unittest.mock import MagicMock, patch

import pytest
from memory_etch import EtchStore, EtchRetriever
from memory_etch.etch import EtchMemoryProvider


@pytest.fixture
def etch_store(tmp_path):
    """Create a temporary EtchStore for testing."""
    store = EtchStore(str(tmp_path / "test_memory.db"), auto_migrate=True)
    yield store
    store.close()


@pytest.fixture
def etch_store_with_facts(etch_store):
    """Pre-populated store with a few facts."""
    etch_store.add_fact("Python is a programming language", category="tech", tags="python")
    etch_store.add_fact("FastAPI is a web framework", category="tech", tags="python,web")
    etch_store.add_fact("SQLite is a database engine", category="tech", tags="sqlite,db")
    etch_store.add_fact("User prefers dark mode", category="user_pref", tags="ui,theme")
    etch_store.add_fact("Flask version is 3.1", category="tech", tags="python,framework")
    return etch_store


@pytest.fixture
def etch_retriever(etch_store_with_facts):
    """EtchRetriever connected to a pre-populated store."""
    return EtchRetriever(etch_store_with_facts)


@pytest.fixture
def mock_llm_extract():
    """Mock _call_llm_extract to return 'Extracted: {content}' for E2E tests.

    Patches the class-level method so any EtchMemoryProvider instance
    calling _call_llm_extract gets the mocked return value.
    """
    mock = MagicMock(return_value="Extracted: {content}")
    with patch.object(EtchMemoryProvider, '_call_llm_extract', mock):
        yield mock
