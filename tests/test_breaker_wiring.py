"""Tests for circuit breaker wiring in curator and classifier."""

import pytest

from memory_etch.circuit_breaker import LLMCircuitBreaker


# Reset breakers between tests to avoid cross-test contamination
@pytest.fixture(autouse=True)
def _reset_breakers():
    import memory_etch.classifier as cls_mod  # noqa: I001
    import memory_etch.curator as cur_mod  # noqa: I001
    cur_mod._breaker.record_success()
    cls_mod._breaker.record_success()
    yield
    cur_mod._breaker.record_success()
    cls_mod._breaker.record_success()


class TestCuratorBreaker:
    """Curator module has a shared circuit breaker."""

    def test_curator_has_module_level_breaker(self):
        import memory_etch.curator as cur_mod
        assert hasattr(cur_mod, "_breaker")
        assert isinstance(cur_mod._breaker, LLMCircuitBreaker)

    def test_curator_breaker_in_cooldown_skips_curate(self, store):
        """When breaker is in cooldown, curate() returns early with skipped stats."""
        import memory_etch.curator as cur_mod
        # Simulate cooldown by setting failures to max
        cur_mod._breaker.record_failure()
        cur_mod._breaker.record_failure()
        cur_mod._breaker.record_failure()
        assert cur_mod._breaker.is_available() is False

        from memory_etch.curator import EtchCurator
        curator = EtchCurator(store)
        stats = curator.curate()
        # Should return early result without running operations
        assert "skipped" in stats
        assert stats["skipped"] is True

    def test_curator_breaker_reset_after_cooldown(self, store):
        """After resetting the breaker, curate() runs normally."""
        import memory_etch.curator as cur_mod
        # Put in cooldown, then reset
        cur_mod._breaker.record_failure()
        cur_mod._breaker.record_failure()
        cur_mod._breaker.record_failure()
        cur_mod._breaker.record_success()
        assert cur_mod._breaker.is_available() is True

        from memory_etch.curator import EtchCurator
        curator = EtchCurator(store)
        stats = curator.curate()
        assert "decayed" in stats
        assert "duration_ms" in stats


class TestClassifierBreaker:
    """Classifier module has a shared circuit breaker."""

    def test_classifier_has_module_level_breaker(self):
        import memory_etch.classifier as cls_mod
        assert hasattr(cls_mod, "_breaker")
        assert isinstance(cls_mod._breaker, LLMCircuitBreaker)

    def test_classifier_breaker_does_not_block_rule_based(self):
        """Classifier's rule-based classify() still works when breaker is open."""
        import memory_etch.classifier as cls_mod
        # Put breaker in cooldown
        cls_mod._breaker.record_failure()
        cls_mod._breaker.record_failure()
        cls_mod._breaker.record_failure()
        assert cls_mod._breaker.is_available() is False

        from memory_etch.classifier import QueryClassifier
        classifier = QueryClassifier()
        result = classifier.classify("tell me about Python")
        assert result["intent"] == "entity"

    def test_classifier_breaker_available_by_default(self):
        import memory_etch.classifier as cls_mod
        assert cls_mod._breaker.is_available() is True
        assert cls_mod._breaker.failures == 0


@pytest.fixture
def store(tmp_path):
    """In-memory store for curator tests."""
    import os

    from memory_etch.store import EtchStore
    db_path = os.path.join(tmp_path, "test_breaker.db")
    s = EtchStore(str(db_path), auto_migrate=True)
    yield s
    s.close()
