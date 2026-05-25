"""Tests for LLMCircuitBreaker — in-memory failure tracking and cooldown."""

import time

from memento.circuit_breaker import LLMCircuitBreaker


class TestCircuitBreakerInitialState:
    """Initial state — no failures, available."""

    def test_default_construction(self):
        b = LLMCircuitBreaker()
        assert b.is_available() is True
        assert b.failures == 0
        assert b.in_cooldown is False

    def test_custom_max_and_cooldown(self):
        b = LLMCircuitBreaker(max_failures=5, cooldown_seconds=120)
        assert b._max == 5
        assert b._cooldown == 120
        assert b.is_available() is True


class TestCircuitBreakerFailureAccumulation:
    """Failures accumulate until max_failures is reached."""

    def test_single_failure_still_available(self):
        b = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
        b.record_failure()
        assert b.failures == 1
        assert b.is_available() is True

    def test_two_failures_still_available(self):
        b = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
        b.record_failure()
        b.record_failure()
        assert b.failures == 2
        assert b.is_available() is True

    def test_max_failures_triggers_cooldown(self):
        b = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
        b.record_failure()
        b.record_failure()
        b.record_failure()
        assert b.failures == 3
        assert b.is_available() is False
        assert b.in_cooldown is True


class TestCircuitBreakerCooldown:
    """Cooldown behavior — time-based recovery."""

    def test_cooldown_expires_and_recovers(self):
        b = LLMCircuitBreaker(max_failures=2, cooldown_seconds=0.1)
        b.record_failure()
        b.record_failure()
        assert b.is_available() is False
        time.sleep(0.15)
        assert b.is_available() is True
        assert b.failures == 0

    def test_cooldown_not_expired_yet(self):
        b = LLMCircuitBreaker(max_failures=2, cooldown_seconds=5)
        b.record_failure()
        b.record_failure()
        assert b.is_available() is False
        # Not enough time has passed
        assert b.is_available() is False

    def test_cooldown_zero_seconds_immediate_recovery(self):
        """With cooldown_seconds=0, the breaker recovers immediately on next check."""
        b = LLMCircuitBreaker(max_failures=1, cooldown_seconds=0)
        b.record_failure()
        # cooldown_until = monotonic() + 0, so the next check will see it's expired
        assert b.is_available() is True
        assert b.failures == 0


class TestCircuitBreakerReset:
    """record_success resets everything."""

    def test_record_success_resets_failures(self):
        b = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
        b.record_failure()
        b.record_failure()
        assert b.failures == 2
        b.record_success()
        assert b.failures == 0
        assert b.is_available() is True
        assert b.in_cooldown is False

    def test_record_success_during_cooldown_brings_back(self):
        b = LLMCircuitBreaker(max_failures=2, cooldown_seconds=60)
        b.record_failure()
        b.record_failure()
        assert b.is_available() is False
        b.record_success()
        assert b.is_available() is True
        assert b.failures == 0


class TestCircuitBreakerEdgeCases:
    """Boundary and edge cases."""

    def test_no_failures_never_in_cooldown(self):
        b = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
        assert b.in_cooldown is False
        for _ in range(100):
            assert b.is_available() is True

    def test_failure_after_recovery_starts_fresh(self):
        b = LLMCircuitBreaker(max_failures=2, cooldown_seconds=0.1)
        b.record_failure()
        b.record_failure()
        assert b.is_available() is False
        time.sleep(0.15)
        assert b.is_available() is True
        b.record_failure()
        assert b.failures == 1
        assert b.is_available() is True

    def test_max_failures_zero(self):
        """max_failures=0 means every failure triggers cooldown."""
        b = LLMCircuitBreaker(max_failures=1, cooldown_seconds=60)
        b.record_failure()
        assert b.is_available() is False

    def test_max_failures_high(self):
        """High max_failures allows many failures before cooldown."""
        b = LLMCircuitBreaker(max_failures=1000, cooldown_seconds=60)
        for _ in range(500):
            b.record_failure()
        assert b.failures == 500
        assert b.is_available() is True
        # Still below max
        for _ in range(500):
            b.record_failure()
        assert b.failures == 1000
        assert b.is_available() is False
