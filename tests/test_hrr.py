"""Tests for HRR (Holographic Reduced Representations) module."""

import math
import pytest
from memento import hrr


class TestHRRCore:
    def test_has_numpy_flag(self):
        # Should be True since numpy is installed in test env
        assert hrr.HAS_NUMPY is not None

    def test_encode_atom_returns_array(self):
        v = hrr.encode_atom("test", dim=256)
        assert len(v) == 256
        assert all(0 <= x < 2 * math.pi for x in v)

    def test_encode_atom_deterministic(self):
        v1 = hrr.encode_atom("hello", dim=256)
        v2 = hrr.encode_atom("hello", dim=256)
        assert (v1 == v2).all()

    def test_encode_atom_different_words(self):
        v1 = hrr.encode_atom("hello", dim=256)
        v2 = hrr.encode_atom("world", dim=256)
        # Different words should produce different vectors
        assert not (v1 == v2).all()

    def test_encode_atom_different_dims(self):
        v256 = hrr.encode_atom("test", dim=256)
        v128 = hrr.encode_atom("test", dim=128)
        assert len(v256) == 256
        assert len(v128) == 128

    def test_bind_operation(self):
        a = hrr.encode_atom("concept_a", dim=256)
        b = hrr.encode_atom("concept_b", dim=256)
        bound = hrr.bind(a, b)
        assert len(bound) == 256
        # Bound vector should be dissimilar to both inputs
        sim_a = hrr.similarity(bound, a)
        sim_b = hrr.similarity(bound, b)
        assert sim_a < 0.3
        assert sim_b < 0.3

    def test_bind_unbind_roundtrip(self):
        a = hrr.encode_atom("key", dim=256)
        b = hrr.encode_atom("value", dim=256)
        bound = hrr.bind(a, b)
        retrieved = hrr.unbind(bound, a)
        sim = hrr.similarity(retrieved, b)
        assert sim > 0.5  # Recovery degrades with bundle noise

    def test_bundle_operation(self):
        a = hrr.encode_atom("item_a", dim=256)
        b = hrr.encode_atom("item_b", dim=256)
        bundled = hrr.bundle(a, b)
        # Bundled vector should be similar to both inputs
        sim_a = hrr.similarity(bundled, a)
        sim_b = hrr.similarity(bundled, b)
        assert sim_a > 0.3
        assert sim_b > 0.3

    def test_similarity_identical(self):
        v = hrr.encode_atom("same", dim=256)
        sim = hrr.similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_similarity_unrelated(self):
        v1 = hrr.encode_atom("alpha", dim=256)
        v2 = hrr.encode_atom("beta", dim=256)
        sim = hrr.similarity(v1, v2)
        assert -0.3 < sim < 0.3


class TestHRRSerialization:
    def test_roundtrip(self):
        v = hrr.encode_atom("serialize_test", dim=256)
        blob = hrr.phases_to_bytes(v)
        v2 = hrr.bytes_to_phases(blob)
        assert (v == v2).all()
        assert len(blob) == 256 * 8  # float64

    def test_bytes_to_phases_mutable(self):
        v = hrr.encode_atom("mutable", dim=256)
        blob = hrr.phases_to_bytes(v)
        v2 = hrr.bytes_to_phases(blob)
        v2[0] = 0  # Should not raise
        assert True


class TestHRRText:
    def test_encode_text_basic(self):
        v = hrr.encode_text("hello world", dim=256)
        assert len(v) == 256

    def test_encode_text_empty(self):
        v = hrr.encode_text("", dim=256)
        assert len(v) == 256  # Returns __hrr_empty__ vector

    def test_encode_text_similar(self):
        v1 = hrr.encode_text("python programming", dim=256)
        v2 = hrr.encode_text("programming python", dim=256)
        sim = hrr.similarity(v1, v2)
        assert sim > 0.3  # Same bag of words

    def test_encode_text_different(self):
        v1 = hrr.encode_text("python programming", dim=256)
        v2 = hrr.encode_text("quantum physics", dim=256)
        sim = hrr.similarity(v1, v2)
        assert sim < 0.5  # Different topics


class TestHRREstimation:
    def test_snr_high_with_few_items(self):
        snr = hrr.snr_estimate(256, 1)
        assert snr > 10

    def test_snr_low_with_many_items(self):
        snr = hrr.snr_estimate(256, 1000)
        assert snr < 2.0

    def test_snr_inf_with_zero(self):
        snr = hrr.snr_estimate(256, 0)
        assert snr == float("inf")


class TestHRREncoding:
    def test_encode_fact_structure(self):
        v = hrr.encode_fact("Python is fast", ["Python"], dim=256)
        assert len(v) == 256
