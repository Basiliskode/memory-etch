"""Benchmark suite for memory systems.

Provides a standard ``MemoryProvider`` ABC and a benchmark runner that
evaluates recall@k on a synthetic dataset using a Gemini judge.

Usage::

    # Benchmark memory-etch (default)
    python -m memory_etch.benchmark

    # With verbose output
    python -m memory_etch.benchmark --verbose

    # Custom document count
    python -m memory_etch.benchmark --n-docs 500

Implement the ``MemoryProvider`` ABC to benchmark ANY memory backend.
"""

from .runner import MemoryProvider, BenchmarkRunner
from .judge import GeminiJudge
from .dataset import SyntheticDataset

__all__ = ["MemoryProvider", "BenchmarkRunner", "GeminiJudge", "SyntheticDataset"]
