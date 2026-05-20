"""Memory benchmark runner and provider interface.

Implements the ``MemoryProvider`` ABC that any memory system can implement,
and ``BenchmarkRunner`` that evaluates recall@k using a synthetic dataset
and a Gemini judge.
"""

import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .dataset import SyntheticDataset, Document
from .judge import GeminiJudge


class MemoryProvider(ABC):
    """Abstract base for memory systems in the benchmark.

    Implement this ABC to benchmark ANY memory backend against the
    same synthetic dataset and judge.

    Example::

        class MyMemoryProvider(MemoryProvider):
            name = "my-memory"
            description = "My custom memory system"

            def prepare(self, store_dir, reset=False):
                ...

            def ingest(self, documents):
                ...

            def retrieve(self, query, k=10, user_id=None):
                ...

            def cleanup(self):
                ...
    """

    name: str = ""
    description: str = ""
    kind: str = ""

    def prepare(self, store_dir: str, reset: bool = False) -> None:
        """Initialize the memory store."""

    @abstractmethod
    def ingest(self, documents: list[Document]) -> None:
        """Ingest documents into memory."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10,
                 user_id: Optional[str] = None,
                 ) -> tuple[list[Document], Optional[dict]]:
        """Retrieve top-k documents relevant to the query."""

    def cleanup(self) -> None:
        """Release resources after the benchmark."""


class BenchmarkRunner:
    """Run a memory benchmark against any MemoryProvider.

    Args:
        provider: A MemoryProvider instance.
        judge: A GeminiJudge instance. If None, uses defaults.
        n_docs: Number of synthetic documents (default: 100).
        seed: Random seed (default: 42).

    Example::

        from memory_etch.benchmark import BenchmarkRunner, SyntheticDataset

        provider = EtchMemoryProvider()
        runner = BenchmarkRunner(provider, n_docs=100)
        results = runner.run()
        print(results["accuracy"])
    """

    def __init__(
        self,
        provider: MemoryProvider,
        judge: Optional[GeminiJudge] = None,
        n_docs: int = 100,
        seed: int = 42,
    ):
        self._provider = provider
        self._judge = judge or GeminiJudge()
        self._dataset = SyntheticDataset(seed=seed, n_docs=n_docs)

    def run(self, query_limit: Optional[int] = None, verbose: bool = False) -> dict:
        """Run the full benchmark cycle.

        Args:
            query_limit: Optional limit on number of queries.
            verbose: Print per-query results.

        Returns:
            dict with keys: accuracy, total_queries, correct, avg_retrieve_time_ms,
            avg_context_tokens, results, etc.
        """
        documents, queries = self._dataset.generate()

        if query_limit:
            queries = queries[:query_limit]

        # Prepare provider
        tmpdir = tempfile.mkdtemp()
        store_dir = os.path.join(tmpdir, "store")
        os.makedirs(store_dir, exist_ok=True)

        try:
            self._provider.prepare(store_dir, reset=True)
            self._provider.ingest(documents)

            results = []
            correct = 0

            for i, q in enumerate(queries, 1):
                t0 = time.time()
                retrieved_docs, meta = self._provider.retrieve(
                    query=q["query"],
                    k=15,
                    user_id=q.get("user_id"),
                )
                retrieve_time = (time.time() - t0) * 1000

                context = "\n---\n".join(d.content for d in retrieved_docs)
                context_tokens = len(context.split())

                gold = q["gold_answers"][0] if q["gold_answers"] else ""
                verdict, token_meta = self._judge.judge(q["query"], context, gold) if gold else (False, {})
                is_correct = bool(verdict) if verdict is not None else False

                result = {
                    "query_id": q["id"],
                    "query": q["query"],
                    "gold_answers": q["gold_answers"],
                    "retrieved_ids": [d.id for d in retrieved_docs],
                    "retrieve_time_ms": round(retrieve_time, 1),
                    "context_tokens": context_tokens,
                    "judge_prompt_tokens": token_meta.get("prompt_tokens", 0),
                    "judge_output_tokens": token_meta.get("output_tokens", 0),
                    "correct": is_correct,
                    "judged": verdict is not None,
                }
                results.append(result)

                if is_correct:
                    correct += 1

                if verbose:
                    status = "[OK]" if is_correct else "[NO]"
                    print(f"  [{i}/{len(queries)}] {status} {q['query'][:70]}... ({retrieve_time:.1f}ms)")

            # Summary
            accuracy = correct / len(queries) if queries else 0
            avg_time = sum(r["retrieve_time_ms"] for r in results) / len(results) if results else 0
            avg_tokens = sum(r["context_tokens"] for r in results) / len(results) if results else 0

            summary = {
                "provider": self._provider.name,
                "total_documents": len(documents),
                "total_queries": len(queries),
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "avg_retrieve_time_ms": round(avg_time, 1),
                "avg_context_tokens": round(avg_tokens, 1),
                "results": results,
            }

            if verbose:
                self._print_summary(summary)

            return summary

        finally:
            self._provider.cleanup()
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _print_summary(summary: dict) -> None:
        print(f"\n{'='*60}")
        print(f"RESULTS: {summary['provider']}")
        print(f"{'='*60}")
        print(f"  Total queries:     {summary['total_queries']}")
        print(f"  Correct:           {summary['correct']}/{summary['total_queries']}")
        print(f"  Accuracy:          {summary['accuracy']:.1%}")
        print(f"  Avg retrieve:      {summary['avg_retrieve_time_ms']:.1f}ms")
        print(f"  Avg ctx tokens:    {summary['avg_context_tokens']:.0f}")
        print(f"{'='*60}\n")
