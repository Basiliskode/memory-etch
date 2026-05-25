"""CLI entry point for the memory benchmark.

Usage::

    # Default (etch provider)
    python -m memento.benchmark

    # Verbose
    python -m memento.benchmark --verbose

    # Custom configuration
    python -m memento.benchmark --n-docs 500 --seed 42

    # Full results JSON
    python -m memento.benchmark --output results.json

    # Custom provider (experimental)
    python -m memento.benchmark --provider etch

Requirements:
    - GEMINI_API_KEY environment variable set
    - memento installed (for the default etch provider)
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="memento benchmark: evaluate memory recall@k",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m memento.benchmark\n"
            "  python -m memento.benchmark --verbose\n"
            "  python -m memento.benchmark --n-docs 500 --output results.json\n"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-query results")
    parser.add_argument("--n-docs", type=int, default=100, help="Number of synthetic documents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--query-limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to path")
    parser.add_argument(
        "--provider", type=str, default="etch",
        choices=["etch", "json-baseline"],
        help="Memory provider to benchmark (default: etch)",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    # Import here so the core package doesn't need benchmark deps
    from .runner import BenchmarkRunner

    if args.provider == "etch":
        from .providers import EtchBenchmarkProvider
        provider = EtchBenchmarkProvider()
    elif args.provider == "json-baseline":
        from .providers import JsonMemoryProvider
        provider = JsonMemoryProvider()
    else:
        print(f"Unknown provider: {args.provider}")
        sys.exit(1)

    runner = BenchmarkRunner(
        provider=provider,
        n_docs=args.n_docs,
        seed=args.seed,
    )

    summary = runner.run(query_limit=args.query_limit, verbose=args.verbose)
    print(f"\nBenchmark complete: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total_queries']})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    main()
