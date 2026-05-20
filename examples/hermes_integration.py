"""
Hermes Agent Memory Provider -- Quick Start.

Requires: pip install memory-etch[hrr]

This example demonstrates the full lifecycle:
  1. Initialize an EtchMemoryProvider with a temp database
  2. Add facts via handle_tool_call (tool dispatch API used by Hermes)
  3. Search facts
  4. Provide feedback on a fact
  5. Shut down and clean up

Run:
    python examples/hermes_integration.py
"""
import json
import tempfile
from pathlib import Path

from memory_etch import EtchMemoryProvider


def main():
    print("=" * 60)
    print("Memory Etch - Hermes Integration Example")
    print("=" * 60)

    # Use a temporary database so cleanup is automatic
    db_path = Path(tempfile.mkdtemp()) / "hermes_demo.db"
    print(f"\n[1] Database path: {db_path}")

    # Create the provider.
    # auto_extract_llm=False disables LLM extraction
    # (otherwise _call_llm_extract would raise RuntimeError).
    provider = EtchMemoryProvider({
        "db_path": str(db_path),
        "auto_extract_llm": False,
    })
    print("[2] EtchMemoryProvider created")

    # Initialize a session
    provider.initialize("demo-session")
    print(f"[3] Initialized session: demo-session")

    # -- Add facts ------------------------------------------------------------
    print("\n-- Adding facts --")

    facts_to_add = [
        {
            "content": "Hermes Agent uses python-telegram-bot v21 for Telegram integration",
            "category": "tool",
            "tags": "python,telegram,hermes",
            "importance": "critical",
        },
        {
            "content": "Memory Etch uses SQLite with FTS5 for full-text search",
            "category": "tool",
            "tags": "python,database,sqlite,memory-etch",
            "importance": "important",
        },
        {
            "content": "HRR vectors provide 256-dimensional similarity search without GPU",
            "category": "tool",
            "tags": "python,vectors,hrr,memory-etch",
            "importance": "important",
        },
    ]

    for fact in facts_to_add:
        result = provider.handle_tool_call("fact_store", {
            "action": "add",
            **fact,
        })
        data = json.loads(result)
        print(f"  Added fact #{data['fact_id']}: {fact['content'][:60]}...")

    # -- Search facts ---------------------------------------------------------
    print("\n-- Searching facts --")

    for query in ["telegram", "sqlite search", "vectors"]:
        result = provider.handle_tool_call("fact_store", {
            "action": "search",
            "query": query,
            "limit": 5,
        })
        data = json.loads(result)
        print(f"\n  Query: \"{query}\" -> {data['count']} result(s)")
        for r in data.get("results", []):
            print(f"    * [{r.get('fact_id')}] {r.get('content', '')[:80]}...")

    # -- Feedback -------------------------------------------------------------
    print("\n-- Providing feedback --")
    # Find the first fact_id to give feedback on
    result = provider.handle_tool_call("fact_store", {
        "action": "search",
        "query": "telegram",
        "limit": 1,
    })
    data = json.loads(result)
    if data.get("results"):
        fid = data["results"][0]["fact_id"]
        provider.handle_tool_call("fact_store", {
            "action": "feedback",
            "fact_id": fid,
            "helpful": True,
        })
        print(f"  Positive feedback recorded for fact #{fid}")

    # -- Shutdown -------------------------------------------------------------
    print("\n-- Cleanup --")
    provider.shutdown()
    print("  Provider shut down successfully")

    # Remove temp database
    if db_path.exists():
        db_path.unlink()
        print(f"  Removed temp database: {db_path}")

    print("\n[OK] Example complete!")


if __name__ == "__main__":
    main()
