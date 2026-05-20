#!/usr/bin/env python3
"""
Standalone AMB-style benchmark runner for memory-etch.

Generates synthetic data to test memory recall across personas, facts,
events, and preferences. Self-contained — no external datasets needed.

Usage:
    set GEMINI_API_KEY=AIzaSy...
    python scripts/run_amb_benchmark.py --query-limit 5
    python scripts/run_amb_benchmark.py --query-limit 5 --memory etch-emb
    python scripts/run_amb_benchmark.py --help
"""

import argparse
import json
import os
import sys
import time
import tempfile
import random
from pathlib import Path

# Ensure we can import memory-etch
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Import etch adapter directly (avoids AMB __init__.py which imports ALL providers)
sys.path.insert(0, str(Path(__file__).parent))  # scripts/
from amb_adapter import EtchMemoryProvider, EtchEmbMemoryProvider, EtchHybridMemoryProvider
from amb_adapter import Document
from google import genai


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════════

PERSONAS = [
    {
        "name": "Alice Chen",
        "age": 32,
        "profession": "software engineer",
        "company": "Google",
        "city": "San Francisco",
        "hobbies": ["rock climbing", "photography", "cooking"],
        "pet": "a golden retriever named Sunny",
        "recent_event": "promoted to Staff Engineer last month",
    },
    {
        "name": "Bob Martinez",
        "age": 45,
        "profession": "architect",
        "company": "self-employed",
        "city": "Barcelona",
        "hobbies": ["sailing", "wine tasting", "reading history"],
        "pet": "two cats, Luna and Milo",
        "recent_event": "won an international design award for a museum in Oslo",
    },
    {
        "name": "Carol Williams",
        "age": 28,
        "profession": "data scientist",
        "company": "Spotify",
        "city": "Stockholm",
        "hobbies": ["bouldering", "electronic music production", "hiking"],
        "pet": "no pets, but wants a ferret",
        "recent_event": "published a paper on music recommendation at KDD 2025",
    },
    {
        "name": "David Kim",
        "age": 38,
        "profession": "product manager",
        "company": "Notion",
        "city": "New York",
        "hobbies": ["chess", "board games", "jazz guitar"],
        "pet": "a parakeet named Pixel",
        "recent_event": "shipped a major AI feature that uses natural language for database queries",
    },
    {
        "name": "Elena Rodriguez",
        "age": 52,
        "profession": "neuroscientist",
        "company": "MIT",
        "city": "Boston",
        "hobbies": ["marathon running", "pottery", "bird watching"],
        "pet": "a rescue greyhound named Bolt",
        "recent_event": "received a $2M grant for Alzheimer's research",
    },
]


def generate_synthetic_dataset(seed: int = 42, n_docs: int = 100, n_queries: int = 20):
    """Generate synthetic documents and queries that test memory retrieval.

    Returns (documents: list[Document], queries: list[dict])
    """
    rng = random.Random(seed)
    doc_id = [0]

    def next_id() -> str:
        doc_id[0] += 1
        return f"synth-{doc_id[0]:04d}"

    documents = []
    queries = []

    # ── Context personas ─────────────────────────────────────────────────
    for p in PERSONAS:
        # Personal facts
        facts = [
            f"{p['name']} is {p['age']} years old.",
            f"{p['name']} works as a {p['profession']} at {p['company']}.",
            f"{p['name']} lives in {p['city']}.",
            f"{p['name']}'s hobbies include {', '.join(p['hobbies'])}.",
            f"{p['name']} has {p['pet']}.",
            f"{p['name']}: {p['recent_event']}",
        ]
        # Add more specific facts
        extra_facts = [
            f"{p['name']}'s favorite food is {'pizza' if rng.random() > 0.5 else 'sushi'}.",
            f"{p['name']} {'speaks Spanish fluently' if rng.random() > 0.5 else 'is learning French'}.",
            f"{p['name']} prefers {'remote work' if rng.random() > 0.5 else 'office work'}.",
        ]
        facts.extend(extra_facts)

        for i, fact in enumerate(facts):
            documents.append(Document(
                id=next_id(),
                content=fact,
                user_id=p["name"].lower().replace(" ", "_"),
                timestamp=f"2025-01-{i+1:02d}T10:00:00",
                metadata={"persona": p["name"]},
            ))

    # ── Event narratives ─────────────────────────────────────────────────
    events = []
    for p in PERSONAS:
        for i in range(3):
            event_descriptions = [
                f"On {2025-i}.{rng.randint(1,4)}.{rng.randint(1,28)}, {p['name']} attended {rng.choice(['a tech conference', 'a design workshop', 'a research symposium', 'a startup meetup'])} in {rng.choice(['Berlin', 'Tokyo', 'London', 'Austin'])}.",
                f"Last {rng.choice(['Monday', 'Wednesday', 'Friday'])}, {p['name']} had a meeting with {rng.choice(['the CEO', 'the CTO', 'a client from Japan', 'the research team'])} about {rng.choice(['Q4 planning', 'the new product roadmap', 'budget allocation', 'collaboration opportunities'])}.",
                f"In {rng.choice(['January', 'March', 'June', 'September'])} {2025 - rng.randint(0,2)}, {p['name']} finished reading '{rng.choice(['Designing Data-Intensive Applications', 'The Pragmatic Programmer', 'Thinking, Fast and Slow', 'The Structure of Scientific Revolutions'])}' and {rng.choice(['highly recommended it', 'found it mediocre', 'plans to re-read it', 'disagreed with the main thesis'])}.",
                f"{p['name']} recently started learning {rng.choice(['Rust', 'Go', 'Elixir', 'three.js', 'Figment'])} and {rng.choice(['loves the experience', 'finds it challenging but rewarding', 'already built a small project with it', 'is struggling with the borrow checker'])}.",
            ]
            doc = Document(
                id=next_id(),
                content=rng.choice(event_descriptions),
                user_id=p["name"].lower().replace(" ", "_"),
                timestamp=f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T14:30:00",
                metadata={"persona": p["name"], "type": "event"},
            )
            events.append(doc)
            documents.append(doc)

    # ── Preference statements ────────────────────────────────────────────
    for p in PERSONAS:
        preferences = [
            f"{p['name']} strongly prefers {rng.choice(['TypeScript', 'Python', 'Rust', 'Go'])} over other programming languages.",
            f"{p['name']} believes the best approach to project management is {rng.choice(['Agile with 2-week sprints', 'Basecamp-style Shape Up', 'Kanban with WIP limits', 'waterfall for regulated projects'])}.",
            f"{p['name']} is {'a morning person who wakes up at 6AM' if rng.random() > 0.5 else 'a night owl who does their best work after midnight'}.",
            f"In meetings, {p['name']} prefers {rng.choice(['written documents read in advance', 'whiteboard brainstorming sessions', 'quick standups, no more than 15 min', 'async Loom videos'])}.",
        ]
        for pref in preferences:
            documents.append(Document(
                id=next_id(),
                content=pref,
                user_id=p["name"].lower().replace(" ", "_"),
                timestamp="2025-06-15T09:00:00",
                metadata={"persona": p["name"], "type": "preference"},
            ))

    # ── Fill up to n_docs with filler facts ────────────────────────────
    filler_topics = [
        "weather", "sports", "movies", "music", "travel",
        "food", "technology", "science", "art", "books"
    ]
    while len(documents) < n_docs:
        p = rng.choice(PERSONAS)
        topic = rng.choice(filler_topics)
        filler = f"{p['name']} mentioned that {topic} has been {rng.choice(['quite interesting lately', 'a topic of discussion at work', 'something they want to explore more', 'on their mind recently'])}."
        documents.append(Document(
            id=next_id(),
            content=filler,
            user_id=p["name"].lower().replace(" ", "_"),
            timestamp=f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(8,18):02d}:00:00",
            metadata={"persona": p["name"]},
        ))

    # ── Generate queries from the data we know exists ─────────────────────
    # Fact queries
    for p in PERSONAS:
        queries.append({
            "id": f"q-fact-{p['name'].split()[0].lower()}",
            "query": f"What is {p['name'].split()[0]}'s profession?",
            "gold_answers": [f"{p['profession']}"],
            "gold_ids": [],
            "user_id": p["name"].lower().replace(" ", "_"),
            "meta": {"type": "fact", "difficulty": "easy"},
        })
        queries.append({
            "id": f"q-event-{p['name'].split()[0].lower()}",
            "query": f"What recent achievement does {p['name'].split()[0]} have?",
            "gold_answers": [p["recent_event"]],
            "gold_ids": [],
            "user_id": p["name"].lower().replace(" ", "_"),
            "meta": {"type": "event", "difficulty": "medium"},
        })
        queries.append({
            "id": f"q-pet-{p['name'].split()[0].lower()}",
            "query": f"Does {p['name'].split()[0]} have any pets?",
            "gold_answers": [f"{p['name']} has {p['pet']}"],
            "gold_ids": [],
            "user_id": p["name"].lower().replace(" ", "_"),
            "meta": {"type": "fact", "difficulty": "easy"},
        })

    # Multi-fact reasoning queries
    queries.append({
        "id": "q-reason-1",
        "query": "Who lives in Barcelona and recently won a design award?",
        "gold_answers": ["Bob Martinez"],
        "gold_ids": [],
        "user_id": None,
        "meta": {"type": "reasoning", "difficulty": "hard"},
    })
    queries.append({
        "id": "q-reason-2",
        "query": "Which person is a data scientist and published a paper at KDD?",
        "gold_answers": ["Carol Williams"],
        "gold_ids": [],
        "user_id": None,
        "meta": {"type": "reasoning", "difficulty": "hard"},
    })
    queries.append({
        "id": "q-reason-3",
        "query": "Who has a dog named Sunny and lives in San Francisco?",
        "gold_answers": ["Alice Chen"],
        "gold_ids": [],
        "user_id": None,
        "meta": {"type": "reasoning", "difficulty": "hard"},
    })

    # Trim to exact counts
    rng.shuffle(queries)
    queries = queries[:n_queries]
    documents = documents[:n_docs]

    return documents, queries


# ═══════════════════════════════════════════════════════════════════════════════
#  Gemini Judge
# ═══════════════════════════════════════════════════════════════════════════════

def judge_answer(question: str, retrieved_context: str, gold_answer: str) -> tuple[bool | None, dict]:
    """Use Gemini to judge if the retrieved context supports the gold answer.

    Returns (True/False on success, None if judge API call fails, metadata dict).
    """
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        prompt = f"""You are a strict judge evaluating a memory retrieval system.

Given a QUESTION, the RETRIEVED CONTEXT, and the EXPECTED ANSWER, determine if the
retrieved context contains enough information to answer the question correctly.

QUESTION: {question}

RETRIEVED CONTEXT:
{retrieved_context}

EXPECTED ANSWER: {gold_answer}

RESPOND WITH ONLY: YES or NO
YES = The retrieved context contains enough information to answer the question correctly
NO = The retrieved context does NOT contain enough information

Your verdict:"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={"temperature": 0.0, "max_output_tokens": 10},
        )

        # Extract token usage
        prompt_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        verdict = response.text.strip().upper().startswith("YES")
        return verdict, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}

    except Exception as exc:
        print(f"  [!] Judge error: {exc}")
        return None, {"prompt_tokens": 0, "output_tokens": 0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(memory_name: str, query_limit: int = None,
                  verbose: bool = False, n_docs: int = 100, seed: int = 42):
    """Run memory benchmark with synthetic data."""

    # Generate synthetic dataset
    print(f"\n{'='*60}")
    print(f"Dataset: synthetic (seed={seed}, {n_docs} docs)")
    print(f"Memory:  {memory_name}")
    print(f"{'='*60}\n")

    print("Generating synthetic dataset...")
    documents, queries = generate_synthetic_dataset(seed=seed, n_docs=n_docs)
    print(f"  {len(documents)} documents, {len(queries)} queries")

    if query_limit:
        queries = queries[:query_limit]
        print(f"  (limited to {query_limit} queries)")

    # Initialize memory provider
    print("\nInitializing memory provider...")

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="amb_etch_")
    tmpdir = tmpdir_obj.name
    store_dir = os.path.join(tmpdir, "store")
    os.makedirs(store_dir, exist_ok=True)

    if memory_name == "etch-emb":
        provider = EtchEmbMemoryProvider()
    elif memory_name == "etch-hybrid":
        provider = EtchHybridMemoryProvider()
    else:
        provider = EtchMemoryProvider()

    provider_ready = False
    try:
        provider.prepare(store_dir=store_dir, unit_ids=None, reset=True)
        provider_ready = True

        # Ingest documents
        print(f"Ingesting {len(documents)} documents...")
        t0 = time.time()
        provider.ingest(documents)
        ingest_time = time.time() - t0
        print(f"  Done in {ingest_time:.2f}s ({len(documents)/ingest_time:.0f} docs/s)")

        # Run queries
        print(f"\nRunning {len(queries)} queries...")
        results = []
        correct = 0

        for i, q in enumerate(queries, 1):
            t0 = time.time()
            retrieved_docs, meta = provider.retrieve(
                query=q["query"],
                k=15,  # recall@15: FTS5+HRR can't reliably rank NL queries in top-5
                user_id=q.get("user_id"),
            )
            retrieve_time = (time.time() - t0) * 1000  # ms
            retrieve_time_ms = round(retrieve_time, 1)

            # Build context string from retrieved docs
            context = "\n---\n".join(d.content for d in retrieved_docs)
            context_tokens = len(context.split())

            # Track retrieved IDs
            retrieved_ids = [d.id for d in retrieved_docs]

            # Judge each gold answer
            gold = q["gold_answers"][0] if q["gold_answers"] else ""
            verdict, token_meta = judge_answer(q["query"], context, gold) if gold else (False, {})
            is_correct = bool(verdict) if verdict is not None else False
            judged = verdict is not None
            judge_prompt_tokens = token_meta.get("prompt_tokens", 0)
            judge_output_tokens = token_meta.get("output_tokens", 0)

            result = {
                "query_id": q["id"],
                "query": q["query"],
                "gold_answers": q["gold_answers"],
                "retrieved_ids": retrieved_ids,
                "retrieve_time_ms": retrieve_time_ms,
                "context_tokens": context_tokens,
                "judge_prompt_tokens": judge_prompt_tokens,
                "judge_output_tokens": judge_output_tokens,
                "correct": is_correct,
                "judged": judged,
            }
            results.append(result)

            if is_correct:
                correct += 1

            if verbose or i <= 3 or not is_correct:
                if not judged:
                    status = "?"
                else:
                    status = "[OK]" if is_correct else "[NO]"
                print(f"  [{i}/{len(queries)}] {status} {q['query'][:70]}... ({retrieve_time_ms}ms)")

        # Summary
        accuracy = correct / len(queries) if queries else 0
        avg_time = sum(r["retrieve_time_ms"] for r in results) / len(results) if results else 0
        avg_tokens = sum(r["context_tokens"] for r in results) / len(results) if results else 0
        total_prompt_tokens = sum(r.get("judge_prompt_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("judge_output_tokens", 0) for r in results)

        print(f"\n{'='*60}")
        print(f"RESULTS: {memory_name}")
        print(f"{'='*60}")
        print(f"  Total queries:    {len(queries)}")
        print(f"  Correct:          {correct}/{len(queries)}")
        print(f"  Accuracy:         {accuracy:.1%}")
        print(f"  Avg retrieve:     {avg_time:.1f}ms")
        print(f"  Avg ctx tokens:   {avg_tokens:.0f}")
        print(f"  Judge prompt:     {total_prompt_tokens} total ({total_prompt_tokens//max(len(queries),1):.0f}/query)")
        print(f"  Judge output:     {total_output_tokens} total ({total_output_tokens//max(len(queries),1):.0f}/query)")
        print(f"  Ingest speed:     {len(documents)/ingest_time:.0f} docs/s")
        print(f"{'='*60}\n")

    finally:
        if provider_ready:
            provider.cleanup()
        try:
            tmpdir_obj.cleanup()
        except Exception:
            pass

    return {
        "dataset": "synthetic",
        "memory": memory_name,
        "seed": seed,
        "total_documents": n_docs,
        "total_queries": len(queries),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_retrieve_time_ms": round(avg_time, 1),
        "avg_context_tokens": round(avg_tokens, 1),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-etch benchmark")
    parser.add_argument("--memory", default="etch", choices=["etch", "etch-emb", "etch-hybrid"],
                        help="Memory provider variant")
    parser.add_argument("--query-limit", type=int, default=None,
                        help="Limit number of queries")
    parser.add_argument("--n-docs", type=int, default=100,
                        help="Number of synthetic documents")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all query results")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    result = run_benchmark(
        memory_name=args.memory,
        query_limit=args.query_limit,
        verbose=args.verbose,
        n_docs=args.n_docs,
        seed=args.seed,
    )

    # Save results
    output_dir = Path("outputs") / result["dataset"] / args.memory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed{args.seed}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {output_path}")
