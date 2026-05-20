"""Synthetic dataset generator for memory benchmarks.

Generates 5 personas with structured facts, events, and preferences,
plus natural-language queries with gold answers.
"""

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """A document in the benchmark dataset."""
    id: str
    content: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)


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


class SyntheticDataset:
    """Generate a synthetic dataset of facts and queries for memory benchmarking.

    Args:
        seed: Random seed for reproducibility.
        n_docs: Number of documents to generate (default: 100).
        n_queries: Number of queries to generate (default: 20).

    Example::

        dataset = SyntheticDataset(seed=42, n_docs=100)
        docs, queries = dataset.generate()
    """

    def __init__(self, seed: int = 42, n_docs: int = 100, n_queries: int = 20):
        self._seed = seed
        self._n_docs = n_docs
        self._n_queries = n_queries

    def generate(self):
        """Generate synthetic documents and queries.

        Returns:
            tuple[list[Document], list[dict]]: Documents and query dicts.
        """
        rng = random.Random(self._seed)
        doc_id = [0]

        def next_id() -> str:
            doc_id[0] += 1
            return f"synth-{doc_id[0]:04d}"

        documents = []
        queries = []

        # ── Context personas ─────────────────────────────────────────────────
        for p in PERSONAS:
            facts = [
                f"{p['name']} is {p['age']} years old.",
                f"{p['name']} works as a {p['profession']} at {p['company']}.",
                f"{p['name']} lives in {p['city']}.",
                f"{p['name']}'s hobbies include {', '.join(p['hobbies'])}.",
                f"{p['name']} has {p['pet']}.",
                f"{p['name']}: {p['recent_event']}",
            ]
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
        for p in PERSONAS:
            for _ in range(3):
                event_descriptions = [
                    f"On 2025.{rng.randint(1,4)}.{rng.randint(1,28)}, {p['name']} attended "
                    f"{rng.choice(['a tech conference', 'a design workshop', 'a research symposium', 'a startup meetup'])} "
                    f"in {rng.choice(['Berlin', 'Tokyo', 'London', 'Austin'])}.",
                    f"Last {rng.choice(['Monday', 'Wednesday', 'Friday'])}, {p['name']} had a meeting with "
                    f"{rng.choice(['the CEO', 'the CTO', 'a client from Japan', 'the research team'])} about "
                    f"{rng.choice(['Q4 planning', 'the new product roadmap', 'budget allocation', 'collaboration opportunities'])}.",
                    f"In {rng.choice(['January', 'March', 'June', 'September'])} {2025 - rng.randint(0,2)}, "
                    f"{p['name']} finished reading "
                    f"'{rng.choice(['Designing Data-Intensive Applications', 'The Pragmatic Programmer', 'Thinking, Fast and Slow', 'The Structure of Scientific Revolutions'])}' "
                    f"and {rng.choice(['highly recommended it', 'found it mediocre', 'plans to re-read it', 'disagreed with the main thesis'])}.",
                    f"{p['name']} recently started learning {rng.choice(['Rust', 'Go', 'Elixir', 'three.js', 'Figment'])} "
                    f"and {rng.choice(['loves the experience', 'finds it challenging but rewarding', 'already built a small project with it', 'is struggling with the borrow checker'])}.",
                ]
                documents.append(Document(
                    id=next_id(),
                    content=rng.choice(event_descriptions),
                    user_id=p["name"].lower().replace(" ", "_"),
                    timestamp=f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T14:30:00",
                    metadata={"persona": p["name"], "type": "event"},
                ))

        # ── Preference statements ────────────────────────────────────────────
        for p in PERSONAS:
            preferences = [
                f"{p['name']} strongly prefers {rng.choice(['TypeScript', 'Python', 'Rust', 'Go'])} over other programming languages.",
                f"{p['name']} believes the best approach to project management is "
                f"{rng.choice(['Agile with 2-week sprints', 'Basecamp-style Shape Up', 'Kanban with WIP limits', 'waterfall for regulated projects'])}.",
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
        while len(documents) < self._n_docs:
            p = rng.choice(PERSONAS)
            topic = rng.choice(["weather", "sports", "movies", "music", "travel",
                                "food", "technology", "science", "art", "books"])
            filler = (f"{p['name']} mentioned that {topic} has been "
                      f"{rng.choice(['quite interesting lately', 'a topic of discussion at work', 'something they want to explore more', 'on their mind recently'])}.")
            documents.append(Document(
                id=next_id(),
                content=filler,
                user_id=p["name"].lower().replace(" ", "_"),
                timestamp=f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(8,18):02d}:00:00",
                metadata={"persona": p["name"]},
            ))

        # ── Generate queries ─────────────────────────────────────────────
        # Fact queries
        for p in PERSONAS:
            first_name = p["name"].split()[0].lower()
            queries.append({
                "id": f"q-fact-{first_name}",
                "query": f"What is {p['name'].split()[0]}'s profession?",
                "gold_answers": [p["profession"]],
                "gold_ids": [],
                "user_id": p["name"].lower().replace(" ", "_"),
                "meta": {"type": "fact", "difficulty": "easy"},
            })
            queries.append({
                "id": f"q-event-{first_name}",
                "query": f"What recent achievement does {p['name'].split()[0]} have?",
                "gold_answers": [p["recent_event"]],
                "gold_ids": [],
                "user_id": p["name"].lower().replace(" ", "_"),
                "meta": {"type": "event", "difficulty": "medium"},
            })
            queries.append({
                "id": f"q-pet-{first_name}",
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

        # Trim
        rng.shuffle(queries)
        queries = queries[:self._n_queries]

        return documents, queries
