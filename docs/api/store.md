# EtchStore

```python
class EtchStore:
    """SQLite-backed fact store with trust scoring, HRR vectors, and consolidation."""

    def __init__(
        self,
        db_path: str,
        hrr_dim: int = 256,
        auto_migrate: bool = True,
    ) -> None: ...
```

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | — | Path to the SQLite database file |
| `hrr_dim` | `int` | `256` | Dimension for HRR vectors |
| `auto_migrate` | `bool` | `True` | Run schema creation and migration on init |

## Methods

### Fact CRUD

```python
def add_fact(
    self,
    content: str,
    category: str = "general",
    tags: str = "",
    trust_score: float | None = None,
    importance: float | None = None,
    project: str = "",
    session_id: str = "",
    topic_key: str = "",
    entities: list[str] | None = None,
    embedding: bytes | None = None,
) -> int: ...
```

Insert a new fact. When `tags` contains `topic:<name>`, `topic_key` is auto-extracted and an existing fact with the same key is updated (topic upsert). Returns the `fact_id`.

---

```python
def add_fact_with_consolidation(
    self,
    content: str,
    category: str = "general",
    tags: str = "",
    trust_score: float | None = None,
    importance: float | None = None,
    project: str = "",
    session_id: str = "",
    topic_key: str = "",
    entities: list[str] | None = None,
    search_fn: Callable | None = None,
    llm_decide_fn: Callable | None = None,
) -> dict: ...
```

Add a fact with active consolidation — merges or deletes old facts on collision. Returns `{"action": "added"|"merged"|"skipped"|"error", "fact_id": int, "detail": str}`.

---

```python
def get_fact(self, fact_id: int) -> dict | None: ...
```

Get a single fact by ID. Returns the fact dict (minus blobs) or `None`.

---

```python
def update_fact(self, fact_id: int, **kwargs) -> bool: ...
```

Update fact fields. Allowed keys: `content`, `category`, `tags`, `trust_score`, `importance`, `project`.

---

```python
def remove_fact(self, fact_id: int) -> bool: ...
```

Permanently delete a fact. Use `soft_delete_fact` for reversible deletion.

---

```python
def soft_delete_fact(self, fact_id: int, reason: str = "") -> bool: ...
```

Soft-delete a fact. It remains in the database but is excluded from searches by default.

---

```python
def list_facts(
    self,
    category: str = "",
    project: str = "",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]: ...
```

List facts with optional filters.

---

```python
def purge_facts(self, dry_run: bool = True) -> dict: ...
```

Purge low-value facts (>90d old, trust <0.3, importance <0.5). Returns stats.

### Search

```python
def search(
    self,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    project: str = "",
) -> list[dict]: ...
```

Full-text search via FTS5 with optional project filter.

---

```python
def search_by_vector(
    self,
    query_vector: bytes,
    limit: int = 10,
    min_trust: float = 0.0,
    category: str = "",
    project: str = "",
) -> list[dict]: ...
```

Search facts by embedding vector (cosine similarity). Returns results sorted by similarity descending.

### Sessions

```python
def start_session(
    self,
    session_id: str,
    project: str = "",
    metadata: dict | None = None,
) -> bool: ...

def end_session(self, session_id: str, summary: str = "") -> bool: ...

def get_session(self, session_id: str) -> dict | None: ...

def list_sessions(self, project: str = "", limit: int = 10) -> list[dict]: ...
```

Session lifecycle management. Sessions track fact chronology and extraction context.

### Relations

```python
def add_relation(
    self,
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str = "related",
    confidence: float = 0.5,
    judged_by: str = "auto",
) -> bool: ...

def get_relations(self, fact_id: int) -> list[dict]: ...

def get_contradictions(self, limit: int = 10) -> list[dict]: ...

def judge_relation(
    self,
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str = "related",
    confidence: float = 0.5,
    judged_by: str = "auto",
) -> dict: ...
```

Record and query fact relations. `relation_type` must be one of: `related`, `compatible`, `scoped`, `conflicts_with`, `supersedes`, `not_conflict`.

### Entities

```python
def get_entities(self, fact_id: int) -> list[dict]: ...
```

Get entities associated with a fact.

### Timeline

```python
def get_timeline(
    self,
    fact_id: int,
    before: int = 5,
    after: int = 5,
) -> dict: ...
```

Get chronological context around a fact.

### Stats & Utilities

```python
def stats(self) -> dict: ...
def projects(self) -> list[str]: ...
def close(self) -> None: ...
```
