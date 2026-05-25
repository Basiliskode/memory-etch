# Memento Examples

Example scripts demonstrating how to use Memento in real-world scenarios.

## Prerequisites

- Python 3.10-3.12
- `pip install memento[hrr]` (adds numpy support for HRR vectors)

## Running Examples

```bash
# Hermes Agent integration — basic provider lifecycle
python examples/hermes_integration.py
```

## What It Shows

### `hermes_integration.py`

Demonstrates the full provider lifecycle:

1. **Initialize** — Create an `EtchMemoryProvider` with a config dict (db_path,
   extractor settings). The provider wraps `EtchStore` and manages sessions.
2. **Add facts** — Use the Hermes `handle_tool_call("fact_store", {"action": "add", ...})`
   API to store structured facts with categories, tags, and importance levels.
3. **Search facts** — Full-text search via FTS5, returning scored results.
4. **Feedback** — Update trust scores on individual facts via the feedback API.
5. **Shutdown** — Cleanly close the database and release resources.

The example uses a temporary directory for the database so no files are left
behind after execution.

## Configuration

Memento is configured through Python dicts at provider creation:

```python
provider = EtchMemoryProvider({
    "db_path": "/path/to/memory.db",
    "auto_extract_llm": True,        # Enable LLM-based fact extraction
    "extract_interval": 5,           # Turns between extractions
    "extract_min_meaningful": 3,     # Min meaningful turns before extraction
    "extract_min_buffer": 5,         # Min buffer size before extraction
    "extract_max_batch": 20,         # Max turns per extraction batch
})
```

For production use with Hermes Agent, configure the provider in
`hermes_config.yml`:

```yaml
memory:
  provider: memento
  config:
    db_path: "~/.hermes/memory.db"
    auto_extract_llm: true
```

## See Also

- [Memento README](https://github.com/Basiliskode/memento)
- [Hermes Agent Documentation](https://github.com/Basiliskode/hermes-agent)
