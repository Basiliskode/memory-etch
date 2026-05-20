# QueryClassifier

```python
class QueryClassifier:
    """Simple rule-based classifier for memory queries."""

    def classify(self, query: str) -> dict: ...
```

## Methods

### classify

```python
def classify(self, query: str) -> dict: ...
```

Classify a query into an intent and extract entities. Uses keyword pattern matching to determine the query intent.

**Returns**:
```python
{
    "intent": str,        # "search" | "entity" | "probe" | "project"
                          # | "relation" | "timeline" | "contradict" | "empty"
    "entities": list[str],  # Capitalized phrases detected as entity names
    "keywords": list[str],  # Meaningful keywords (nouns, no stopwords)
}
```

**Intents**:

| Intent | Trigger Patterns |
|--------|-----------------|
| `entity` | "what do i know about", "tell me about", "what is", "who is" |
| `probe` | "probe" |
| `project` | "project:", "project ..." |
| `relation` | "relation", "between" |
| `timeline` | "timeline", "history of" |
| `search` | "search" (default fallback) |
| `contradict` | "contradict", "conflict" |
| `empty` | empty input, greetings (hi, hello, thanks, etc.) |

The classifier is used internally by the retrieval system to route queries to the appropriate search strategy.
