     1|"""
     2|QueryExpander for Universal Translator.
     3|Expands multi-hop questions into sub-queries for FTS5 retrieval.
     4|
     5|Two modes:
     6|  - Simple (default): regex patterns + FTS5 value derivation. 0 LLM calls.
     7|  - LLM-light (fallback): MiniMax decomposes complex patterns. No JSON.
     8|"""
     9|
    10|import logging
    11|import os
    12|import re
    13|from pathlib import Path
    14|from typing import Optional
    15|
    16|logger = logging.getLogger(__name__)
    17|
    18|# ── Alias loader ───────────────────────────────────────────────────────────
    19|
    20|_ALIASES: dict[str, str] | None = None
    21|
    22|
    23|def _load_aliases() -> dict[str, str]:
    24|    """Load alias → canonical name mapping from aliases.yaml."""
    25|    global _ALIASES
    26|    if _ALIASES is not None:
    27|        return _ALIASES
    28|
    29|    _ALIASES = {}
    30|    yaml_path = Path(__file__).parent / "aliases.yaml"
    31|    if not yaml_path.exists():
    32|        logger.warning("[translator:alias] aliases.yaml not found")
    33|        return _ALIASES
    34|
    35|    try:
    36|        import yaml
    37|        with open(yaml_path) as f:
    38|            data = yaml.safe_load(f)
    39|        for alias, canonical in data.get("aliases", {}).items():
    40|            _ALIASES[alias.lower().strip()] = canonical
    41|    except ImportError:
    42|        # Fallback: manual parse
    43|        with open(yaml_path) as f:
    44|            for line in f:
    45|                line = line.strip()
    46|                if ":" in line and line[0] not in ("#", " "):
    47|                    parts = line.split(":", 1)
    48|                    key = parts[0].strip().strip('"').lower()
    49|                    val = parts[1].strip().strip('"')
    50|                    if key and val:
    51|                        _ALIASES[key] = val
    52|
    53|    logger.info(f"[translator:alias] Loaded {len(_ALIASES)} aliases")
    54|    return _ALIASES
    55|
    56|
    57|def expand_aliases(query: str) -> str:
    58|    """Replace known aliases in query with canonical names."""
    59|    aliases = _load_aliases()
    60|    q_lower = query.lower()
    61|    for alias, canonical in sorted(aliases.items(), key=lambda x: -len(x[0])):
    62|        # Match as word boundary
    63|        pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
    64|        if pattern.search(q_lower):
    65|            q_lower = pattern.sub(canonical, q_lower)
    66|    return q_lower
    67|
    68|
    69|# ── Pattern-based expansion ────────────────────────────────────────────────
    70|
    71|# Recognized multi-hop question patterns
    72|# Each entry: (regex, extractor_fn)
    73|# extractor_fn(match) -> list of sub-query templates with {value} placeholders
    74|
    75|_PATTERNS: list[tuple[re.Pattern, callable]] = [
    76|    # "Who was X when Y was Z?" or "Who was X in the year that Y was Z?"
    77|    (
    78|        re.compile(r"(Who|What)\s+was\s+(.+?)\s+(?:when|in\s+the\s+(?:year|month|day)\s+that)\s+(.+?)\s+(was|were|did)\s+(.+?)\?",
    79|                   re.IGNORECASE),
    80|        lambda m: (
    81|            _extract_subject(m.group(5)),
    82|            _extract_relation(m.group(3), m.group(4), m.group(5)),
    83|        ),
    84|    ),
    85|    # "In what year did X that Y?" → [Y, X founded year]
    86|    (
    87|        re.compile(r"In\s+what\s+year\s+(?:did|was|were)\s+(.+?)\s+that\s+(.+?)\s*(\?|\.)",
    88|                   re.IGNORECASE),
    89|        lambda m: (
    90|            m.group(2).strip(),
    91|            f"{m.group(1).strip()} founded year",
    92|        ),
    93|    ),
    94|    # "When did X Y?" → [X Y]
    95|    (
    96|        re.compile(r"When\s+(?:did|was|were)\s+(.+?)\s+(.+?)\s*(\?|\.)",
    97|                   re.IGNORECASE),
    98|        lambda m: (
    99|            f"{m.group(1).strip()} {m.group(2).strip()}",
   100|        ),
   101|    ),
   102|    # "What is the X of Y that Z?" → [Y Z?, Y X]
   103|    (
   104|        re.compile(r"What\s+(?:is|was|are|were)\s+the\s+(.+?)\s+of\s+(.+?)\s+that\s+(.+?)\s*(\?|\.)",
   105|                   re.IGNORECASE),
   106|        lambda m: (
   107|            f"{m.group(2).strip()} {m.group(3).strip()}",
   108|            f"{m.group(2).strip()} {m.group(1).strip()}",
   109|        ),
   110|    ),
   111|    # "What is the X of Y?" → [Y X]
   112|    (
   113|        re.compile(r"What\s+(?:is|was|are|were)\s+the\s+(.+?)\s+of\s+(.+?)\s*(\?|\.)",
   114|                   re.IGNORECASE),
   115|        lambda m: (
   116|            f"{m.group(2).strip()} {m.group(1).strip()}",
   117|        ),
   118|    ),
   119|    # "Where was X born?" → [X born]
   120|    (
   121|        re.compile(r"Where\s+(?:was|were|is|are)\s+(.+?)\s+(born|located|founded|created|established)\s*(\?|\.)",
   122|                   re.IGNORECASE),
   123|        lambda m: (f"{m.group(1).strip()} {m.group(2).strip()}",),
   124|    ),
   125|    # "Who was the X of Y?" → [Y, Y X]
   126|    (
   127|        re.compile(r"Who\s+was\s+the\s+(.+?)\s+of\s+(.+?)\s*(\?|\.)",
   128|                   re.IGNORECASE),
   129|        lambda m: (
   130|            m.group(2).strip(),
   131|            f"{m.group(2).strip()} {m.group(1).strip()}",
   132|        ),
   133|    ),
   134|    # "In the year X, who was Y?" → [X, Y X]
   135|    (
   136|        re.compile(r"In\s+the\s+year\s+(.+?)[,]\s+who\s+was\s+(.+?)\s*(\?|\.)",
   137|                   re.IGNORECASE),
   138|        lambda m: (
   139|            m.group(1).strip(),
   140|            f"{m.group(2).strip()} {m.group(1).strip()}",
   141|        ),
   142|    ),
   143|    # "Which X Y Z?" → [Y Z]
   144|    (
   145|        re.compile(r"Which\s+(.+?)\s+(.+?)\s+(.+?)\s*(\?|\.)",
   146|                   re.IGNORECASE),
   147|        lambda m: (f"{m.group(2).strip()} {m.group(3).strip()}",),
   148|    ),
   149|    # Generic: extract entity and relation from any multi-hop pattern
   150|    (
   151|        re.compile(r"(Who|What)\s+was\s+(.+?)\s+(?:of|in|for|at)\s+(.+?)\s*(\?|\.)",
   152|                   re.IGNORECASE),
   153|        lambda m: (
   154|            m.group(3).strip(),
   155|            f"{m.group(3).strip()} {m.group(2).strip()}",
   156|        ),
   157|    ),
   158|]
   159|
   160|
   161|def _extract_subject(text: str) -> str:
   162|    """Clean up a subject/entity phrase."""
   163|    return text.strip().strip(".,;:!?")
   164|
   165|
   166|def _extract_relation(entity: str, verb: str, subject: str) -> str:
   167|    """Build a relation query like 'Citibank founded'."""
   168|    e = _extract_subject(entity)
   169|    s = _extract_subject(subject)
   170|    return f"{e} {s}"
   171|
   172|
   173|class QueryExpander:
   174|    """
   175|    Expands a question into sub-queries for multi-hop retrieval.
   176|    Simple mode uses regex patterns (0 LLM calls).
   177|    Light mode uses MiniMax for complex patterns.
   178|    """
   179|
   180|    def __init__(self, llm=None):
   181|        self._llm = llm  # Optional LLM client for light mode
   182|
   183|    def expand(self, query: str, store=None, intent: Optional[dict] = None) -> list[str]:
   184|        """
   185|        Expand question into sub-queries.
   186|
   187|        Args:
   188|            query: The user's question.
   189|            store: Optional EtchStore for value derivation.
   190|            intent: Optional QueryIntent from classifier.py.
   191|
   192|        Returns:
   193|            List of sub-queries (1-3).
   194|        """
   195|        # Step 1: Apply alias expansion
   196|        expanded_query = expand_aliases(query)
   197|
   198|        # Step 2: Check intent — negation queries bypass expansion
   199|        if intent and intent.get("strategy") in ("negation", "list"):
   200|            logger.debug(f"[translator:expand] Bypassing expansion for intent={intent.get('strategy')}")
   201|            return [expanded_query]
   202|
   203|        # Step 3: Try regex patterns
   204|        sub_queries = self._pattern_expand(expanded_query)
   205|        if sub_queries:
   206|            logger.debug(f"[translator:expand] Pattern match: {sub_queries}")
   207|            return sub_queries
   208|
   209|        # Step 4: Try value derivation (e.g., extract year from first query then expand)
   210|        if store:
   211|            derived = self._derive_values(expanded_query, store)
   212|            if derived:
   213|                return derived
   214|
   215|        # Step 5: LLM-light mode (fallback for complex patterns)
   216|        if self._llm:
   217|            light = self._llm_light_expand(expanded_query)
   218|            if light:
   219|                return light
   220|
   221|        # Step 6: No expansion possible — return original
   222|        logger.debug(f"[translator:expand] No expansion found, returning original")
   223|        return [expanded_query]
   224|
   225|    def _pattern_expand(self, query: str) -> list[str]:
   226|        """Try regex pattern matching. Returns sub-queries or empty list."""
   227|        for pattern, handler in _PATTERNS:
   228|            m = pattern.search(query)
   229|            if m:
   230|                result = handler(m)
   231|                if isinstance(result, str):
   232|                    return [result]
   233|                return list(result)
   234|        return []
   235|
   236|    def _derive_values(self, query: str, store) -> list[str]:
   237|        """
   238|        Derive intermediate values from the store.
   239|        E.g., for "Who was president when Citibank was founded?",
   240|        find "Citibank founded 1812" in store, then return ["Citibank founded", "president 1812"].
   241|        """
   242|        # Extract potential entities (capitalized words)
   243|        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
   244|        if not entities:
   245|            return []
   246|
   247|        # Try to find a year/value match in the store
   248|        derived = []
   249|        for ent in entities[:3]:
   250|            # Search for triples about this entity
   251|            try:
   252|                results = store.search_facts(query=f"{ent} | *", category="triple", limit=3)
   253|            except Exception:
   254|                continue
   255|
   256|            for r in results:
   257|                content = r.get("content", "")
   258|                parsed = parse_triple(content)
   259|                if not parsed:
   260|                    continue
   261|                s, p, o = parsed
   262|                # If the object looks like a year or value, use it for expansion
   263|                if re.match(r"^\d{4}$", o.strip()) or re.match(r"^\d+$", o.strip()):
   264|                    # Build second sub-query: "X <entity> <year>"
   265|                    remaining = self._remaining_entity(query, ent)
   266|                    if remaining:
   267|                        derived.append(f"{remaining} {o}")
   268|                    derived.append(f"{ent} {p}")
   269|                    return derived
   270|
   271|        return []
   272|
   273|    def _remaining_entity(self, query: str, matched_entity: str) -> str:
   274|        """Extract the 'other' entity from a multi-hop question."""
   275|        # Remove the matched entity and common filler words
   276|        remaining = re.sub(re.escape(matched_entity), "", query, flags=re.IGNORECASE)
   277|        remaining = re.sub(
   278|            r"^(Who|What|When|Where|How|Which|Why|In what|In which)\s+(was|were|did|is|are|the|a|an)\s+",
   279|            "", remaining, flags=re.IGNORECASE
   280|        )
   281|        remaining = remaining.strip(" ?.,;:")
   282|        if len(remaining) > 3:
   283|            return remaining
   284|        return ""
   285|
   286|    def _llm_light_expand(self, query: str) -> list[str]:
   287|        """Use a cheap LLM (no JSON) to decompose complex questions."""
   288|        if not self._llm:
   289|            return []
   290|
   291|        prompt = (
   292|            f"Break this question into simpler sub-questions. One per line, no numbering.\n"
   293|            f"Question: {query}\n"
   294|            f"Sub-questions:\n"
   295|        )
   296|        try:
   297|            raw = self._llm(prompt)
   298|            sub_queries = [
   299|                line.strip().strip("-•*1234567890. ") for line in raw.strip().split("\n")
   300|                if line.strip() and not line.strip().startswith(("Sub-questions", "Question:", "```"))
   301|            ]
   302|            return sub_queries[:3]  # hard limit
   303|        except Exception as e:
   304|            logger.warning(f"[translator:expand] LLM-light failed: {e}")
   305|            return []
   306|
   307|
   308|def parse_triple(content: str) -> Optional[tuple[str, str, str]]:
   309|    """Parse 'subject | predicate | object' back to tuple."""
   310|    parts = [p.strip() for p in content.split(" | ")]
   311|    if len(parts) == 3 and all(parts):
   312|        return (parts[0], parts[1], parts[2])
   313|    return None
   314|