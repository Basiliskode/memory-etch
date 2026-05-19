     1|"""
     2|TranslatorPipeline — Universal Translator for Etch.
     3|
     4|Orchestrates the 3-stage pipeline:
     5|  1. INGEST: Raw text → TripleExtractor → Normalizer → EtchStore
     6|  2. QUERY: Question → QueryExpander → EtchRetriever → ResponseFormatter
     7|  3. FALLBACK: Triple → Raw facts → list_facts (3 levels)
     8|
     9|Runtime requires 0 JSON from the LLM. Works with any model.
    10|"""
    11|
    12|import logging
    13|import os
    14|import sys
    15|import time
    16|from typing import Optional
    17|
    18|logger = logging.getLogger(__name__)
    19|
    20|# Lazy imports to avoid circular dependencies
    21|_EXTRACTOR = None
    22|_FORMATTER = None
    23|_EXPANDER = None
    24|_NORMALIZER = None
    25|
    26|
    27|def _get_extractor(**kwargs):
    28|    global _EXTRACTOR
    29|    if _EXTRACTOR is None:
    30|        from .extractor import TripleExtractor
    31|        _EXTRACTOR = TripleExtractor(**kwargs)
    32|    return _EXTRACTOR
    33|
    34|
    35|def _get_formatter():
    36|    from .formatter import format_triple, format_answer_prompt, extract_answer
    37|    return format_triple, format_answer_prompt, extract_answer
    38|
    39|
    40|def _get_expander(llm=None):
    41|    global _EXPANDER
    42|    if _EXPANDER is None:
    43|        from .expander import QueryExpander
    44|        _EXPANDER = QueryExpander(llm=llm)
    45|    return _EXPANDER
    46|
    47|
    48|class TranslatorPipeline:
    49|    """
    50|    Universal Translator Pipeline.
    51|
    52|    Usage:
    53|        pipeline = TranslatorPipeline(extractor_model="deepseek-v4-pro")
    54|        pipeline.ingest("Citibank was founded in 1812.", doc_id="citibank")
    55|        answer = pipeline.query("Who was president when Citibank was founded?")
    56|    """
    57|
    58|    def __init__(
    59|        self,
    60|        etch_store=None,
    61|        etch_retriever=None,
    62|        extractor_model: str = "deepseek-v4-pro",
    63|        extractor_api_key: Optional[str] = None,
    64|        extractor_base_url: Optional[str] = None,
    65|        answer_llm: Optional[callable] = None,
    66|        translator_db_path: Optional[str] = None,
    67|    ):
    68|        self._etch_store = etch_store
    69|        self._etch_retriever = etch_retriever
    70|        self._translator_db_path = translator_db_path
    71|
    72|        # Lazy init for translator-specific store
    73|        self._translator_store = None
    74|        self._translator_retriever = None
    75|
    76|        # Extractor config
    77|        self._extractor_kwargs = {
    78|            "model": extractor_model,
    79|            "api_key": extractor_api_key,
    80|            "base_url": extractor_base_url,
    81|        }
    82|
    83|        # Answer LLM (called with prompt, returns text — no JSON needed)
    84|        self._answer_llm = answer_llm or self._default_answer_llm
    85|
    86|        # Stats
    87|        self._stats = {
    88|            "triples_extracted": 0,
    89|            "triples_stored": 0,
    90|            "queries_total": 0,
    91|            "expansions_mode_simple": 0,
    92|            "expansions_mode_light": 0,
    93|            "fallback_triple": 0,
    94|            "fallback_raw": 0,
    95|            "fallback_list": 0,
    96|            "empty_responses": 0,
    97|        }
    98|
    99|    # ── Ingestion ──────────────────────────────────────────────────────────
   100|
   101|    def ingest(self, text: str, doc_id: str, title: str = "") -> int:
   102|        """
   103|        Ingest raw text into the translator's triple store.
   104|
   105|        Args:
   106|            text: Raw text content (e.g., Wikipedia article).
   107|            doc_id: Unique document identifier for rollback tracking.
   108|            title: Optional document title.
   109|
   110|        Returns:
   111|            Number of triples stored.
   112|        """
   113|        from .normalizer import normalize_value
   114|
   115|        extractor = _get_extractor(**self._extractor_kwargs)
   116|        fmt_triple, _, _ = _get_formatter()
   117|
   118|        # Step 1: Extract triples
   119|        raw_triples = extractor.extract(text)
   120|        self._stats["triples_extracted"] += len(raw_triples)
   121|
   122|        if not raw_triples:
   123|            logger.warning(f"[translator:ingest] No triples extracted for doc_id={doc_id}")
   124|            return 0
   125|
   126|        # Step 2: Normalize and format
   127|        store = self._get_translator_store()
   128|        session_id = f"ingest_{doc_id}_{int(time.time())}"
   129|        stored_count = 0
   130|        skip_count = 0
   131|
   132|        for t in raw_triples:
   133|            s = t["subject"].strip()
   134|            p = t["predicate"].strip().lower()
   135|            o = normalize_value(t["object"])
   136|            formatted = fmt_triple(s, p, o)
   137|
   138|            try:
   139|                store.add_fact(
   140|                    content=formatted,
   141|                    category="triple",
   142|                    tags=f"source:{doc_id},triple",
   143|                    session_id=session_id,
   144|                )
   145|                stored_count += 1
   146|            except Exception as e:
   147|                # Handle UNIQUE constraint gracefully
   148|                skip_count += 1
   149|                logger.debug(f"[translator:ingest] Skip triple (dup/error): {formatted[:60]} — {e}")
   150|
   151|        store.flush_banks()
   152|        self._stats["triples_stored"] += stored_count
   153|
   154|        logger.info(
   155|            f"[translator:ingest] doc_id={doc_id} stored={stored_count} "
   156|            f"skipped={skip_count} total_extracted={len(raw_triples)}"
   157|        )
   158|        return stored_count
   159|
   160|    def rollback_ingest(self, doc_id: str) -> int:
   161|        """Remove all triples for a given document (rollback)."""
   162|        store = self._get_translator_store()
   163|        # Find all facts with matching source tag (filter in Python since search_facts has no tags param)
   164|        facts = store.search_facts(query=f"source:{doc_id}", category="triple", limit=1000)
   165|        removed = 0
   166|        for f in facts:
   167|            tags = f.get("tags", "")
   168|            if f"source:{doc_id}" in tags:
   169|                try:
   170|                    store.remove_fact(f["fact_id"])
   171|                    removed += 1
   172|                except Exception:
   173|                    pass
   174|        logger.info(f"[translator:rollback] doc_id={doc_id} removed={removed}")
   175|        return removed
   176|
   177|    # ── Query ──────────────────────────────────────────────────────────────
   178|
   179|    def query(self, question: str, intent: Optional[dict] = None) -> str:
   180|        """
   181|        Answer a question using the translator pipeline.
   182|
   183|        Args:
   184|            question: The user's question.
   185|            intent: Optional QueryIntent from classifier.py.
   186|
   187|        Returns:
   188|            Plain text answer (no JSON, no structured output).
   189|        """
   190|        _, fmt_prompt, extract_ans = _get_formatter()
   191|        self._stats["queries_total"] += 1
   192|        t0 = time.time()
   193|
   194|        # Step 1: Expand into sub-queries
   195|        expander = _get_expander(llm=self._llm_light)
   196|        store = self._get_translator_store()
   197|
   198|        # Step 2: Retrieve facts — broad search first, then sub-query expansion
   199|        all_facts = []
   200|
   201|        # First: direct broad search using the full question
   202|        direct_results = self._retrieve_triples(question)
   203|        for r in direct_results:
   204|            if r not in all_facts:
   205|                all_facts.append(r)
   206|
   207|        # Second: expand sub-queries for multi-hop retrieval
   208|        sub_queries = expander.expand(question, store=store, intent=intent)
   209|        if len(sub_queries) > 1:
   210|            self._stats["expansions_mode_simple"] += 1
   211|            for sq in sub_queries:
   212|                retrieved = self._retrieve_triples(sq)
   213|                for r in retrieved:
   214|                    if r not in all_facts:
   215|                        all_facts.append(r)
   216|        else:
   217|            self._stats["expansions_mode_light"] += 1
   218|
   219|        # Step 3: Fallback chain if no triples found
   220|        if not all_facts:
   221|            all_facts = self._fallback(question)
   222|
   223|        # Step 4: Format prompt and call LLM
   224|        prompt = fmt_prompt(all_facts, question)
   225|        try:
   226|            raw_answer = self._answer_llm(prompt)
   227|        except Exception as e:
   228|            logger.error(f"[translator:query] LLM call failed: {e}")
   229|            raw_answer = ""
   230|
   231|        answer = extract_ans(raw_answer)
   232|
   233|        if not answer:
   234|            self._stats["empty_responses"] += 1
   235|
   236|        elapsed = time.time() - t0
   237|        logger.info(
   238|            f"[translator:query] sub_queries={len(sub_queries)} "
   239|            f"facts={len(all_facts)} empty={not answer} latency={elapsed:.1f}s"
   240|        )
   241|        return answer
   242|
   243|    def _retrieve_triples(self, query: str) -> list[str]:
   244|        """Retrieve formatted triples from translator store."""
   245|        store = self._get_translator_store()
   246|        retriever = self._get_translator_retriever()
   247|
   248|        try:
   249|            results = retriever.search(query, category="triple", limit=5)
   250|        except Exception as e:
   251|            logger.warning(f"[translator:retrieve] Search failed: {e}")
   252|            return []
   253|
   254|        return [r.get("content", "") for r in results if r.get("content")]
   255|
   256|    def _fallback(self, question: str) -> list[str]:
   257|        """
   258|        3-level fallback chain:
   259|          Level 1: Triples (already checked, empty)
   260|          Level 2: Raw facts from EtchStore
   261|          Level 3: list_facts(limit=3)
   262|        """
   263|        # Level 2: Raw Etch facts
   264|        if self._etch_retriever:
   265|            self._stats["fallback_raw"] += 1
   266|            try:
   267|                results = self._etch_retriever.search(question, limit=5)
   268|                if results:
   269|                    logger.info(f"[translator:fallback] Level 2 (raw facts): {len(results)} results")
   270|                    return [f"[Raw fact] {r.get('content', '')}" for r in results if r.get("content")]
   271|            except Exception as e:
   272|                logger.warning(f"[translator:fallback] Level 2 failed: {e}")
   273|
   274|        # Level 3: list_facts
   275|        if self._etch_store:
   276|            self._stats["fallback_list"] += 1
   277|            try:
   278|                results = self._etch_store.list_facts(limit=3, min_trust=0.3)
   279|                if results:
   280|                    logger.info(f"[translator:fallback] Level 3 (list): {len(results)} facts")
   281|                    return [f"[General memory] {r.get('content', '')}" for r in results if r.get("content")]
   282|            except Exception as e:
   283|                logger.warning(f"[translator:fallback] Level 3 failed: {e}")
   284|
   285|        # Level 4: Truly no facts — return generic context so LLM answers from knowledge
   286|        logger.info("[translator:fallback] Level 4: no facts available, using knowledge")
   287|        return ["(no relevant documents found. Answer from general knowledge if possible)"]
   288|
   289|    # ── LLM calls ──────────────────────────────────────────────────────────
   290|
   291|    def _default_answer_llm(self, prompt: str) -> str:
   292|        """Default LLM for answering. Uses env-configured model."""
   293|        from openai import OpenAI
   294|
   295|        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY", "")
   296|        url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
   297|        model = os.environ.get("TRANSLATOR_ANSWER_MODEL", "minimax-m2.7")
   298|        
   299|        import httpx
   300|        client = OpenAI(
   301|            api_key=key, base_url=url,
   302|            http_client=httpx.Client(timeout=httpx.Timeout(60.0)),
   303|        )
   304|        resp = client.chat.completions.create(
   305|            model=model,
   306|            messages=[{"role": "user", "content": prompt}],
   307|            temperature=0.0,
   308|            max_tokens=256,
   309|        )
   310|        return resp.choices[0].message.content or ""
   311|
   312|    def _llm_light(self, prompt: str) -> str:
   313|        """Ultra-cheap LLM for query decomposition. Uses MiniMax (no JSON)."""
   314|        from openai import OpenAI
   315|
   316|        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY", "")
   317|        url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
   318|        
   319|        import httpx
   320|        client = OpenAI(
   321|            api_key=key, base_url=url,
   322|            http_client=httpx.Client(timeout=httpx.Timeout(60.0)),
   323|        )
   324|        resp = client.chat.completions.create(
   325|            model="minimax-m2.7",
   326|            messages=[{"role": "user", "content": prompt}],
   327|            temperature=0.0,
   328|            max_tokens=256,
   329|        )
   330|        return resp.choices[0].message.content or ""
   331|
   332|    # ── Store helpers ──────────────────────────────────────────────────────
   333|
   334|    def _get_translator_store(self):
   335|        """Lazy-init translator DB store."""
   336|        if self._translator_store is not None:
   337|            return self._translator_store
   338|
   339|        db_path = self._translator_db_path or os.path.join(
   340|            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
   341|            "translator.db",
   342|        )
   343|
   344|        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
   345|        from store import EtchStore
   346|
   347|        self._translator_store = EtchStore(db_path=db_path)
   348|        return self._translator_store
   349|
   350|    def _get_translator_retriever(self):
   351|        """Lazy-init translator DB retriever."""
   352|        if self._translator_retriever is not None:
   353|            return self._translator_retriever
   354|
   355|        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
   356|        from retrieval import EtchRetriever
   357|
   358|        self._translator_retriever = EtchRetriever(self._get_translator_store())
   359|        return self._translator_retriever
   360|
   361|    # ── Stats ──────────────────────────────────────────────────────────────
   362|
   363|    def get_stats(self) -> dict:
   364|        """Return current pipeline statistics."""
   365|        return dict(self._stats)
   366|
   367|    def reset_stats(self):
   368|        """Reset all statistics counters."""
   369|        for k in self._stats:
   370|            self._stats[k] = 0
   371|