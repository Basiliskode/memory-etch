     1|"""
     2|TripleExtractor for Universal Translator.
     3|Extracts (subject | predicate | object) triples from raw text using an LLM.
     4|Runs OFFLINE — once per document, model-agnostic.
     5|"""
     6|
     7|import logging
     8|import os
     9|import re
    10|from typing import Optional
    11|
    12|logger = logging.getLogger(__name__)
    13|
    14|_DEFAULT_MODEL = os.environ.get("TRANSLATOR_EXTRACTOR_MODEL", "minimax-m2.7")
    15|
    16|_EXTRACTOR_PROMPT = """\
    17|Extract facts from the following text.
    18|
    19|Return each fact on its own line using this format:
    20|Subject | Predicate | Object
    21|
    22|Rules:
    23|- subject = the main entity (person, organization, place, thing)
    24|- predicate = the relationship or action
    25|- object = the value or related entity
    26|- Use EXACT names from the text
    27|- Skip opinions, speculation, and meta-commentary
    28|
    29|Text:
    30|{text}
    31|
    32|Extracted facts:"""
    33|
    34|
    35|class TripleExtractor:
    36|    """Offline triple extractor. Uses an LLM to parse raw text into (s, p, o)."""
    37|
    38|    def __init__(
    39|        self,
    40|        model: str = _DEFAULT_MODEL,
    41|        api_key: Optional[str] = None,
    42|        base_url: Optional[str] = None,
    43|    ):
    44|        self.model = model
    45|        self.api_key = api_key
    46|        self.base_url = base_url
    47|
    48|    def extract(self, text: str) -> list[dict]:
    49|        """Extract triples from raw text. Returns list of {'subject', 'predicate', 'object'}."""
    50|        if not text or not text.strip():
    51|            logger.warning("[translator:extract] Empty text, skipping")
    52|            return []
    53|
    54|        # For long text, chunk it
    55|        if len(text) > 8000:
    56|            chunks = self._chunk_text(text, 8000)
    57|            all_triples = []
    58|            for chunk in chunks:
    59|                all_triples.extend(self._extract_chunk(chunk))
    60|            return all_triples
    61|
    62|        return self._extract_chunk(text)
    63|
    64|    def _extract_chunk(self, text: str) -> list[dict]:
    65|        """Extract triples from a single chunk of text."""
    66|        prompt = _EXTRACTOR_PROMPT.format(text=text)
    67|
    68|        try:
    69|            raw = self._query_llm(prompt)
    70|        except Exception as e:
    71|            logger.error(f"[translator:extract] LLM query failed: {e}")
    72|            return self._fallback(text)
    73|
    74|        triples = self._parse_response(raw, text)
    75|        if not triples:
    76|            logger.warning("[translator:extract] No triples parsed from LLM response")
    77|            return self._fallback(text)
    78|
    79|        logger.info(f"[translator:extract] Extracted {len(triples)} triples")
    80|        return triples
    81|
    82|    def _parse_response(self, raw: str, original_text: str) -> list[dict]:
    83|        """Parse LLM response lines into triple dicts.
    84|        Supports both pipe-delimited and narrative formats."""
    85|        triples = []
    86|        for line in raw.strip().split("\n"):
    87|            line = line.strip()
    88|            if not line or line.startswith(("```", "Triples:", "#", "//", "Fact")):
    89|                continue
    90|            # Strip surrounding parentheses the model sometimes adds
    91|            line = line.strip("()（）")
    92|            # Parse "subject | predicate | object"
    93|            parts = [p.strip() for p in line.split("|")]
    94|            if len(parts) >= 3 and all(parts[:3]):
    95|                triples.append({
    96|                    "subject": parts[0],
    97|                    "predicate": parts[1].lower(),
    98|                    "object": " | ".join(p.strip() for p in parts[2:]),
    99|                })
   100|            else:
   101|                # Try narrative format: "Citibank was founded in 1812"
   102|                narrative = self._parse_narrative(line)
   103|                if narrative:
   104|                    triples.append(narrative)
   105|        return triples
   106|
   107|    def _parse_narrative(self, line: str) -> dict | None:
   108|        """Try to extract a triple from narrative text without pipes."""
   109|        # Pattern: "X was Y" / "X was Z of Y" / "X is Y"
   110|        # e.g. "Citibank was founded in 1812" → (Citibank | founded in | 1812)
   111|        patterns = [
   112|            (r"^(.+?)\s+is\s+a\s+(.+?)(?:\s+of\s+(.+?))?$",
   113|             lambda m: (m.group(1), f"is a {m.group(2)}" if not m.group(3) else f"is a {m.group(2)} of", m.group(3) or "")),
   114|            (r"^(.+?)\s+is\s+(?:the\s+)?(.+?)(?:\s+of\s+(.+?))?$",
   115|             lambda m: (m.group(1), f"is the {m.group(2)}" if not m.group(3) else f"is the {m.group(2)} of", m.group(3) or "")),
   116|            (r"^(.+?)\s+was\s+(?:a\s+)?(.+?)\s+of\s+(.+)$",
   117|             lambda m: (m.group(1), f"was {m.group(2)} of", m.group(3))),
   118|            (r"^(.+?)\s+was\s+(?:a\s+)?(.+?)(?:\s+and\s.*)?$",
   119|             lambda m: (m.group(1), "was", m.group(2))),
   120|            (r"^(.+?)\s+was\s+(founded|established|created|born|elected)\s+(?:in\s+|on\s+|at\s+)?(.+)$",
   121|             lambda m: (m.group(1), m.group(2), m.group(3))),
   122|            (r"^(.+?)\s+served\s+as\s+(.+?)(?:\s+from\s+(.+?))?$",
   123|             lambda m: (m.group(1), "served as", m.group(2) + (" from " + m.group(3) if m.group(3) else ""))),
   124|        ]
   125|        for pat, builder in patterns:
   126|            m = re.match(pat, line, re.IGNORECASE)
   127|            if m:
   128|                parts = builder(m)
   129|                if all(parts):
   130|                    return {"subject": parts[0].strip(), "predicate": parts[1].strip().lower(), "object": parts[2].strip()}
   131|        return None
   132|
   133|    def _fallback(self, text: str) -> list[dict]:
   134|        """Fallback when LLM extraction fails: split into sentence-level triples."""
   135|        logger.info("[translator:extract] Using sentence-split fallback")
   136|        triples = []
   137|        # Split into sentences
   138|        sentences = re.split(r'(?<=[.!?])\s+', text[:2000].replace("\n", " "))
   139|        for sent in sentences[:10]:
   140|            sent = sent.strip()
   141|            if not sent or len(sent) < 15:
   142|                continue
   143|            # Try narrative patterns first
   144|            narrative = self._parse_narrative(sent)
   145|            if narrative:
   146|                triples.append(narrative)
   147|                continue
   148|            # Extract capitalized entity as subject
   149|            entities = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', sent)
   150|            if entities:
   151|                subject = entities[0]
   152|                # Everything after the entity minus known verbs
   153|                rest = sent[sent.index(subject) + len(subject):].strip()
   154|                if rest.startswith((" is", " was", " has", " had", " contains", " includes")):
   155|                    verb = rest.split()[0]
   156|                    obj = rest[len(verb):].strip(" ,.")
   157|                    triples.append({"subject": subject, "predicate": verb.strip().lower(), "object": obj})
   158|            if len(triples) >= 5:
   159|                break
   160|        if not triples:
   161|            # Ultimate fallback: generic triple
   162|            snippet = text[:200].replace("\n", " ").strip()
   163|            triples.append({"subject": "document", "predicate": "contains", "object": snippet})
   164|        return triples
   165|
   166|    def _query_llm(self, prompt: str) -> str:
   167|        """Query the LLM. Falls back through available providers."""
   168|        from openai import OpenAI
   169|
   170|        # Try OpenCode Go first, then explicit config
   171|        key = self.api_key or os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
   172|        url = self.base_url or os.environ.get("OPENCODE_GO_BASE_URL") or os.environ.get("OPENAI_BASE_URL", None)
   173|
   174|        if "OPENCODE_GO_API_KEY" in os.environ and not self.base_url:
   175|            url = "https://opencode.ai/zen/go/v1"
   176|
   177|        if not key:
   178|            raise ValueError("No API key found for TripleExtractor. Set OPENCODE_GO_API_KEY or TRANSLATOR_EXTRACTOR_KEY.")
   179|
   180|        import httpx
   181|        client = OpenAI(
   182|            api_key=key, base_url=url,
   183|            http_client=httpx.Client(timeout=httpx.Timeout(120.0)),
   184|        )
   185|
   186|        resp = client.chat.completions.create(
   187|            model=self.model,
   188|            messages=[{"role": "user", "content": prompt}],
   189|            temperature=0.1,
   190|            max_tokens=4096,
   191|        )
   192|        return resp.choices[0].message.content or ""
   193|
   194|    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
   195|        """Split long text into overlapping chunks at paragraph boundaries."""
   196|        paragraphs = text.split("\n\n")
   197|        chunks = []
   198|        current = ""
   199|        for para in paragraphs:
   200|            if len(current) + len(para) > max_chars and current:
   201|                chunks.append(current)
   202|                current = para
   203|            else:
   204|                current = (current + "\n\n" + para) if current else para
   205|        if current:
   206|            chunks.append(current)
   207|        return chunks
   208|
   209|
   210|def extract(text: str, **kwargs) -> list[dict]:
   211|    """Convenience function."""
   212|    return TripleExtractor(**kwargs).extract(text)
   213|