     1|"""
     2|Memory Unit Formatter + Response Formatter for Universal Translator.
     3|
     4|MemoryUnitFormatter: converts (subject, predicate, object) to canonical FTS5 string.
     5|ResponseFormatter: generates LLM prompt and extracts answer.
     6|"""
     7|
     8|import re
     9|from typing import Optional
    10|from .normalizer import normalize_value
    11|
    12|# ── MemoryUnitFormatter ────────────────────────────────────────────────────
    13|
    14|_SEPARATOR = " | "
    15|
    16|
    17|def format_triple(subject: str, predicate: str, obj: str) -> str:
    18|    """Convert (s, p, o) to canonical FTS5 string."""
    19|    return _SEPARATOR.join([subject.strip(), predicate.strip(), obj.strip()])
    20|
    21|
    22|def parse_triple(content: str) -> Optional[tuple[str, str, str]]:
    23|    """Parse a canonical triple string back to (s, p, o). Returns None if invalid."""
    24|    parts = content.split(_SEPARATOR)
    25|    if len(parts) == 3:
    26|        s, p, o = [x.strip() for x in parts]
    27|        if s and p and o:
    28|            return (s, p, o)
    29|    return None
    30|
    31|
    32|def normalize_triple(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    33|    """Normalize each component of a triple."""
    34|    return (
    35|        subject.strip(),
    36|        predicate.strip().lower(),
    37|        normalize_value(obj),
    38|    )
    39|
    40|
    41|# ── ResponseFormatter ──────────────────────────────────────────────────────
    42|
    43|_ANSWER_PROMPT_TEMPLATE = """\
    44|Facts:
    45|{facts}
    46|
    47|Question: {question}
    48|
    49|Answer this question concisely using ONLY the facts above. Respond with a single phrase or number — no explanation, no commentary, no full sentences unless required by the answer itself (e.g. a person's full name)."""
    50|
    51|
    52|def format_answer_prompt(facts: list[str], question: str) -> str:
    53|    """Build a plain-text prompt for any LLM. No JSON, no structured output."""
    54|    if facts:
    55|        # Convert triple format "X | Y | Z" to natural language "X Y Z"
    56|        # MiniMax is confused by pipe symbols in facts
    57|        clean_facts = []
    58|        for f in facts:
    59|            if " | " in f:
    60|                parts = f.split(" | ")
    61|                if len(parts) == 3:
    62|                    s, p, o = [p.strip() for p in parts]
    63|                    # Map known predicates to natural language
    64|                    p_natural = p
    65|                    if p in ("was", "is", "are"):
    66|                        clean_facts.append(f"{s} {p} {o}")
    67|                    elif p.startswith("was "):
    68|                        clean_facts.append(f"{s} {p} {o}")
    69|                    elif p == "contains":
    70|                        clean_facts.append(f"{o}")
    71|                    else:
    72|                        clean_facts.append(f"{s} {p} {o}")
    73|                else:
    74|                    clean_facts.append(f)
    75|            else:
    76|                clean_facts.append(f)
    77|        facts_text = "\n".join(f"- {f}" for f in clean_facts)
    78|    else:
    79|        facts_text = "(no specific facts available)"
    80|    return f"Facts:\n{facts_text}\n\nQuestion: {question}\n\nAnswer:"
    81|
    82|
    83|def extract_answer(llm_output: str) -> str:
    84|    """Extract answer from raw LLM output, handling verbose narratives."""
    85|    if not llm_output:
    86|        return ""
    87|
    88|    text = llm_output.strip()
    89|    if not text:
    90|        return ""
    91|
    92|    # Clean smart quotes / dashes
    93|    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'") \
    94|               .replace('\u2019', "'").replace('\u2013', "-").replace('\u2014', "-")
    95|
    96|    # ── Step 0: Handle newline-separated answers ──
    97|    # When LLM outputs "James Madison\nBecause reasons...", first line IS the answer
    98|    lines = [l.strip() for l in text.split("\n") if l.strip()]
    99|    if len(lines) > 1:
   100|        first = lines[0]
   101|        # If first line looks like an answer (named entity, year, short)
   102|        if (re.match(r'^[A-Z][a-z]*(?:\s+[A-Z][a-zA-Z0-9]*)*$', first) or
   103|            re.match(r'^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$', first) or
   104|            re.match(r'^\d{4}$', first)):
   105|            return first
   106|        # If first line has no preamble words, it's likely the answer
   107|        if len(first) <= 60 and not re.match(
   108|            r'^(?:the|based|according|in\s+response|to\s+answer|here|i\s+would|so|answer)',
   109|            first, re.IGNORECASE
   110|        ):
   111|            return first.rstrip(".,;:!?\"'")
   112|
   113|    # ── Step 1: Remove preamble from the beginning ──
   114|    # Apply multiple rounds to strip nested preambles
   115|    for _ in range(5):
   116|        prev = text
   117|        text = re.sub(
   118|            r'^(?:the\s+)?(?:user|question)\s+(?:asks|wants\s+to\s+know|wonders|provides\s+facts|is\s+asking\s+about)[^:;]*?(?:[:;])\s+',
   119|            '', text, flags=re.IGNORECASE
   120|        )
   121|        text = re.sub(
   122|            r'^(?:based\s+on\s+(?:the\s+)?(?:facts|context|information|above)|according\s+to\s+(?:the\s+)?(?:facts|context))[^:;]*?(?:[:;])\s+',
   123|            '', text, flags=re.IGNORECASE
   124|        )
   125|        text = re.sub(
   126|            r'^(?:in\s+(?:response|answer)\s+to\s+(?:your\s+)?(?:question|query)|to\s+answer\s+(?:your\s+)?(?:question|query)|the\s+(?:answer|response)\s+(?:to\s+your\s+(?:question|query)\s+)?is)\s*[:;,.]*\s+',
   127|            '', text, flags=re.IGNORECASE
   128|        )
   129|        text = re.sub(r'^answer\s*:\s*', '', text, flags=re.IGNORECASE)
   130|        text = text.strip()
   131|        if len(text) == len(prev):
   132|            break
   133|
   134|    # ── Step 2: Remove preamble "the answer is" / "the answer was" at text level ──
   135|    text = re.sub(r'^the\s+answer\s+(?:is|was)\s+', '', text, flags=re.IGNORECASE).strip()
   136|    text = re.sub(r'^based\s+on\s+(?:the\s+)?(?:facts|context|information|above)[,:;.\-–—]*\s+', '', text, flags=re.IGNORECASE).strip()
   137|
   138|    # ── Step 3: Remove quoted embedded questions ──
   139|    text = re.sub(r'\s*["""][^"”“\n]+["""]\s*', ' ', text).strip()
   140|    text = re.sub(r'\s+', ' ', text).strip()
   141|
   142|    # ── Step 3: Try concise answer patterns (whole text) ──
   143|    # If text is a single named entity, number, or date
   144|    if re.match(r'^[A-Z][a-z]*(?:\s[A-Z][a-zA-Z0-9]*)*$', text):
   145|        return text
   146|    if re.match(r'^\d{4}$', text):
   147|        return text
   148|    if re.match(r'^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$', text) or re.match(r'^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}$', text):
   149|        return text
   150|
   151|    # ── Step 4: Sentence-level extraction ──
   152|    # Process sentences in order (first sentence usually contains the answer)
   153|    sentences = re.split(r'(?<=[.!?])\s+', text)
   154|    for sentence in sentences:
   155|        s = sentence.strip()
   156|        if not s or len(s) < 5:
   157|            continue
   158|
   159|        # Strip "that would be" / "the answer is" from sentence start
   160|        s = re.sub(
   161|            r'^(?:that\s+(?:would\s+be|is)\s+|the\s+(?:answer|response)\s+(?:is|would\s+be)\s+)',
   162|            '', s, flags=re.IGNORECASE
   163|        ).strip()
   164|
   165|        # Clean trailing punctuation + "and more"
   166|        s = re.sub(r'\s*(?:,?\s+and\s+more[.!]*)$', '', s, flags=re.IGNORECASE).strip()
   167|
   168|        # Try answer at end patterns
   169|        # "launched by Titan IIIE" / "by Titan IIIE"
   170|        m = re.search(r'(?:by|called|named)\s+((?:[A-Z][a-z]+(?:\s[A-Z0-9]+)*))\s*$', s)
   171|        if m:
   172|            return m.group(1).strip(".,;:!?\"")
   173|
   174|        # "was Titan IIIE" / "is James Madison" / "were Voyager 2"
   175|        m = re.search(r'(?:was|were|is|are)\s+((?:[A-Z][a-z]+(?:\s[A-Z0-9]+)*))\s*$', s)
   176|        if m:
   177|            return m.group(1).strip(".,;:!?\"")
   178|
   179|        # "in 1999" / "on April 30, 1789" / "on 20 June 1837"
   180|        m = re.search(r'(?:in|on|at)\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\s*$', s)
   181|        if m:
   182|            return m.group(1).strip(".,;:!?\"")
   183|
   184|        # "entered office on April 30, 1789" → extract date
   185|        m = re.search(r'(?:on|in|at)\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', s)
   186|        if m:
   187|            return m.group(1).strip(".,;:!?\"")
   188|
   189|        # First named entity in the sentence (accepts acronyms: "Titan IIIE")
   190|        m = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-zA-Z0-9]+)+)', s)
   191|        if m:
   192|            return m.group(1).strip(".,;:!?\"")
   193|
   194|    # ── Step 5: Broad search across whole text ──
   195|    for pat in [
   196|        r'(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
   197|        r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
   198|        r'([A-Z][a-z]+(?:\s[A-Z][a-zA-Z0-9]+)+)',
   199|        r'(\d{4})',
   200|    ]:
   201|        m = re.search(pat, text)
   202|        if m:
   203|            return m.group(1).strip(".,;:!?\"")
   204|
   205|    # ── Last resort ──
   206|    first_line = text.split("\n")[0].strip()
   207|    return first_line[:100].strip(".,;!?\"'")
   208|