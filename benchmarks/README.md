# External Benchmarks

Memory Etch benchmark adapters in this directory target recognized external datasets. They are separate from the internal synthetic health checks under `scripts/` and `memory_etch.benchmark`.

## LOCOMO

LOCOMO is the long-term conversational memory benchmark from Snap Research, introduced in `Evaluating Very Long-Term Conversational Memory of LLM Agents` (arXiv:2402.17753). The official repository is `https://github.com/snap-research/locomo`.

### Get The Data

Do not vendor LOCOMO into this repo. Clone or download the official dataset yourself:

```bash
git clone https://github.com/snap-research/locomo.git ../locomo
```

The current official README documents `data/locomo10.json` as the released dataset. Each sample contains `conversation` sessions and annotated `qa` items with `question`, `answer`, `category`, and `evidence` dialog IDs when available.

### Variant Support

Only `etch-noop` (FTS-only) is currently supported. The `etch` (hybrid FTS+HRR) and
`etch-fastembed` (vector) branches are excluded because the embedding-based retrieval
path does not yet filter results by `project`/`source` before RRF fusion. This can
leak retrieved turns across LOCOMO conversations and inflate evidence metrics.

### Run Retrieval Only

This produces retrieval traces and a dry-run prediction marker. It is useful for validating ingestion and retrieval, but it is not a LOCOMO QA score.

```bash
python -m benchmarks.locomo.runner \
  --data-dir ../locomo \
  --output results/locomo-memory-etch.jsonl \
  --top-k 10 \
  --memory-variant etch-noop \
  --seed 42
```

You can also point directly at the official file:

```bash
python -m benchmarks.locomo.runner \
  --input ../locomo/data/locomo10.json \
  --output results/locomo-memory-etch.json
```

### Run With An Answerer

Answer generation is optional and requires your own API key. The runner still does not compute an official judged score; it writes model predictions and retrieval context for later evaluation.

```bash
set OPENAI_API_KEY=...
python -m benchmarks.locomo.runner \
  --data-dir ../locomo \
  --output results/locomo-memory-etch-openai.jsonl \
  --answerer-provider openai \
  --answerer-model gpt-4o-mini \
  --top-k 10
```

### Output

JSONL output starts with a summary row, followed by one row per QA item. JSON output contains `summary` and `results`.

The summary includes retrieval diagnostics such as `evidence_recall_avg`, `any_evidence_hit_rate`, and `all_evidence_hit_rate`. These are evidence-retrieval checks, not LOCOMO QA accuracy.

Each result includes:

- `question_id`, `conversation_id`, `question`
- `category`, `gold_answer`, `adversarial_answer`, `evidence_ids`
- `retrieved_ids`, `retrieved_context`
- `retrieval_latency_ms`
- `prediction`, `prediction_meta`

### Comparability

Public LOCOMO numbers are usually self-run using the official data and scripts rather than taken from a single central leaderboard. Do not report a public score from this adapter unless you also document the full evaluation setup.

Scores can differ because of:

- Answerer model and version
- Judge model and grading prompt
- Retrieval `top-k`
- Whether dialog turns, observations, session summaries, or multimodal captions are indexed
- Preprocessing and evidence handling
- Prompt wording and context ordering

This adapter currently implements reproducible ingestion, retrieval, optional answer generation, and trace output. A comparable public score still needs an explicit judging/evaluation step aligned with the LOCOMO paper or official scripts.
