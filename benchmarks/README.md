# External Benchmarks

Memento benchmark adapters in this directory target recognized external datasets. They are separate from the internal synthetic health checks under `scripts/` and `memento.benchmark`.

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
  --output results/locomo-memento.jsonl \
  --top-k 10 \
  --memory-variant etch-noop \
  --seed 42
```

You can also point directly at the official file:

```bash
python -m benchmarks.locomo.runner \
  --input ../locomo/data/locomo10.json \
  --output results/locomo-memento.json
```

### Run With An Answerer

Answer generation is optional. Use `extractive` for a deterministic local smoke run without API keys; it selects the retrieved turn with the highest lexical overlap and is not an LLM.

```bash
python -m benchmarks.locomo.runner \
  --input ../locomo/data/locomo10.json \
  --output results/locomo-memento-extractive.json \
  --answerer-provider extractive \
  --local-eval \
  --top-k 10 \
  --qa-limit 20
```

For model predictions, configure your own API key. The prompt is context-only: the answerer is instructed to use only retrieved turns, avoid guessing, and answer `No information available` when context is insufficient.

```bash
set OPENAI_API_KEY=...
python -m benchmarks.locomo.runner \
  --data-dir ../locomo \
  --output results/locomo-memento-openai.jsonl \
  --answerer-provider openai \
  --answerer-model gpt-4o-mini \
  --answer-timeout-seconds 60 \
  --local-eval \
  --top-k 10
```

Anthropic uses `ANTHROPIC_API_KEY`:

```bash
set ANTHROPIC_API_KEY=...
python -m benchmarks.locomo.runner \
  --data-dir ../locomo \
  --output results/locomo-memento-anthropic.jsonl \
  --answerer-provider anthropic \
  --answerer-model claude-3-5-haiku-latest \
  --answer-timeout-seconds 60 \
  --local-eval \
  --top-k 10
```

Gemini uses `GEMINI_API_KEY` and the Google Generative Language API. The default
model is `gemini-1.5-flash`; override it with `--answerer-model` if you want a
newer Flash model available to your account.

```bash
set GEMINI_API_KEY=...
python -m benchmarks.locomo.runner \
  --data-dir ../locomo \
  --output results/locomo-memento-gemini.jsonl \
  --answerer-provider gemini \
  --answerer-model gemini-1.5-flash \
  --answer-timeout-seconds 60 \
  --local-eval \
  --top-k 10
```

If a provider key is missing, the runner exits before issuing requests. If an individual provider request errors, the QA row is still written with empty `prediction` and error metadata so partial traces remain inspectable.

### Local QA Diagnostics

Passing `--local-eval` adds `summary.qa_diagnostics` when gold answers are present. These metrics are intentionally named as diagnostics, not official scores:

- `exact-ish_avg`: normalized token-set equality, similar to the official helper's exact-ish check.
- `contains_avg`: whether normalized prediction contains the gold answer, or vice versa.
- `token_f1_avg`: normalized unigram F1.

These diagnostics are useful for smoke tests and regression checks. Do not publish them as LOCOMO leaderboard or paper-comparable scores.

### Official LOCOMO Evaluation

The official clone contains `task_eval/evaluate_qa.py` plus provider-specific scripts under `scripts/`. That script generates model answers from the original LOCOMO format and then computes the repository's F1-style aggregate stats; it does not directly consume Memento JSONL traces without conversion.

To run the official script against its own supported provider flow from the LOCOMO repo:

```bash
cd ../locomo
python task_eval/evaluate_qa.py \
  --data-file data/locomo10.json \
  --out-file outputs/locomo10_qa.json \
  --model gpt-4-turbo \
  --batch-size 20
```

For RAG-style official evaluation, use the official `--use-rag`, `--rag-mode`, `--emb-dir`, `--top-k`, and `--retriever` flags after preparing embeddings in the format LOCOMO expects. A public Memento LOCOMO score should document the exact LOCOMO commit, data file, answerer model/version, judge/evaluation command, RAG conversion if any, `top-k`, seed, and output files.

### Output

JSONL output starts with a summary row, followed by one row per QA item. JSON output contains `summary` and `results`.

The summary includes reproduction metadata (`dataset_path`, `runner_version`, `memory_variant`, `answerer_provider`, `answerer_model`, `answer_timeout_seconds`, `prompt_version`, `seed`, `top_k`) plus retrieval diagnostics such as `evidence_recall_avg`, `any_evidence_hit_rate`, and `all_evidence_hit_rate`. These retrieval metrics are evidence-retrieval checks, not LOCOMO QA accuracy.

Each result includes:

- `question_id`, `conversation_id`, `question`
- `category`, `gold_answer`, `adversarial_answer`, `evidence_ids`
- `retrieved_ids`, `retrieved_context`
- `retrieval_latency_ms`
- `prediction`, `prediction_meta`

Safe-to-publish artifacts usually include summary metrics, commands, code commit, and redacted traces. Review `retrieved_context`, `gold_answer`, and `prediction` before publishing because they contain LOCOMO dataset text and model outputs; do not publish API keys, raw provider errors with secrets, or private local paths if that matters for your release.

### Comparability

Public LOCOMO numbers are usually self-run using the official data and scripts rather than taken from a single central leaderboard. Do not report a public score from this adapter unless you also document the full evaluation setup.

Scores can differ because of:

- Answerer model and version
- Judge model and grading prompt
- Retrieval `top-k`
- Whether dialog turns, observations, session summaries, or multimodal captions are indexed
- Preprocessing and evidence handling
- Prompt wording and context ordering

This adapter currently implements reproducible ingestion, retrieval, optional answer generation, local QA diagnostics, and trace output. A comparable public score still needs an explicit judging/evaluation step aligned with the LOCOMO paper or official scripts.
