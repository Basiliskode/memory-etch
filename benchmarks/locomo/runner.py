"""Run memento retrieval over the external LOCOMO QA benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memento import EtchStore  # noqa: E402
from memento.embedding import NoopProvider  # noqa: E402
from memento.store._schema import _sanitize_fts5  # noqa: E402

from .loader import LOCOMO_SOURCE_URL, LocomoQA, LocomoSample, LocomoTurn, load_locomo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run memento retrieval on official LOCOMO data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-dir",
        help="Path to a cloned/downloaded LOCOMO repo or data dir",
    )
    data_group.add_argument("--input", help="Path to an official LOCOMO JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Result path (.jsonl or .json)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of contexts to retrieve per QA",
    )
    parser.add_argument(
        "--memory-variant",
        choices=["none", "etch-noop"],
        default="etch-noop",
        help=(
            "memento retrieval variant. Currently only 'etch-noop' (FTS-only) is "
            "supported because the embedding vector branch does not yet filter results by "
            "project/source before RRF fusion. This means 'etch-fastembed' can leak "
            "retrieved turns across LOCOMO conversations, inflating evidence metrics."
        ),
    )
    parser.add_argument("--store-path", help="Optional persistent Etch SQLite DB path")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit conversations for smoke runs",
    )
    parser.add_argument("--qa-limit", type=int, default=None, help="Limit QA items after loading")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic ordering",
    )
    parser.add_argument(
        "--answerer-provider",
        choices=["none", "extractive", "openai", "anthropic", "gemini"],
        default="none",
        help=(
            "Optional answer generation provider. 'none' emits a dry-run stub; "
            "'extractive' is deterministic/local for smoke tests."
        ),
    )
    parser.add_argument(
        "--answerer-model",
        default="",
        help="Answerer model name, if provider is set",
    )
    parser.add_argument(
        "--answer-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request timeout for network answerer providers",
    )
    parser.add_argument(
        "--local-eval",
        action="store_true",
        help="Compute local exact-ish/contains/token-F1 diagnostics when gold answers exist.",
    )
    args = parser.parse_args(argv)

    try:
        samples = load_locomo(args.input or args.data_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        print(f"Official data: {LOCOMO_SOURCE_URL} (data/locomo10.json)", file=sys.stderr)
        return 2

    runner = LocomoRunner(
        samples=samples,
        output_path=Path(args.output),
        top_k=args.top_k,
        memory_variant=args.memory_variant,
        answerer_provider=args.answerer_provider,
        answerer_model=args.answerer_model,
        answer_timeout_seconds=args.answer_timeout_seconds,
        local_eval=args.local_eval,
        store_path=Path(args.store_path) if args.store_path else None,
        sample_limit=args.sample_limit,
        qa_limit=args.qa_limit,
        seed=args.seed,
        dataset_path=args.input or args.data_dir,
    )
    try:
        summary = runner.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "LOCOMO run complete: "
        f"{summary['qa_count']} QA, {summary['turn_count']} turns, "
        f"avg retrieval {summary['avg_retrieval_latency_ms']} ms"
    )
    print(f"Results saved to {args.output}")
    return 0


class LocomoRunner:
    def __init__(
        self,
        samples: list[LocomoSample],
        output_path: Path,
        top_k: int = 10,
        memory_variant: str = "etch-noop",
        answerer_provider: str = "none",
        answerer_model: str = "",
        answer_timeout_seconds: float = 60.0,
        local_eval: bool = False,
        store_path: Path | None = None,
        sample_limit: int | None = None,
        qa_limit: int | None = None,
        seed: int = 42,
        dataset_path: str = "",
    ) -> None:
        self.samples = list(samples[:sample_limit] if sample_limit else samples)
        self.output_path = output_path
        self.top_k = top_k
        self.memory_variant = memory_variant
        self.answerer_provider = answerer_provider
        self.answerer_model = answerer_model
        self.answer_timeout_seconds = answer_timeout_seconds
        self.local_eval = local_eval
        self.store_path = store_path
        self.qa_limit = qa_limit
        self.seed = seed
        self.dataset_path = dataset_path
        self._turn_by_internal_id: dict[int, LocomoTurn] = {}

    def run(self) -> dict[str, Any]:
        random.seed(self.seed)
        tmpdir: tempfile.TemporaryDirectory[str] | None = None
        db_path = self.store_path
        if db_path is None:
            tmpdir = tempfile.TemporaryDirectory(prefix="memento-locomo-")
            db_path = Path(tmpdir.name) / "locomo_etch.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        store = self._make_store(db_path)
        try:
            turn_count = self._ingest(store)
            qa_items = [qa for sample in self.samples for qa in sample.qa]
            if self.qa_limit is not None:
                qa_items = qa_items[: self.qa_limit]
            answerer = make_answerer(
                self.answerer_provider, self.answerer_model, self.answer_timeout_seconds
            )

            results = []
            total_qas = len(qa_items)
            for idx, qa in enumerate(qa_items, start=1):
                cat = qa.category or "uncategorized"
                print(f"QA {idx}/{total_qas} [{cat}]: {qa.question[:55]}...", file=sys.stderr)
                results.append(self._run_qa(store, qa, answerer))
            summary = self._summary(turn_count, results)
            self._write_output(summary, results)
            return summary
        finally:
            store.close()
            if tmpdir is not None:
                tmpdir.cleanup()

    def _make_store(self, db_path: Path) -> EtchStore:
        if self.memory_variant == "none":
            return EtchStore(str(db_path), auto_migrate=True, embedding_provider=NoopProvider())
        if self.memory_variant != "etch-noop":
            raise ValueError(f"Unsupported memory variant: {self.memory_variant}")
        return EtchStore(str(db_path), auto_migrate=True, embedding_provider=NoopProvider())

    def _ingest(self, store: EtchStore) -> int:
        count = 0
        for sample in self.samples:
            for turn in sample.turns:
                fact_id = store.add_fact(
                    content=format_turn_content(turn),
                    category="locomo_dialog",
                    tags=format_turn_tags(turn),
                    project=turn.conversation_id,
                    session_id=turn.session_id,
                    source_harness="locomo",
                    source_kind="dialog_turn",
                    where_text=turn.dia_id,
                )
                self._turn_by_internal_id[int(fact_id)] = turn
                count += 1
        flush_hrr(store)
        return count

    def _run_qa(self, store: EtchStore, qa: LocomoQA, answerer: Answerer) -> dict[str, Any]:
        started = time.perf_counter()
        retrieved = self._retrieve_dialog_turns(store, qa)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        contexts = [self._context_item(item) for item in retrieved]
        contexts = self._expand_contexts_with_adjacent(contexts)
        try:
            prediction = answerer.answer(qa.question, contexts)
        except Exception as exc:  # noqa: BLE001 - preserve partial benchmark traces on API errors.
            prediction = {
                "text": "",
                "meta": {
                    "provider": self.answerer_provider,
                    "model": self.answerer_model,
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                },
            }
        return {
            "question_id": qa.question_id,
            "conversation_id": qa.conversation_id,
            "question": qa.question,
            "category": qa.category,
            "gold_answer": qa.answer,
            "adversarial_answer": qa.adversarial_answer,
            "evidence_ids": qa.evidence,
            "retrieved_ids": [item["dia_id"] for item in contexts],
            "retrieved_context": contexts,
            "retrieval_latency_ms": latency_ms,
            "prediction": prediction["text"],
            "prediction_meta": prediction["meta"],
        }

    def _retrieve_dialog_turns(self, store: EtchStore, qa: LocomoQA) -> list[dict[str, Any]]:
        if self.memory_variant == "none":
            return []
        """Retrieve LOCOMO dialog turns without treating questions as strict FTS AND queries.

        ``EtchStore.search()`` is intentionally conservative: after FTS5 sanitising, a natural
        language question such as "When did Caroline go to the LGBTQ support group?" becomes a
        multi-token MATCH expression where every token must match. LOCOMO evidence turns often
        contain the entity and answer but not the question words, so the strict query can return
        no context at all. For the benchmark adapter, fall back to an OR query over meaningful
        question terms while preserving project/source/scope filters.
        """

        exact = store.search(
            qa.question,
            limit=self.top_k,
            project=qa.conversation_id,
            source_harness="locomo",
            source_kind="dialog_turn",
        )
        if len(exact) >= self.top_k:
            return exact[: self.top_k]

        fallback = self._keyword_retrieve_dialog_turns(store, qa)
        by_id: dict[int, dict[str, Any]] = {}
        for item in [*exact, *fallback]:
            fact_id = int(item.get("fact_id") or 0)
            if fact_id and fact_id not in by_id:
                by_id[fact_id] = item
            if len(by_id) >= self.top_k:
                break
        return list(by_id.values())

    def _expand_contexts_with_adjacent(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Expand context items with adjacent dialog turns (±1).

        Retrieval often returns the turn that ASKS a question but not the turn
        that ANSWERS it. This method fills that gap by injecting neighbor turns
        from the same session.
        """
        if not contexts:
            return contexts

        # Collect dia_ids already present to avoid duplicates
        seen_dia: set[str] = {c.get("dia_id", "") for c in contexts}

        # Find the sample(s) involved
        sample_dia_map: dict[str, LocomoTurn] = {}
        for sample in self.samples:
            for turn in sample.turns:
                sample_dia_map[turn.dia_id] = turn

        extra_items: list[dict[str, Any]] = []
        for ctx in contexts:
            dia_id = ctx.get("dia_id", "")
            turn = sample_dia_map.get(dia_id)
            if turn is None:
                continue
            for sample in self.samples:
                for t in sample.turns:
                    if (
                        t.session_index == turn.session_index
                        and abs(t.turn_index - turn.turn_index) == 1
                        and t.dia_id not in seen_dia
                    ):
                        extra_items.append({
                            "internal_fact_id": 0,
                            "dia_id": t.dia_id,
                            "session_id": t.session_id,
                            "turn_id": t.turn_index,
                            "speaker": t.speaker,
                            "timestamp": t.timestamp,
                            "text": t.text,
                            "score": ctx.get("score"),
                            "_adjacent": True,
                        })
                        seen_dia.add(t.dia_id)
                        if len(extra_items) >= self.top_k // 2:
                            break
                if extra_items:
                    break

        if extra_items:
            contexts = contexts + extra_items
        return contexts[: self.top_k * 2]

    def _find_sample(self, qa: LocomoQA) -> LocomoSample | None:
        """Find the LOCOMO sample that contains this QA."""
        for sample in self.samples:
            for sqa in sample.qa:
                if sqa.question_id == qa.question_id:
                    return sample
        return None

    def _keyword_retrieve_dialog_turns(
        self, store: EtchStore, qa: LocomoQA
    ) -> list[dict[str, Any]]:
        terms = significant_question_terms(qa.question)
        if not terms:
            return []

        fts_query = " OR ".join(
            _sanitize_fts5(term) for term in terms if _sanitize_fts5(term)
        )
        if not fts_query:
            return []
        try:
            with store._lock:
                rows = store._conn.execute(
                    """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                              f.created_at, f.updated_at, f.project, f.topic_key,
                              f.revision_count, f.importance, f.session_id,
                              f.source_harness, f.source_agent, f.source_kind, f.scope,
                              f.fact_type
                       FROM facts f
                       JOIN facts_fts fts ON fts.rowid = f.fact_id
                       WHERE facts_fts MATCH ?
                         AND (f.deleted IS NULL OR f.deleted = 0)
                         AND f.project = ?
                         AND f.scope = 'canonical'
                         AND f.source_harness = 'locomo'
                         AND f.source_kind = 'dialog_turn'
                       ORDER BY fts.rank
                       LIMIT ?""",
                    (fts_query, qa.conversation_id, self.top_k * 3),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError:
            return []

    def _context_item(self, item: dict[str, Any]) -> dict[str, Any]:
        internal_id = int(item.get("fact_id") or 0)
        turn = self._turn_by_internal_id.get(internal_id)
        if turn is None:
            return {
                "internal_fact_id": internal_id,
                "dia_id": "",
                "session_id": item.get("session_id", ""),
                "speaker": "",
                "timestamp": "",
                "text": item.get("content", ""),
                "score": item.get("score"),
            }
        return {
            "internal_fact_id": internal_id,
            "dia_id": turn.dia_id,
            "session_id": turn.session_id,
            "turn_id": turn.turn_index,
            "speaker": turn.speaker,
            "timestamp": turn.timestamp,
            "text": turn.text,
            "score": item.get("score"),
        }

    def _summary(self, turn_count: int, results: list[dict[str, Any]]) -> dict[str, Any]:
        avg_latency = 0.0
        if results:
            avg_latency = sum(item["retrieval_latency_ms"] for item in results) / len(results)
        retrieval_metrics = evidence_retrieval_metrics(results)
        summary = {
            "benchmark": "locomo",
            "benchmark_source": LOCOMO_SOURCE_URL,
            "dataset_path": self.dataset_path,
            "runner_version": "locomo-answerer-v1",
            "memory_variant": self.memory_variant,
            "answerer_provider": self.answerer_provider,
            "answerer_model": self.answerer_model,
            "answer_timeout_seconds": self.answer_timeout_seconds,
            "prompt_version": ANSWER_PROMPT_VERSION,
            "dry_run": self.answerer_provider == "none",
            "seed": self.seed,
            "top_k": self.top_k,
            "sample_count": len(self.samples),
            "turn_count": turn_count,
            "qa_count": len(results),
            "avg_retrieval_latency_ms": round(avg_latency, 3),
            "retrieval_metrics": retrieval_metrics,
            "score": None,
            "score_note": (
                "No official LOCOMO judge score is computed by this runner. "
                "Use qa_diagnostics only as local exact-ish/contains/token-F1 diagnostics."
            ),
        }
        if self.local_eval:
            summary["qa_diagnostics"] = evaluate_qa_diagnostics(results)
        return summary

    def _write_output(self, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.suffix.lower() == ".jsonl":
            with self.output_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
                for result in results:
                    fh.write(json.dumps({"type": "result", **result}, ensure_ascii=False) + "\n")
            return
        self.output_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def format_turn_content(turn: LocomoTurn) -> str:
    parts = [f"[{turn.dia_id}] {turn.speaker}: {turn.text}"]
    if turn.timestamp:
        parts.append(f"Session timestamp: {turn.timestamp}")
    if turn.metadata.get("blip_caption"):
        parts.append(f"Image caption: {turn.metadata['blip_caption']}")
    return "\n".join(parts)


def format_turn_tags(turn: LocomoTurn) -> str:
    safe_speaker = turn.speaker.replace(",", " ")
    return ",".join(
        [
            "benchmark:locomo",
            f"dia_id:{turn.dia_id}",
            f"session:{turn.session_id}",
            f"turn:{turn.turn_index}",
            f"speaker:{safe_speaker}",
        ]
    )


LOCOMO_QUERY_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}


def significant_question_terms(question: str) -> list[str]:
    """Return stable LOCOMO query terms suitable for broad FTS retrieval."""

    cleaned = _sanitize_fts5(question)
    terms: list[str] = []
    seen: set[str] = set()
    for raw in cleaned.split():
        term = re.sub(r"[^A-Za-z0-9_]", " ", raw).strip()
        if " " in term:
            continue
        lowered = term.lower()
        if len(lowered) < 3 or lowered in LOCOMO_QUERY_STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(term)
    return terms


def evidence_retrieval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute retrieval diagnostics without presenting them as QA accuracy."""

    with_evidence = [item for item in results if item.get("evidence_ids")]
    if not with_evidence:
        return {
            "qa_with_evidence": 0,
            "empty_retrieval_count": 0,
            "evidence_recall_avg": 0.0,
            "any_evidence_hit_rate": 0.0,
            "all_evidence_hit_rate": 0.0,
        }

    empty_retrieval_count = 0
    any_hits = 0
    all_hits = 0
    recall_sum = 0.0
    for item in with_evidence:
        evidence = set(item.get("evidence_ids") or [])
        retrieved = set(item.get("retrieved_ids") or [])
        if not retrieved:
            empty_retrieval_count += 1
        intersection = evidence & retrieved
        if intersection:
            any_hits += 1
        if evidence <= retrieved:
            all_hits += 1
        recall_sum += len(intersection) / len(evidence)

    total = len(with_evidence)
    return {
        "qa_with_evidence": total,
        "empty_retrieval_count": empty_retrieval_count,
        "evidence_recall_avg": round(recall_sum / total, 4),
        "any_evidence_hit_rate": round(any_hits / total, 4),
        "all_evidence_hit_rate": round(all_hits / total, 4),
    }


def flush_hrr(store: EtchStore) -> None:
    try:
        store._hrr_flush_signal.set()
        thread = getattr(store, "_hrr_flush_thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=5)
    except Exception:
        pass


class Answerer:
    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class DryRunAnswerer(Answerer):
    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        del question
        return {
            "text": "DRY_RUN_NO_ANSWERER_CONFIGURED",
            "meta": {
                "dry_run": True,
                "note": "Retrieval context was produced, but no answerer model was configured.",
                "context_count": len(contexts),
            },
        }


class ExtractiveAnswerer(Answerer):
    """Deterministic local answerer for smoke tests and no-key benchmark plumbing."""

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        question_terms = set(significant_question_terms(question))
        best_context: dict[str, Any] | None = None
        best_score = -1
        for context in contexts:
            text_terms = {
                term.lower() for term in significant_question_terms(context.get("text", ""))
            }
            score = len({term.lower() for term in question_terms} & text_terms)
            if score > best_score:
                best_context = context
                best_score = score

        if best_context is None:
            text = "No information available"
            source_dia_id = ""
        else:
            text = str(best_context.get("text") or "No information available")
            source_dia_id = str(best_context.get("dia_id") or "")
        return {
            "text": text,
            "meta": {
                "provider": "extractive",
                "model": "deterministic-overlap-v1",
                "prompt_version": ANSWER_PROMPT_VERSION,
                "context_count": len(contexts),
                "source_dia_id": source_dia_id,
                "note": "Selects the retrieved turn with highest lexical overlap; not an LLM.",
            },
        }


class OpenAIAnswerer(Answerer):
    def __init__(self, model: str, timeout_seconds: float) -> None:
        from openai import OpenAI

        self.client = OpenAI(timeout=timeout_seconds)
        self.model = model or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_answer_prompt(question, contexts)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "Answer LOCOMO questions using only the supplied context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=256,
        )
        return {
            "text": response.output_text.strip(),
            "meta": {
                "provider": "openai",
                "model": self.model,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "context_count": len(contexts),
                "context_ids": [item.get("dia_id") for item in contexts],
                "timeout_seconds": self.timeout_seconds,
            },
        }


class AnthropicAnswerer(Answerer):
    def __init__(self, model: str, timeout_seconds: float) -> None:
        import anthropic

        self.client = anthropic.Anthropic(timeout=timeout_seconds)
        self.model = model or "claude-3-5-haiku-latest"
        self.timeout_seconds = timeout_seconds

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_answer_prompt(question, contexts)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system="Answer LOCOMO questions using only the supplied context.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return {
            "text": text.strip(),
            "meta": {
                "provider": "anthropic",
                "model": self.model,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "context_count": len(contexts),
                "context_ids": [item.get("dia_id") for item in contexts],
                "timeout_seconds": self.timeout_seconds,
            },
        }


class GeminiAnswerer(Answerer):
    def __init__(self, model: str, timeout_seconds: float) -> None:
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = model or "gemini-2.5-flash"
        self.timeout_seconds = timeout_seconds

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        payload = build_gemini_payload(question, contexts)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.reason
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
                message = error_payload.get("error", {}).get("message", message)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            raise RuntimeError(f"Gemini request failed: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc

        text = extract_gemini_text(response_payload)
        return {
            "text": text.strip(),
            "meta": {
                "provider": "gemini",
                "model": self.model,
                "prompt_version": ANSWER_PROMPT_VERSION,
                "context_count": len(contexts),
                "context_ids": [item.get("dia_id") for item in contexts],
                "timeout_seconds": self.timeout_seconds,
            },
        }


def make_answerer(provider: str, model: str, timeout_seconds: float = 60.0) -> Answerer:
    if provider == "none":
        return DryRunAnswerer()
    if provider == "extractive":
        return ExtractiveAnswerer()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --answerer-provider openai")
        return OpenAIAnswerer(model, timeout_seconds)
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --answerer-provider anthropic")
        return AnthropicAnswerer(model, timeout_seconds)
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for --answerer-provider gemini")
        return GeminiAnswerer(model, timeout_seconds)
    raise ValueError(f"Unsupported answerer provider: {provider}")


ANSWER_PROMPT_VERSION = "locomo-context-v2"


def build_answer_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        "\n".join(
            [
                f"Context ID: {item.get('dia_id', '')}",
                f"Session: {item.get('session_id', '')}",
                f"Timestamp: {item.get('timestamp', '')}",
                f"Speaker: {item.get('speaker', '')}",
                f"Text: {item.get('text', '')}",
            ]
        )
        for item in contexts
    )
    return (
        "You are answering a LOCOMO long-term conversational-memory question.\n"
        "Rules:\n"
        "- Use only the retrieved context below.\n"
        "- Do not use outside knowledge or guess.\n"
        "- If the context uses relative dates (yesterday, last week, next month)\n"
        "  and a Session timestamp is provided, resolve to absolute dates.\n"
        "  Example: 'yesterday' with timestamp '8 May 2023' → '7 May 2023'.\n"
        "- Prefer a short, precise answer copied from the context when possible.\n"
        "- If the context does not contain the answer, reply exactly: "
        "No information available\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context_text or '[no retrieved context]'}\n\n"
        "Short answer:"
    )


def build_gemini_payload(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "systemInstruction": {
            "parts": [{"text": (
                "Answer LOCOMO questions using only the supplied context. "
                "If the context uses relative dates (yesterday, last week, next month) "
                "and a Session timestamp is provided, resolve to absolute dates."
            )}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_answer_prompt(question, contexts)}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 256,
        },
    }


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    parts = (
        response_payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    return "".join(str(part.get("text", "")) for part in parts)


def evaluate_qa_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute local QA diagnostics; this is not the official LOCOMO judge protocol."""

    scored: list[dict[str, float | int | str]] = []
    for result in results:
        gold = result.get("gold_answer")
        if gold is None or gold == "":
            continue
        prediction = str(result.get("prediction") or "")
        gold_text = str(gold)
        scored.append(
            {
                "question_id": str(result.get("question_id", "")),
                "exact-ish": float(exact_ish_match(prediction, gold_text)),
                "contains": float(contains_match(prediction, gold_text)),
                "token_f1": round(token_f1(prediction, gold_text), 4),
            }
        )

    if not scored:
        return {
            "metric_set": "local_qa_diagnostics_v1",
            "score_note": "No gold answers were available for local diagnostics.",
            "qa_with_gold": 0,
            "exact-ish_avg": 0.0,
            "contains_avg": 0.0,
            "token_f1_avg": 0.0,
            "per_question": [],
        }

    total = len(scored)
    return {
        "metric_set": "local_qa_diagnostics_v1",
        "score_note": (
            "Local exact-ish/contains/token-F1 diagnostics; not official LOCOMO LLM judge."
        ),
        "qa_with_gold": total,
        "exact-ish_avg": round(sum(item["exact-ish"] for item in scored) / total, 4),
        "contains_avg": round(sum(item["contains"] for item in scored) / total, 4),
        "token_f1_avg": round(sum(item["token_f1"] for item in scored) / total, 4),
        "per_question": scored,
    }


def normalize_answer(text: str) -> str:
    text = text.lower().replace(",", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def exact_ish_match(prediction: str, gold: str) -> bool:
    return set(normalize_answer(prediction).split()) == set(normalize_answer(gold).split())


def contains_match(prediction: str, gold: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    return bool(
        normalized_gold and normalized_prediction
        and (
            normalized_gold in normalized_prediction
            or normalized_prediction in normalized_gold
        )
    )


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


if __name__ == "__main__":
    raise SystemExit(main())
