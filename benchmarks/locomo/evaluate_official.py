"""Adapter: run official LOCOMO evaluation metrics on memento benchmark results.

Usage:
    py -m benchmarks.locomo.evaluate_official \\
        --input results/locomo-memento-gemini.json \\
        --output results/locomo-memento-official-eval.json

This is an offline evaluation of already-generated predictions.
No API keys are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LOCOMO_EVAL_DIR = Path(r"F:\OPENCODE proyectos\locomo\task_eval").resolve()
if str(LOCOMO_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_EVAL_DIR))

from evaluation import bert_score as locomo_bert_score  # noqa: E402
from evaluation import eval_question_answering  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run official LOCOMO evaluation on memento benchmark outputs."
    )
    parser.add_argument("--input", required=True, help="Results JSON from LOCOMO runner")
    parser.add_argument("--output", required=True, help="Path for official eval results JSON")
    parser.add_argument(
        "--eval-key",
        default="prediction",
        help="Field name containing the prediction text",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation (useful on first run before model download)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    with input_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    results: list[dict[str, Any]] = payload.get("results", [])
    if not results:
        print(f"No results found in {input_path}", file=sys.stderr)
        return 2

    qas = _map_to_official_format(results, args.eval_key)

    print(
        f"Running official LOCOMO eval on {len(qas)} QA items "
        f"(cats: {_category_counts(qas)})...",
        file=sys.stderr,
    )

    all_f1, _lengths, all_recall = eval_question_answering(qas, eval_key=args.eval_key)

    if args.skip_bertscore:
        print("Skipping BERTScore", file=sys.stderr)
        bert_scores = [None] * len(qas)
    else:
        print("Computing BERTScore per QA...", file=sys.stderr)
        bert_scores = _compute_bert_scores(qas, args.eval_key)

    per_category = _aggregate_by_category(qas, all_f1, all_recall, bert_scores)

    summary = _build_summary(per_category, qas, all_f1, all_recall, bert_scores)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for cat, data in sorted(per_category.items(), key=lambda kv: int(kv[0])):
        print(
            f"  Cat {cat}: count={data['count']}, "
            f"f1={data['f1_avg']}, "
            f"recall={data['recall_avg']}, "
            f"bert={data.get('bert_score_avg', 'N/A')}",
            file=sys.stderr,
        )
    print(
        f"  Overall: count={summary['qa_count']}, "
        f"f1={summary['f1_score_avg']}, "
        f"bert={summary['bert_score_avg']}, "
        f"adv_acc={summary['adversarial_accuracy']}",
        file=sys.stderr,
    )

    print(f"Official eval results saved to {args.output}", file=sys.stderr)
    return 0


def _map_to_official_format(
    results: list[dict[str, Any]], eval_key: str
) -> list[dict[str, Any]]:
    """Map memento runner results to the format expected by eval_question_answering()."""

    qas: list[dict[str, Any]] = []
    for item in results:
        category = item.get("category")
        if category is None:
            continue

        output = item.get(eval_key) or ""
        answer = item.get("gold_answer")

        if answer is None and category != 5:
            continue

        mapped: dict[str, Any] = {
            eval_key: output,
            "answer": answer,
            "category": category,
            "evidence": item.get("evidence_ids") or [],
        }

        ctx = item.get("retrieved_ids") or []
        # Official LOCOMO eval assumes non-empty context when evidence exists.
        # Provide a sentinel so it doesn't IndexError on empty retrieval lists.
        mapped[eval_key + "_context"] = ctx if ctx else [""]

        qas.append(mapped)

    return qas


def _compute_bert_scores(
    qas: list[dict[str, Any]], eval_key: str
) -> list[float | None]:
    """Compute BERTScore for each QA using the official locomo bert_score wrapper.

    Returns None for QAs where gold answer is None (category 5 adversarial).
    """
    scores: list[float | None] = []
    for qa in qas:
        gold = qa.get("answer")
        if gold is None:
            scores.append(None)
            continue

        prediction = qa.get(eval_key) or ""
        if not prediction.strip():
            scores.append(0.0)
            continue

        try:
            bs = locomo_bert_score(prediction, str(gold))
            scores.append(round(bs, 4))
        except Exception as exc:
            print(
                f"  BERTScore error (cat {qa['category']}): {exc}",
                file=sys.stderr,
            )
            scores.append(None)

    return scores


def _aggregate_by_category(
    qas: list[dict[str, Any]],
    all_f1: list[float],
    all_recall: list[float],
    bert_scores: list[float | None],
) -> dict[str, dict[str, Any]]:
    categories: dict[str, dict[str, Any]] = {}

    for idx, qa in enumerate(qas):
        cat = str(qa["category"])
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "f1_sum": 0.0,
                "recall_sum": 0.0,
                "bert_sum": 0.0,
                "bert_count": 0,
            }

        cat_data = categories[cat]
        cat_data["count"] += 1
        cat_data["f1_sum"] += all_f1[idx]
        cat_data["recall_sum"] += all_recall[idx]

        bs = bert_scores[idx]
        if bs is not None:
            cat_data["bert_sum"] += bs
            cat_data["bert_count"] += 1

    result: dict[str, dict[str, Any]] = {}
    for cat, data in sorted(categories.items(), key=lambda kv: int(kv[0])):
        count = data["count"]
        result[cat] = {
            "count": count,
            "f1_avg": round(data["f1_sum"] / count, 4) if count else 0.0,
            "recall_avg": round(data["recall_sum"] / count, 4) if count else 0.0,
        }
        if data["bert_count"]:
            result[cat]["bert_score_avg"] = round(
                data["bert_sum"] / data["bert_count"], 4
            )
        else:
            result[cat]["bert_score_avg"] = None

    return result


def _build_summary(
    per_category: dict[str, dict[str, Any]],
    qas: list[dict[str, Any]],
    all_f1: list[float],
    all_recall: list[float],
    bert_scores: list[float | None],
) -> dict[str, Any]:
    total = len(qas)
    f1_avg = round(sum(all_f1) / total, 4) if total else 0.0
    recall_avg = round(sum(all_recall) / total, 4) if total else 0.0

    valid_bert = [bs for bs in bert_scores if bs is not None]
    bert_avg = round(sum(valid_bert) / len(valid_bert), 4) if valid_bert else None

    cat_5_data = per_category.get("5", {})
    cat_5_count = cat_5_data.get("count", 0)
    cat_5_f1_avg = cat_5_data.get("f1_avg", 0.0)

    return {
        "eval_source": "official LOCOMO task_eval.evaluation.eval_question_answering",
        "eval_note": (
            "Official LOCOMO F1 with PorterStemmer + adversarial check + BERTScore. "
            "Not comparable to local token_f1 diagnostics."
        ),
        "qa_count": total,
        "f1_score_avg": f1_avg,
        "f1_per_category": {
            cat: data["f1_avg"]
            for cat, data in sorted(per_category.items(), key=lambda kv: int(kv[0]))
        },
        "recall_avg": recall_avg,
        "adversarial_accuracy": cat_5_f1_avg,
        "adversarial_count": cat_5_count,
        "bert_score_avg": bert_avg,
        "bert_score_count": len(valid_bert),
        "per_category": per_category,
    }


def _category_counts(qas: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for qa in qas:
        cat = qa["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
