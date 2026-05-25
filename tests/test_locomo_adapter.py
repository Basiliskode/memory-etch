import json
from pathlib import Path

from benchmarks.locomo.loader import load_locomo
from benchmarks.locomo.runner import (
    GeminiAnswerer,
    LocomoRunner,
    build_answer_prompt,
    build_gemini_payload,
    evaluate_qa_diagnostics,
    main,
)


def test_load_locomo_official_shape(tmp_path: Path):
    data_path = tmp_path / "locomo10.json"
    data_path.write_text(json.dumps([_sample()]), encoding="utf-8")

    samples = load_locomo(data_path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "sample_alpha"
    assert [turn.dia_id for turn in sample.turns] == ["D1:1", "D1:2", "D2:1"]
    assert sample.turns[0].timestamp == "10:00 am on 1 May, 2023"
    assert sample.turns[1].metadata["blip_caption"] == "a blue mug on a table"
    assert sample.qa[0].question_id == "sample_alpha:q1"
    assert sample.qa[0].category == 1
    assert sample.qa[0].evidence == ["D1:2", "D2:1"]


def test_locomo_runner_writes_dry_run_jsonl(tmp_path: Path):
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.jsonl"
    data_path.write_text(json.dumps([_sample()]), encoding="utf-8")

    runner = LocomoRunner(
        samples=load_locomo(data_path),
        output_path=output_path,
        top_k=2,
        memory_variant="etch-noop",
        answerer_provider="none",
        seed=123,
    )
    summary = runner.run()

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["dry_run"] is True
    assert summary["retrieval_metrics"]["qa_with_evidence"] == 1
    assert lines[0]["type"] == "summary"
    assert lines[1]["type"] == "result"
    assert lines[1]["prediction"] == "DRY_RUN_NO_ANSWERER_CONFIGURED"
    assert lines[1]["evidence_ids"] == ["D1:2", "D2:1"]
    assert set(lines[1]["retrieved_ids"])


def test_locomo_runner_broadens_natural_language_questions(tmp_path: Path):
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.json"
    sample = _sample()
    sample["qa"] = [
        {
            "question": "When did Alice paint the blue mug?",
            "answer": "10:00 am on 1 May, 2023",
            "category": 3,
            "evidence": ["D1:2"],
        }
    ]
    data_path.write_text(json.dumps([sample]), encoding="utf-8")

    runner = LocomoRunner(
        samples=load_locomo(data_path),
        output_path=output_path,
        top_k=2,
        memory_variant="etch-noop",
        answerer_provider="none",
        seed=123,
    )
    runner.run()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["retrieval_metrics"]["any_evidence_hit_rate"] == 1.0
    result = payload["results"][0]
    assert "D1:2" in result["retrieved_ids"]


def test_locomo_runner_isolates_conversations(tmp_path: Path):
    """Two conversations with overlapping dia_ids must not leak turns across retrieval.

    conv_B has turns that match the same query terms as conv_A's QA question.
    Without project-scoped retrieval the fallback path would leak conv_B turns.
    """
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.json"
    samples_data = [
        {
            "sample_id": "conv_A",
            "conversation": {
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice loves red cars."},
                    {"speaker": "Alice", "dia_id": "D1:2", "text": "Her birthday is in June."},
                ],
            },
            "qa": [
                {
                    "question": "red car",
                    "answer": "Alice loves red cars",
                    "category": 1,
                    "evidence": ["D1:1"],
                }
            ],
        },
        {
            "sample_id": "conv_B",
            "conversation": {
                "session_1": [
                    {"speaker": "Bob", "dia_id": "D1:1", "text": "Bob loves red cars too."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "His birthday is in March."},
                ],
            },
            "qa": [],
        },
    ]
    data_path.write_text(json.dumps(samples_data), encoding="utf-8")

    runner = LocomoRunner(
        samples=load_locomo(data_path),
        output_path=output_path,
        top_k=10,
        memory_variant="etch-noop",
        answerer_provider="none",
        seed=42,
    )
    runner.run()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    result = payload["results"][0]
    assert result["conversation_id"] == "conv_A"
    for ctx in result["retrieved_context"]:
        assert ctx.get("turn_id") is not None
    # All retrieved turns MUST be from conv_A (no leakage from conv_B).
    # Conv_A has "Alice loves red cars"; conv_B has "Bob loves red cars too".
    # If project filtering in the fallback path worked, no "Bob" should appear.
    retrieved_texts = [ctx["text"] for ctx in result["retrieved_context"]]
    assert "red cars" in " ".join(retrieved_texts)
    assert all("Bob" not in text for text in retrieved_texts), (
        f"Conv_B leaked into retrieval: {retrieved_texts}"
    )
    assert "D1:1" in result["retrieved_ids"]


def test_locomo_extractive_answerer_writes_prediction_and_metadata(tmp_path: Path):
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.json"
    data_path.write_text(json.dumps([_sample()]), encoding="utf-8")

    runner = LocomoRunner(
        samples=load_locomo(data_path),
        output_path=output_path,
        top_k=2,
        memory_variant="etch-noop",
        answerer_provider="extractive",
        local_eval=True,
        seed=123,
        dataset_path=str(data_path),
    )
    summary = runner.run()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    assert summary["dry_run"] is False
    assert summary["prompt_version"] == "locomo-context-only-v1"
    assert summary["qa_diagnostics"]["metric_set"] == "local_qa_diagnostics_v1"
    assert result["prediction"] == "Alice painted a blue mug in pottery class."
    assert result["prediction_meta"]["provider"] == "extractive"
    assert result["prediction_meta"]["source_dia_id"] == "D1:2"


def test_answer_prompt_requires_context_only_short_answer():
    prompt = build_answer_prompt(
        "What did Alice paint?",
        [
            {
                "dia_id": "D1:2",
                "session_id": "session_1",
                "timestamp": "10:00 am",
                "speaker": "Alice",
                "text": "Alice painted a blue mug.",
            }
        ],
    )

    assert "Use only the retrieved context" in prompt
    assert "Do not use outside knowledge or guess" in prompt
    assert "No information available" in prompt
    assert "Context ID: D1:2" in prompt
    assert "Short answer:" in prompt


def test_evaluate_qa_diagnostics_on_synthetic_outputs():
    metrics = evaluate_qa_diagnostics(
        [
            {
                "question_id": "q1",
                "gold_answer": "a blue mug",
                "prediction": "Alice painted a blue mug in pottery class.",
            },
            {
                "question_id": "q2",
                "gold_answer": "Dana",
                "prediction": "No information available",
            },
        ]
    )

    assert metrics["metric_set"] == "local_qa_diagnostics_v1"
    assert metrics["qa_with_gold"] == 2
    assert metrics["contains_avg"] == 0.5
    assert metrics["exact-ish_avg"] == 0.0
    assert metrics["token_f1_avg"] == 0.2222


def test_locomo_cli_local_eval_metadata(tmp_path: Path):
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.json"
    data_path.write_text(json.dumps([_sample()]), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(data_path),
            "--output",
            str(output_path),
            "--answerer-provider",
            "extractive",
            "--local-eval",
            "--qa-limit",
            "1",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["summary"]["answerer_provider"] == "extractive"
    assert payload["summary"]["dataset_path"] == str(data_path)
    assert payload["summary"]["qa_diagnostics"]["qa_with_gold"] == 1


def test_locomo_cli_accepts_gemini_and_reports_missing_key(
    tmp_path: Path, monkeypatch, capsys
):
    data_path = tmp_path / "locomo10.json"
    output_path = tmp_path / "results.json"
    data_path.write_text(json.dumps([_sample()]), encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    exit_code = main(
        [
            "--input",
            str(data_path),
            "--output",
            str(output_path),
            "--answerer-provider",
            "gemini",
            "--qa-limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "GEMINI_API_KEY is required" in captured.err
    assert "invalid choice" not in captured.err
    assert "fake-test-key" not in captured.err


def test_gemini_payload_uses_context_only_prompt():
    payload = build_gemini_payload(
        "What did Alice paint?",
        [
            {
                "dia_id": "D1:2",
                "session_id": "session_1",
                "timestamp": "10:00 am",
                "speaker": "Alice",
                "text": "Alice painted a blue mug.",
            }
        ],
    )

    prompt = payload["contents"][0]["parts"][0]["text"]
    assert payload["systemInstruction"]["parts"][0]["text"] == (
        "Answer LOCOMO questions using only the supplied context."
    )
    assert "Use only the retrieved context" in prompt
    assert "Context ID: D1:2" in prompt
    assert payload["generationConfig"]["maxOutputTokens"] == 256


def test_gemini_answerer_posts_generate_content_request(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "a blue mug"}]}}
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("benchmarks.locomo.runner.urllib.request.urlopen", fake_urlopen)

    answerer = GeminiAnswerer("gemini-test-model", timeout_seconds=12.5)
    result = answerer.answer(
        "What did Alice paint?",
        [
            {
                "dia_id": "D1:2",
                "session_id": "session_1",
                "timestamp": "10:00 am",
                "speaker": "Alice",
                "text": "Alice painted a blue mug.",
            }
        ],
    )

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert result["text"] == "a blue mug"
    assert result["meta"]["provider"] == "gemini"
    assert result["meta"]["model"] == "gemini-test-model"
    assert timeout == 12.5
    assert "gemini-test-model:generateContent" in request.full_url
    assert payload["contents"][0]["parts"][0]["text"].endswith("Short answer:")


def _sample():
    return {
        "sample_id": "sample_alpha",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "10:00 am on 1 May, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I started pottery today."},
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Alice painted a blue mug in pottery class.",
                    "blip_caption": "a blue mug on a table",
                },
            ],
            "session_2_date_time": "11:00 am on 2 May, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "The mug is a gift for Dana."},
            ],
        },
        "qa": [
            {
                "question": "blue mug",
                "answer": "a blue mug",
                "category": 1,
                "evidence": ["D1:2; D2:1"],
            }
        ],
    }
