"""Load official LOCOMO conversation and QA data.

The official LOCOMO repository documents ``data/locomo10.json`` as a list of
conversation samples. Each sample has a ``conversation`` object with
``session_<num>`` turn lists and ``session_<num>_date_time`` timestamps, plus a
``qa`` list containing question, answer, category, and evidence dialog IDs.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCOMO_SOURCE_URL = "https://github.com/snap-research/locomo"


@dataclass(frozen=True)
class LocomoTurn:
    conversation_id: str
    session_id: str
    session_index: int
    turn_index: int
    dia_id: str
    speaker: str
    text: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocomoQA:
    question_id: str
    conversation_id: str
    question: str
    answer: Any = None
    category: Any = None
    evidence: list[str] = field(default_factory=list)
    adversarial_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocomoSample:
    sample_id: str
    turns: list[LocomoTurn]
    qa: list[LocomoQA]
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_locomo_input(path: str | Path) -> Path:
    """Resolve a LOCOMO file or directory to a JSON/JSONL input file."""

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for rel in ("data/locomo10.json", "locomo10.json", "locomo.json", "locomo10.jsonl"):
            nested = candidate / rel
            if nested.is_file():
                return nested
        json_files = sorted(candidate.glob("*.json")) + sorted(candidate.glob("*.jsonl"))
        if json_files:
            return json_files[0]

    raise FileNotFoundError(
        "LOCOMO data file not found. Clone or download the official dataset from "
        f"{LOCOMO_SOURCE_URL} and pass --data-dir /path/to/locomo or "
        "--input /path/to/locomo/data/locomo10.json."
    )


def load_locomo(path: str | Path) -> list[LocomoSample]:
    """Load LOCOMO samples from an official JSON/JSONL file or dataset directory."""

    input_path = resolve_locomo_input(path)
    raw = _read_json_or_jsonl(input_path)
    if isinstance(raw, dict) and "samples" in raw:
        raw_samples = raw["samples"]
    elif isinstance(raw, dict):
        raw_samples = [raw]
    elif isinstance(raw, list):
        raw_samples = raw
    else:
        raise ValueError(f"Unsupported LOCOMO root shape in {input_path}: {type(raw).__name__}")

    samples = [_parse_sample(item, index) for index, item in enumerate(raw_samples)]
    if not samples:
        raise ValueError(f"No LOCOMO samples found in {input_path}")
    return samples


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _parse_sample(raw: dict[str, Any], index: int) -> LocomoSample:
    if not isinstance(raw, dict):
        raise ValueError(f"LOCOMO sample #{index} must be an object")

    conversation_id = str(
        raw.get("sample_id")
        or raw.get("conversation_id")
        or raw.get("id")
        or f"sample_{index + 1}"
    )
    conversation = raw.get("conversation") or raw.get("dialogue") or raw.get("dialog")
    if not isinstance(conversation, dict):
        raise ValueError(f"LOCOMO sample {conversation_id} is missing conversation object")

    turns = _parse_turns(conversation_id, conversation)
    qa = _parse_qa(conversation_id, raw.get("qa") or raw.get("questions") or [])
    metadata = {
        key: value
        for key, value in raw.items()
        if key not in {"conversation", "dialogue", "dialog", "qa", "questions"}
    }
    return LocomoSample(sample_id=conversation_id, turns=turns, qa=qa, metadata=metadata)


def _parse_turns(conversation_id: str, conversation: dict[str, Any]) -> list[LocomoTurn]:
    session_keys = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            session_keys.append((int(match.group(1)), key))
    session_keys.sort()

    turns: list[LocomoTurn] = []
    for session_index, session_key in session_keys:
        timestamp = str(conversation.get(f"{session_key}_date_time") or "")
        for turn_index, turn in enumerate(conversation[session_key], start=1):
            if not isinstance(turn, dict):
                continue
            dia_id = str(turn.get("dia_id") or f"D{session_index}:{turn_index}")
            speaker = str(turn.get("speaker") or "unknown")
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            metadata = {
                key: value
                for key, value in turn.items()
                if key not in {"dia_id", "speaker", "text"}
            }
            turns.append(
                LocomoTurn(
                    conversation_id=conversation_id,
                    session_id=session_key,
                    session_index=session_index,
                    turn_index=turn_index,
                    dia_id=dia_id,
                    speaker=speaker,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )
    if not turns:
        raise ValueError(f"LOCOMO sample {conversation_id} contains no session turns")
    return turns


def _parse_qa(conversation_id: str, raw_qa: Any) -> list[LocomoQA]:
    if not isinstance(raw_qa, list):
        raise ValueError(f"LOCOMO sample {conversation_id} qa must be a list")

    items: list[LocomoQA] = []
    for index, item in enumerate(raw_qa, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("query") or "").strip()
        if not question:
            continue
        evidence = _normalize_evidence(item.get("evidence") or item.get("evidence_ids") or [])
        items.append(
            LocomoQA(
                question_id=str(
                    item.get("question_id") or item.get("id") or f"{conversation_id}:q{index}"
                ),
                conversation_id=conversation_id,
                question=question,
                answer=item.get("answer"),
                category=item.get("category"),
                evidence=evidence,
                adversarial_answer=item.get("adversarial_answer"),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "question_id",
                        "id",
                        "question",
                        "query",
                        "answer",
                        "category",
                        "evidence",
                        "evidence_ids",
                        "adversarial_answer",
                    }
                },
            )
        )
    return items


def _normalize_evidence(raw_evidence: Any) -> list[str]:
    if isinstance(raw_evidence, str):
        raw_items = [raw_evidence]
    elif isinstance(raw_evidence, list):
        raw_items = raw_evidence
    else:
        raw_items = []

    evidence: list[str] = []
    for value in raw_items:
        for part in str(value).split(";"):
            normalized = part.strip()
            if normalized:
                evidence.append(normalized)
    return evidence
