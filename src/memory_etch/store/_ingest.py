"""Ingest pipeline — import data from files or text into the store.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
from pathlib import Path
from typing import Optional

from memory_etch import ingest as _ingest

logger = logging.getLogger(__name__)


def ingest(
    store,
    source: str | Path,
    format: str = "auto",
    project: str = "",
    category: str = "general",
    tags: str = "",
    delimiter: str | int = "paragraph",
    content_key: str | None = None,
    encoding: str = "utf-8",
    batch_size: int = 50,
) -> dict:
    """Ingest data from a file or text blob into the store.

    Reads content from *source* (file path or raw text string), detects
    or uses the specified *format*, parses the content into facts, and
    inserts them via ``add_fact``.  Commits every *batch_size* facts.

    Args:
        source: File path (``str`` or ``Path``) or raw text string.
        format: ``"auto"`` (detect from extension/content), ``"markdown"``,
            ``"text"``, ``"json"``, ``"csv"``.
        project: Project namespace for all ingested facts.
        category: Category for all ingested facts.
        tags: Tags for all ingested facts.
        delimiter: For ``format="text"`` — ``"paragraph"``, ``"line"``,
            or an integer chunk size.
        content_key: For ``format="json"`` — extract content from this
            key in each dict item.
        encoding: File encoding when *source* is a file path.
        batch_size: Commit to SQLite every *batch_size* facts.

    Returns:
        Stats dict with keys ``total``, ``created``, ``deduped``,
        ``errors``.
    """
    # --- Determine whether source is a file path ---
    source_path: Path | None = None
    if isinstance(source, Path):
        if source.exists() and source.is_file():
            source_path = source
    elif isinstance(source, str):
        # Treat as path if it looks like one (contains path separators
        # or ends with a recognised extension and exists on disk)
        if not source.strip():
            source_path = None  # empty string — treat as text
        else:
            p = Path(source)
            if p.exists() and p.is_file():
                source_path = p

    # --- Resolve format ---
    fmt = format
    text: str | None = None
    if source_path is not None:
        if fmt == "auto":
            fmt = _ingest.detect_format(source_path)
        text = source_path.read_text(encoding=encoding)
    else:
        text = str(source)
        if fmt == "auto":
            fmt = _ingest.detect_format(source, text=text)

    if text is None or not text.strip():
        return {"total": 0, "created": 0, "deduped": 0, "errors": 0}

    # --- Dispatch to parser ---
    if fmt == "markdown":
        parser = _ingest.parse_markdown(text)
    elif fmt == "text":
        parser = _ingest.parse_text(text, delimiter=delimiter)
    elif fmt == "json":
        parser = _ingest.parse_json(text, content_key=content_key)
    elif fmt == "csv":
        parser = _ingest.parse_csv(text)
    else:
        raise ValueError(f"Unknown ingest format: {fmt!r}")

    # --- Ingest loop ---
    total = 0
    created = 0
    deduped = 0
    errors = 0

    for content, metadata in parser:
        total += 1
        try:
            result = store.add_fact(
                content=content,
                category=category,
                tags=tags,
                project=project,
                source_harness="ingest",
                source_kind=fmt,
                return_metadata=True,
            )
            if isinstance(result, dict):
                if result.get("status") == "created":
                    created += 1
                elif result.get("status") in ("dedup", "updated"):
                    deduped += 1
                else:
                    created += 1  # fallback
            elif result:
                created += 1
        except Exception:
            logger.exception("Ingest failed for fact %d: %s", total, content[:80])
            errors += 1

        # Periodic commit
        if total % batch_size == 0:
            with store._lock:
                store._conn.commit()

    # Final commit for remainder
    with store._lock:
        store._conn.commit()
        store._log_event(
            "ingest_completed",
            project=project,
            metadata={
                "format": fmt,
                "total": total,
                "created": created,
                "deduped": deduped,
                "errors": errors,
                "category": category,
            },
        )

    return {
        "total": total,
        "created": created,
        "deduped": deduped,
        "errors": errors,
    }
