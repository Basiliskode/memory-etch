#!/usr/bin/env python3
"""Backfill missing embedding vectors for existing facts.

Queries facts where ``embedding IS NULL``, encodes them via BGE-M3
(in batches), and updates the rows.

Usage:
    python scripts/backfill_embeddings.py path/to/memory.db [--batch 32]

Requires ``pip install memento[bge-m3]``.
"""

import argparse
import logging
import sqlite3
import struct
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def backfill(db_path: str, batch_size: int = 32) -> int:
    """Encode facts with NULL embedding and store the vectors.

    Args:
        db_path: Path to the SQLite database.
        batch_size: Facts per encoding batch (default 32).

    Returns:
        Number of facts backfilled.
    """
    try:
        from memento.plugins.bge_m3 import BgeM3Plugin
    except ImportError as exc:
        logger.error(
            "BGE-M3 plugin not available. Install with: pip install memento[bge-m3]\n%s",
            exc,
        )
        return 0

    plugin = BgeM3Plugin()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Count facts needing backfill
    total = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE embedding IS NULL AND (deleted IS NULL OR deleted = 0)"
    ).fetchone()[0]

    if total == 0:
        logger.info("No facts need backfill — all have embeddings.")
        conn.close()
        return 0

    logger.info("Backfilling %d facts (batch size: %d)…", total, batch_size)
    done = 0

    while done < total:
        rows = conn.execute(
            "SELECT fact_id, content FROM facts "
            "WHERE embedding IS NULL AND (deleted IS NULL OR deleted = 0) "
            "LIMIT ?",
            (batch_size,),
        ).fetchall()

        if not rows:
            break

        texts = [r["content"] for r in rows]
        vectors = plugin.encode_batch(texts)

        for row, vec in zip(rows, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (blob, row["fact_id"]),
            )

        conn.commit()
        done += len(rows)
        logger.info("  … %d / %d", done, total)

    conn.close()
    logger.info("Backfill complete: %d facts updated.", done)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing embedding vectors")
    parser.add_argument("db_path", type=str, help="Path to memento SQLite database")
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    args = parser.parse_args()

    if not Path(args.db_path).exists():
        logger.error("Database not found: %s", args.db_path)
        sys.exit(1)

    count = backfill(args.db_path, batch_size=args.batch)
    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()
