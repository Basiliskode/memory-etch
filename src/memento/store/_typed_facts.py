"""Typed facts — schema registry and validation.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def register_schema(
    store,
    fact_type: str,
    description: str = "",
    required_fields: Optional[list[str]] = None,
    optional_fields: Optional[list[str]] = None,
) -> dict:
    """Register or update a fact schema.

    Args:
        fact_type: The fact type name (primary key).
        description: Optional description of the schema.
        required_fields: List of field names that must be non-empty.
        optional_fields: List of field names that are optional.

    Returns:
        The schema dict as stored.
    """
    with store._lock:
        store._conn.execute(
            """INSERT OR REPLACE INTO fact_schemas
               (fact_type, description, required_fields, optional_fields, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (fact_type, description,
             json.dumps(required_fields or []),
             json.dumps(optional_fields or [])),
        )
        store._conn.commit()
        store._log_event("schema_registered", fact_id=0, project="",
                         metadata={"fact_type": fact_type})
        return store.get_schema(fact_type)  # type: ignore[return-value]


def get_schema(store, fact_type: str) -> Optional[dict]:
    """Get a fact schema by type.

    Args:
        fact_type: The fact type name.

    Returns:
        Schema dict or None if not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM fact_schemas WHERE fact_type = ?",
            (fact_type,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["required_fields"] = json.loads(d.get("required_fields", "[]"))
    d["optional_fields"] = json.loads(d.get("optional_fields", "[]"))
    return d


def list_schemas(store) -> list[dict]:
    """List all registered fact schemas.

    Returns:
        List of schema dicts, ordered by fact_type.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM fact_schemas ORDER BY fact_type",
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["required_fields"] = json.loads(d.get("required_fields", "[]"))
        d["optional_fields"] = json.loads(d.get("optional_fields", "[]"))
        result.append(d)
    return result


def delete_schema(store, fact_type: str) -> bool:
    """Delete a fact schema.

    Args:
        fact_type: The fact type to delete.

    Returns:
        True if a schema was deleted, False if not found.
    """
    with store._lock:
        cur = store._conn.execute(
            "DELETE FROM fact_schemas WHERE fact_type = ?",
            (fact_type,),
        )
        store._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            store._log_event("schema_deleted", fact_id=0, project="",
                             metadata={"fact_type": fact_type})
    return deleted


def _validate_fact_type(
    store,
    fact_type: str,
    what: Optional[str],
    why: Optional[str],
    where_text: Optional[str],
    learned: Optional[str],
) -> None:
    """Validate that a fact meets its schema requirements.

    Called inside ``store._lock`` — does NOT acquire the lock.

    Args:
        fact_type: The fact type to validate against.
        what: The ``what`` field value.
        why: The ``why`` field value.
        where_text: The ``where_text`` field value.
        learned: The ``learned`` field value.

    Raises:
        ValueError: If the fact type is not registered or a required
            field is empty.
    """
    if not fact_type:
        return
    row = store._conn.execute(
        "SELECT required_fields FROM fact_schemas WHERE fact_type = ?",
        (fact_type,),
    ).fetchone()
    if not row:
        raise ValueError(
            f"Fact type '{fact_type}' is not registered. "
            "Call register_schema() first."
        )
    required = json.loads(row["required_fields"])
    field_map = {
        "what": what,
        "why": why,
        "where": where_text,
        "learned": learned,
    }
    for field in required:
        val = field_map.get(field)
        if not val or (isinstance(val, str) and not val.strip()):
            raise ValueError(
                f"Fact type '{fact_type}' requires field '{field}'"
            )
