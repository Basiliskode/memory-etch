"""EtchIndex — lightweight code index for Hermes agents."""
import sqlite3
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    files_indexed: int = 0
    functions_indexed: int = 0
    classes_indexed: int = 0


@dataclass
class IndexItem:
    qualified_name: str
    file_path: str
    item_type: str = "function"
    line_number: int = 0


class EtchIndex:
    """Lightweight code index for a project.

    Indexes Python functions and classes for fast lookup.
    """

    def __init__(self, db_path: str):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS index_entries ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  qualified_name TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  item_type TEXT NOT NULL,"
            "  line_number INTEGER DEFAULT 0"
            ")"
        )
        self._conn.commit()

    def index(self, path: Path) -> IndexStats:
        """Index Python files in *path*."""
        stats = IndexStats()
        if not path.exists():
            return stats

        py_files = list(path.rglob("*.py")) if path.is_dir() else [path]
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
                items = self._extract_items(tree, py_file)
                if items:
                    stats.files_indexed += 1
                    self._store_items(items)
            except SyntaxError:
                logger.warning("Skipping %s: syntax error", py_file)

        return stats

    def _extract_items(self, tree: ast.AST, file_path: Path) -> list[tuple]:
        """Extract function/class definitions from AST."""
        items = []
        rel_path = str(file_path)
        file_name = file_path.name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                qualified = f"{file_name}::{node.name}"
                items.append((qualified, rel_path, "function", node.lineno or 0))
            elif isinstance(node, ast.AsyncFunctionDef):
                qualified = f"{file_name}::{node.name}"
                items.append((qualified, rel_path, "function", node.lineno or 0))
            elif isinstance(node, ast.ClassDef):
                qualified = f"{file_name}::{node.name}"
                items.append((qualified, rel_path, "class", node.lineno or 0))
        return items

    def _store_items(self, items: list[tuple]) -> None:
        for name, fpath, itype, lineno in items:
            self._conn.execute(
                "INSERT OR REPLACE INTO index_entries (qualified_name, file_path, item_type, line_number) "
                "VALUES (?, ?, ?, ?)",
                (name, fpath, itype, lineno),
            )
        self._conn.commit()

    def search(self, query: str, limit: int = 10) -> list[IndexItem]:
        """Search for indexed items by name."""
        rows = self._conn.execute(
            "SELECT qualified_name, file_path, item_type, line_number FROM index_entries "
            "WHERE qualified_name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            IndexItem(
                qualified_name=r[0],
                file_path=r[1],
                item_type=r[2],
                line_number=r[3],
            )
            for r in rows
        ]
