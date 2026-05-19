"""Tests for the Memory Etch viewer (P6)."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from memory_etch.viewer import ViewerHandler, find_db_path


@pytest.fixture(scope="module")
def viewer_server(tmp_path_factory):
    """Start a viewer server for the duration of the module."""
    import sqlite3
    db_file = tmp_path_factory.mktemp("etch") / "etch_memory.db"
    from memory_etch.store import EtchStore
    store = EtchStore(str(db_file))
    store.add_fact("HermesDM is a D&D bot", category="general", tags="dnd,bot,topic:app")
    store.add_fact("User likes dark mode", category="preference", tags="ui,theme")
    store.add_fact("Flask version is 3.1", category="tech", tags="python,framework")
    conn = store._conn
    conn.row_factory = sqlite3.Row
    server = HTTPServer(("127.0.0.1", 0), ViewerHandler)  # random port
    port = server.server_port
    server._db = conn
    server._db_path = str(db_file)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    conn.close()


def _get(url):
    try:
        r = urllib.request.urlopen(url, timeout=5)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}


class TestViewer:
    """Integration tests for the viewer HTTP API."""

    def test_serves_html(self, viewer_server):
        """GET / serves the HTML page."""
        r = urllib.request.urlopen(viewer_server + "/", timeout=5)
        assert r.status == 200
        html = r.read().decode()
        assert "<!DOCTYPE html>" in html
        assert "Etch Viewer" in html
        assert "application/json" not in r.headers["Content-Type"]

    def test_stats_endpoint(self, viewer_server):
        """GET /api/stats returns counts."""
        status, data = _get(viewer_server + "/api/stats")
        assert status == 200
        assert "fact_count" in data
        assert "session_count" in data
        assert "relation_count" in data
        assert isinstance(data["fact_count"], int)

    def test_facts_list_defaults(self, viewer_server):
        """GET /api/facts returns facts sorted by trust DESC."""
        status, data = _get(viewer_server + "/api/facts")
        assert status == 200
        assert "facts" in data
        assert len(data["facts"]) <= 50  # default limit
        assert "count" in data
        if len(data["facts"]) > 1:
            scores = [f["trust_score"] for f in data["facts"]]
            assert scores == sorted(scores, reverse=True)

    def test_facts_pagination(self, viewer_server):
        """GET /api/facts?limit=3 works."""
        status, data = _get(viewer_server + "/api/facts?limit=3")
        assert status == 200
        assert len(data["facts"]) == 3
        assert data["count"] >= 3

    def test_facts_category_filter(self, viewer_server):
        """GET /api/facts?category=general filters correctly."""
        status, data = _get(viewer_server + "/api/facts?category=general")
        assert status == 200
        for f in data["facts"]:
            assert f["category"] == "general"

    def test_fact_detail(self, viewer_server):
        """GET /api/facts/<id> returns a single fact."""
        # First get a valid ID
        _, listing = _get(viewer_server + "/api/facts?limit=1")
        if not listing["facts"]:
            pytest.skip("No facts in DB")
        fid = listing["facts"][0]["fact_id"]
        status, data = _get(f"{viewer_server}/api/facts/{fid}")
        assert status == 200
        assert data["fact_id"] == fid
        assert "content" in data
        assert "category" in data
        assert "trust_score" in data

    def test_fact_detail_not_found(self, viewer_server):
        """GET /api/facts/999999 returns 404."""
        status, data = _get(viewer_server + "/api/facts/999999")
        assert status == 404
        assert "error" in data

    def test_fact_detail_invalid_id(self, viewer_server):
        """GET /api/facts/abc returns 400."""
        status, data = _get(viewer_server + "/api/facts/abc")
        assert status == 400

    def test_search(self, viewer_server):
        """GET /api/search?q= finds matching facts."""
        status, data = _get(viewer_server + "/api/search?q=HermesDM")
        assert status == 200
        assert "results" in data
        assert data["count"] >= 1
        for r in data["results"]:
            assert "content" in r

    def test_search_empty_query(self, viewer_server):
        """GET /api/search?q= returns empty."""
        status, data = _get(viewer_server + "/api/search?q=")
        assert status == 200
        assert data["count"] == 0

    def test_projects(self, viewer_server):
        """GET /api/projects returns list."""
        status, data = _get(viewer_server + "/api/projects")
        assert status == 200
        assert isinstance(data, dict)

    def test_db_endpoint(self, viewer_server):
        """GET /api/db returns path info."""
        status, data = _get(viewer_server + "/api/db")
        assert status == 200
        assert "path" in data
        assert data["path"].endswith("etch_memory.db")

    def test_relations_endpoint(self, viewer_server):
        """GET /api/relations handles missing table gracefully."""
        status, data = _get(viewer_server + "/api/relations/8")
        assert status == 200
        assert "relations" in data

    def test_timeline_endpoint(self, viewer_server):
        """GET /api/timeline/<id> returns before/after."""
        status, data = _get(viewer_server + "/api/timeline/1")
        assert status == 200
        assert "fact" in data
        assert "before" in data
        assert "after" in data

    def test_timeline_not_found(self, viewer_server):
        """GET /api/timeline/999999 returns 404."""
        status, data = _get(viewer_server + "/api/timeline/999999")
        assert status == 404

    def test_unknown_api(self, viewer_server):
        """GET /api/nonexistent returns 404."""
        status, data = _get(viewer_server + "/api/nonexistent")
        assert status == 404

    def test_cors_headers(self, viewer_server):
        """API responses include CORS headers."""
        r = urllib.request.urlopen(viewer_server + "/api/stats", timeout=5)
        assert r.getheader("Access-Control-Allow-Origin") == "*"
