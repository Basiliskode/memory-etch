"""Tests for Memento Atlas — semantic fact map layer.

All tests use :memory: SQLite store. Covers CRUD for maps, regions,
edges, FTS5 search, traversal, lazy loading, edge cases, and
integration with EtchStore delegation, MCP, and Hermes.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from memento import EtchStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


# ===================================================================
# Task 1.1-1.2: Schema — atlas tables are created during migration
# ===================================================================

class TestAtlasSchema:
    """Verify atlas tables exist after store init."""

    def test_atlas_maps_table_exists(self, store):
        tables = {
            r["name"] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atlas_maps" in tables

    def test_atlas_regions_table_exists(self, store):
        tables = {
            r["name"] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atlas_regions" in tables

    def test_atlas_edges_table_exists(self, store):
        tables = {
            r["name"] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atlas_edges" in tables

    def test_atlas_maps_fts_table_exists(self, store):
        tables = {
            r["name"] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atlas_maps_fts" in tables

    def test_atlas_regions_fts_table_exists(self, store):
        tables = {
            r["name"] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atlas_regions_fts" in tables


# ===================================================================
# Task 1.3: Map CRUD — create_map, get_map, update_map, delete_map,
#            list_maps
# ===================================================================

class TestMapCRUD:
    """Covers task 4.1: 5 test cases for map CRUD."""

    def test_create_map_roundtrip(self, store):
        """RED → GREEN: create_map returns id, get_map retrieves it."""
        mid = store.create_map("Test Map", description="A test map",
                               tags="test,atlas", project="p1")
        assert isinstance(mid, int)
        assert mid > 0

        m = store.get_map(mid)
        assert m is not None
        assert m["name"] == "Test Map"
        assert m["description"] == "A test map"
        assert m["tags"] == "test,atlas"
        assert m["project"] == "p1"
        assert m["deleted"] == 0

    def test_get_map_nonexistent(self, store):
        """get_map returns None for missing id."""
        m = store.get_map(99999)
        assert m is None

    def test_list_maps(self, store):
        """list_maps returns all maps."""
        store.create_map("Map A", project="p1")
        store.create_map("Map B", project="p1")
        maps = store.list_maps()
        names = [m["name"] for m in maps]
        assert "Map A" in names
        assert "Map B" in names

    def test_delete_map(self, store):
        """delete_map soft-deletes and get_map excludes it."""
        mid = store.create_map("To Delete")
        assert store.delete_map(mid) is True
        m = store.get_map(mid)
        assert m is None or m["deleted"] == 1

    def test_update_map(self, store):
        """update_map changes name/description/tags."""
        mid = store.create_map("Original Name")
        ok = store.update_map(mid, name="Updated Name", description="New desc")
        assert ok is True
        m = store.get_map(mid)
        assert m["name"] == "Updated Name"
        assert m["description"] == "New desc"


# ===================================================================
# Task 1.4: Region CRUD — create_region, get_region, list_regions
# ===================================================================

class TestRegionCRUD:
    """Covers task 4.2: 3 test cases for regions."""

    def test_create_region_under_map(self, store):
        """Create region inside a map and retrieve it."""
        mid = store.create_map("Project Map")
        rid = store.create_region(mid, "Backend", description="Backend services")
        assert isinstance(rid, int)
        assert rid > 0

        r = store.get_region(rid)
        assert r is not None
        assert r["name"] == "Backend"
        assert r["map_id"] == mid

    def test_parent_region_hierarchy(self, store):
        """Parent region is stored and returned."""
        mid = store.create_map("Project Map")
        parent_id = store.create_region(mid, "Parent Region")
        child_id = store.create_region(mid, "Child Region",
                                       parent_region_id=parent_id)
        child = store.get_region(child_id)
        assert child["parent_region_id"] == parent_id

    def test_list_regions_by_map(self, store):
        """list_regions returns all regions for a map."""
        mid = store.create_map("Project Map")
        store.create_region(mid, "Region A")
        store.create_region(mid, "Region B")
        regions = store.list_regions(mid)
        assert len(regions) >= 2


# ===================================================================
# Task 1.5: Edge CRUD — link_fact, link_region, unlink, get_edges
# ===================================================================

class TestEdgeCRUD:
    """Covers task 4.3: 4 test cases for edges."""

    def test_link_fact(self, store):
        """link_fact creates an edge, get_edges returns it."""
        mid = store.create_map("Map")
        fid = store.add_fact("A fact for the map")
        edge_id = store.link_fact(mid, fid, relation_type="contains")
        assert isinstance(edge_id, int)
        assert edge_id > 0

        edges = store.get_edges()
        edge_ids = [e["edge_id"] for e in edges]
        assert edge_id in edge_ids

    def test_link_region(self, store):
        """link_region connects two regions."""
        mid = store.create_map("Map")
        r1 = store.create_region(mid, "Region A")
        r2 = store.create_region(mid, "Region B")
        edge_id = store.link_region(mid, r1, r2, relation_type="adjacent")
        assert edge_id > 0

    def test_unlink(self, store):
        """unlink removes an edge."""
        mid = store.create_map("Map")
        fid = store.add_fact("Fact")
        edge_id = store.link_fact(mid, fid)
        assert store.unlink(edge_id) is True

        edges = store.get_edges()
        eids = [e["edge_id"] for e in edges]
        assert edge_id not in eids

    def test_get_edges_with_filters(self, store):
        """get_edges filters by source_type and source_id."""
        mid = store.create_map("Map")
        fid1 = store.add_fact("Fact 1")
        fid2 = store.add_fact("Fact 2")
        e1 = store.link_fact(mid, fid1)
        store.link_fact(mid, fid2)
        edges = store.get_edges(node_type="fact", node_id=fid1)
        eids = [e["edge_id"] for e in edges]
        assert e1 in eids
        assert len(eids) == 1


# ===================================================================
# Task 1.6: FTS5 search — search_map
# ===================================================================

class TestAtlasSearch:
    """Covers task 4.4: 2 test cases for FTS5 search."""

    def test_search_finds_by_name(self, store):
        """search_map finds maps by name via FTS5."""
        store.create_map("Python Projects", description="Python related")
        store.create_map("Rust Projects", description="Rust related")
        results = store.search_map("python")
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert "Python Projects" in names

    def test_search_empty_query_returns_empty(self, store):
        """Empty query returns empty list."""
        store.create_map("Some Map")
        results = store.search_map("")
        assert results == []


# ===================================================================
# Task 1.7: Traverse path + map_stats
# ===================================================================

class TestTraverseAndStats:
    """Covers task 4.5: 2 test cases for traverse_path."""

    def test_traverse_path_found(self, store):
        """traverse_path returns path when regions are linked across maps."""
        m1 = store.create_map("Start")
        m2 = store.create_map("End")
        # Create regions in each map
        r1 = store.create_region(m1, "Region A")
        r2 = store.create_region(m2, "Region B")
        # Link regions between maps
        e1 = store.link_region(m1, r1, r2, relation_type="adjacent")
        assert e1 > 0
        path = store.traverse_path(r1, r2, max_depth=5)
        assert len(path) >= 1

    def test_traverse_path_depth_exceeded(self, store):
        """Empty path when depth exceeded or no connection."""
        m1 = store.create_map("Start")
        m2 = store.create_map("Unreachable")
        path = store.traverse_path(m1, m2, max_depth=2)
        assert path == []

    def test_map_stats(self, store):
        """map_stats returns counts for a map."""
        mid = store.create_map("Stats Map")
        store.create_region(mid, "Region A")
        fid = store.add_fact("Some fact")
        store.link_fact(mid, fid)
        stats = store.map_stats(mid)
        assert isinstance(stats, dict)
        assert "region_count" in stats
        assert "edge_count" in stats


# ===================================================================
# Task 1.6-1.7: Lazy content loading — search_map vs get_map_node
# ===================================================================

class TestLazyContentLoading:
    """Covers task 4.6: 2 test cases for lazy loading."""

    def test_search_map_returns_summary(self, store):
        """search_map returns summary (truncated to 200 chars), not full description."""
        store.create_map("Full Map", description="A" * 500)
        results = store.search_map("full")
        assert len(results) >= 1
        r = results[0]
        # Lazy-loading contract: summary computed, truncated to 200 chars
        assert "summary" in r
        assert len(r["summary"]) <= 200
        assert r["summary"] == "A" * 200

    def test_search_map_summary_with_short_description(self, store):
        """search_map summary equals full description when under 200 chars."""
        store.create_map("Short Map", description="Short description")
        results = store.search_map("short")
        assert len(results) >= 1
        r = results[0]
        assert "summary" in r
        assert r["summary"] == "Short description"

    def test_search_map_summary_with_empty_description(self, store):
        """search_map summary is empty string when description is empty."""
        store.create_map("No Desc Map", description="")
        results = store.search_map("no desc")
        assert len(results) >= 1
        r = results[0]
        assert "summary" in r
        assert r["summary"] == ""

    def test_get_map_node_returns_full(self, store):
        """get_map_node returns full record."""
        mid = store.create_map("Full Map", description="Detailed " * 20)
        node = store.get_map_node("map", mid)
        assert node is not None
        assert node["name"] == "Full Map"
        assert "description" in node


# ===================================================================
# Task 4.7: Edge cases — empty store, deleted excluded, project filter
# ===================================================================

class TestAtlasEdgeCases:
    """Covers task 4.7: 3 test cases."""

    def test_empty_store_returns_empty(self, store):
        """list_maps returns [] on empty store."""
        assert store.list_maps() == []

    def test_deleted_maps_excluded(self, store):
        """Deleted maps excluded from list_maps."""
        mid = store.create_map("Delete Me")
        store.delete_map(mid)
        maps = store.list_maps()
        mids = [m["map_id"] for m in maps]
        assert mid not in mids

    def test_project_filter(self, store):
        """list_maps filters by project."""
        store.create_map("Project A Map", project="proj_a")
        store.create_map("Project B Map", project="proj_b")
        maps_a = store.list_maps(project="proj_a")
        assert all(m["project"] == "proj_a" for m in maps_a)

    def test_create_map_empty_name_raises(self, store):
        """Empty map name raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="Map name must not be empty"):
            store.create_map("")

    def test_create_region_empty_name_raises(self, store):
        """Empty region name raises ValueError."""
        import pytest
        mid = store.create_map("Test")
        with pytest.raises(ValueError, match="Region name must not be empty"):
            store.create_region(mid, "")


# ===================================================================
# Task 4.8: Integration — EtchStore delegation wiring
# ===================================================================

class TestDelegationWiring:
    """Covers task 4.8: atlas methods exist on EtchStore."""

    def test_atlas_methods_exist(self, store):
        """Atlas methods are wired via delegation."""
        assert hasattr(store, "create_map")
        assert hasattr(store, "get_map")
        assert hasattr(store, "update_map")
        assert hasattr(store, "delete_map")
        assert hasattr(store, "list_maps")
        assert hasattr(store, "create_region")
        assert hasattr(store, "get_region")
        assert hasattr(store, "list_regions")
        assert hasattr(store, "link_fact")
        assert hasattr(store, "link_region")
        assert hasattr(store, "unlink")
        assert hasattr(store, "get_edges")
        assert hasattr(store, "search_map")
        assert hasattr(store, "get_map_node")
        assert hasattr(store, "traverse_path")
        assert hasattr(store, "map_stats")

    def test_create_map_via_store_roundtrip(self, store):
        """Full roundtrip: create → get via store delegation."""
        mid = store.create_map("Delegation Map")
        m = store.get_map(mid)
        assert m is not None
        assert m["name"] == "Delegation Map"


# ===================================================================
# Task 4.9: Integration — EtchRetriever explore_map / traverse_path /
#            search_map
# ===================================================================

class TestRetrieverIntegration:
    """Covers task 4.9: retrieval methods for atlas."""

    def test_explore_map(self, store):
        """EtchRetriever.explore_map returns map + linked items."""
        from memento.retrieval import EtchRetriever
        retriever = EtchRetriever(store)

        mid = store.create_map("Explorable Map")
        fid = store.add_fact("Referenced fact")
        store.link_fact(mid, fid)

        result = retriever.explore_map(mid)
        assert isinstance(result, list)

    def test_retriever_search_map(self, store):
        """EtchRetriever.search_map returns found maps."""
        from memento.retrieval import EtchRetriever
        retriever = EtchRetriever(store)

        store.create_map("Searchable Map")

        results = retriever.search_map("searchable")
        assert len(results) >= 1

    def test_retriever_traverse_path(self, store):
        """EtchRetriever.traverse_path connects maps via edges."""
        from memento.retrieval import EtchRetriever
        retriever = EtchRetriever(store)

        m1 = store.create_map("A")
        m2 = store.create_map("B")
        r1 = store.create_region(m1, "Region A")
        r2 = store.create_region(m2, "Region B")
        store.link_region(m1, r1, r2, relation_type="adjacent")
        path = retriever.traverse_path(r1, r2)
        assert len(path) >= 1


# ===================================================================
# Task 4.10: Integration — MCP tools for atlas
# ===================================================================

class TestMCPIntegration:
    """Covers task 4.10: MCP tool roundtrip for atlas."""

    def test_list_maps_empty(self):
        """list_maps returns [] when no maps."""
        from memento.mcp.server import list_maps as mcp_list_maps
        import json
        data = json.loads(mcp_list_maps())
        assert isinstance(data, list)

    def test_create_map_via_mcp(self):
        """Call create_map tool via MCP server function."""
        from memento.mcp.server import create_map as mcp_create_map
        import json
        result = mcp_create_map(
            name="MCP Map",
            description="Created via MCP",
        )
        data = json.loads(result)
        assert "map_id" in data
        assert data["map_id"] > 0

    def test_search_map_via_mcp(self):
        """search_map tool finds created map."""
        from memento.mcp.server import create_map as mcp_create_map
        from memento.mcp.server import search_map as mcp_search_map
        import json
        mcp_create_map(name="MCP Searchable", project="test")
        result = mcp_search_map(query="searchable")
        data = json.loads(result)
        assert len(data) >= 1


# ===================================================================
# Task 4.12: Snapshot roundtrip — atlas data preserved
# ===================================================================

class TestSnapshotIntegration:
    """Covers task 4.12: snapshot includes atlas data."""

    def test_snapshot_includes_atlas(self, store):
        """Snapshot data includes atlas maps/regions/edges."""
        mid = store.create_map("Snapshot Map")
        store.create_region(mid, "Snapshot Region")
        store.link_fact(mid, store.add_fact("Snapped fact"))

        snap = store.create_snapshot("atlas_snap")
        assert snap["fact_count"] >= 1

        # Retrieve the snapshot data and verify atlas sections exist
        full_snap = store.get_snapshot("atlas_snap")
        data = full_snap["data"]
        assert "atlas_maps" in data
        assert len(data["atlas_maps"]) >= 1
        assert data["atlas_maps"][0]["name"] == "Snapshot Map"
        assert "atlas_regions" in data
        assert len(data["atlas_regions"]) >= 1
        assert data["atlas_regions"][0]["name"] == "Snapshot Region"
        assert "atlas_edges" in data
        assert len(data["atlas_edges"]) >= 1


# ===================================================================
# Task 4.13: Export/Import roundtrip — v2 format preserves atlas
# ===================================================================

class TestExportImportIntegration:
    """Covers task 4.13: export/import preserves atlas data."""

    def test_export_import_preserves_atlas(self, tmp_path, store):
        """Export v2 → import → maps/regions/edges preserved."""
        mid = store.create_map("Export Map", project="test")
        store.create_region(mid, "Export Region")
        store.link_fact(mid, store.add_fact("Export fact"))

        out = tmp_path / "atlas_export.json"
        store.export_memory(str(out))

        store2 = EtchStore(":memory:", auto_migrate=True)
        store2.import_memory(str(out))

        maps = store2.list_maps()
        assert len(maps) >= 1
        names = [m["name"] for m in maps]
        assert "Export Map" in names
        store2.close()
