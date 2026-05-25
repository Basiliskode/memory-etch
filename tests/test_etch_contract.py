from pathlib import Path


def test_etch_memory_provider_and_index_public_api(tmp_path):
    from memento.etch import EtchMemoryProvider
    from plugins.etch_index import EtchIndex

    assert EtchMemoryProvider.__name__ == "EtchMemoryProvider"
    assert EtchIndex.__name__ == "EtchIndex"

    (tmp_path / "app.py").write_text("def etch_demo():\n    return 1\n", encoding="utf-8")
    index = EtchIndex(tmp_path / ".hermes" / "etch-index.db")
    stats = index.index(tmp_path)

    assert stats.files_indexed == 1
    assert index.search("etch_demo", limit=1)[0].qualified_name == "app.py::etch_demo"


def test_etch_branding_does_not_expose_old_names_in_plugin_metadata():
    plugin_yaml = Path("plugins/memory/etch/plugin.yaml").read_text(encoding="utf-8")

    assert "name: etch" in plugin_yaml
    assert "memento" in plugin_yaml
    assert "holo" + "graphic" not in plugin_yaml.lower()
    assert "code" + "graph" not in plugin_yaml.lower()
