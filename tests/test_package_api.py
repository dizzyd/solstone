import importlib


def test_public_functions(tmp_path):
    think = importlib.import_module("think")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "entities.md").write_text("* Person: Jane\n")
    ent = think.Entities(str(tmp_path))
    ent.rescan()
    res = ent.index()
    assert "Person" in res
    assert hasattr(think, "cluster")
