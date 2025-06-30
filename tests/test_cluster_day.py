import importlib
from pathlib import Path


def test_cluster_day(tmp_path):
    mod = importlib.import_module("think.cluster_day")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "120000_audio.json").write_text('{"text": "hi"}')
    (day / "120500_screen.md").write_text("screen summary")
    result, count = mod.cluster_day(str(day))
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result
