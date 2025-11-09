import importlib

from think.utils import day_path


def test_get_raw_file(tmp_path, monkeypatch):
    utils = importlib.import_module("think.utils")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    # Create periods
    period_123000 = day_dir / "123000"
    period_123000.mkdir()
    period_090000 = day_dir / "090000"
    period_090000.mkdir()

    (period_123000 / "monitor_1_diff.png").write_bytes(b"data")
    (period_123000 / "monitor_1_diff.json").write_text(
        '{"visual_description": "screen", "raw": "monitor_1_diff.png"}'
    )

    (period_090000 / "raw.flac").write_bytes(b"data")
    # Write JSONL format: metadata first, then entry
    (period_090000 / "audio.jsonl").write_text(
        '{"raw": "raw.flac"}\n{"text": "hello"}\n'
    )

    path, mime, meta = utils.get_raw_file("20240101", "123000/monitor_1_diff.json")
    assert path == "monitor_1_diff.png"
    assert mime == "image/png"
    assert meta["visual_description"] == "screen"

    path, mime, meta = utils.get_raw_file("20240101", "090000/audio.jsonl")
    assert path == "raw.flac"
    assert mime == "audio/flac"
    # JSONL format returns a list: [entry1_dict, ...]
    assert isinstance(meta, list) and len(meta) == 1
    assert meta[0]["text"] == "hello"
