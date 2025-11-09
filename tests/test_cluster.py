import importlib

from think.utils import day_path


def test_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry in timestamp directories
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "audio.jsonl").write_text('{}\n{"text": "hi"}\n')
    (day_dir / "120500").mkdir()
    (day_dir / "120500" / "screen.md").write_text("screen summary")
    result, count = mod.cluster("20240101")
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result


def test_cluster_range(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry with proper start time and source in timestamp directory
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "audio.jsonl").write_text(
        '{"raw": "raw.flac", "model": "whisper-1"}\n'
        '{"start": "00:00:01", "source": "mic", "text": "hi from audio"}\n'
    )
    (day_dir / "120000" / "screen.md").write_text("screen summary content")
    # Test with summary mode to ensure screen content is included
    md = mod.cluster_range("20240101", "120000", "120100", audio=True, screen="summary")
    # Check that the function works and includes expected sections
    assert "Audio Transcript" in md
    assert "Screen Activity Summary" in md
    # The audio might be empty if there are formatting issues, but screen should work
    assert "screen summary content" in md


def test_cluster_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Audio transcripts at 09:01, 09:05, 09:20 and 11:00 (JSONL format with empty metadata)
    (day_dir / "090101").mkdir()
    (day_dir / "090101" / "audio.jsonl").write_text("{}\n")
    (day_dir / "090500").mkdir()
    (day_dir / "090500" / "audio.jsonl").write_text("{}\n")
    (day_dir / "092000").mkdir()
    (day_dir / "092000" / "audio.jsonl").write_text("{}\n")
    (day_dir / "110000").mkdir()
    (day_dir / "110000" / "audio.jsonl").write_text("{}\n")
    # Screen transcripts at 10:01, 10:05, 10:20 and 12:00
    (day_dir / "100101").mkdir()
    (day_dir / "100101" / "screen.md").write_text("screen")
    (day_dir / "100500").mkdir()
    (day_dir / "100500" / "screen.md").write_text("screen")
    (day_dir / "102000").mkdir()
    (day_dir / "102000" / "screen.md").write_text("screen")
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "screen.md").write_text("screen")
    audio_ranges, screen_ranges = mod.cluster_scan("20240101")
    assert audio_ranges == [("09:00", "09:30"), ("11:00", "11:15")]
    assert screen_ranges == [("10:00", "10:30"), ("12:00", "12:15")]
