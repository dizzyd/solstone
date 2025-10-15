import importlib
import sys
import types

import numpy as np


def test_get_buffer_and_flac(tmp_path, monkeypatch):
    stub = types.ModuleType("hear.input_detect")

    def _input_detect(*args, **kwargs):
        return None, None

    stub.input_detect = _input_detect
    sys.modules["hear.input_detect"] = stub

    cap = importlib.import_module("hear.capture")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    rec = cap.AudioRecorder()
    stereo_chunk = np.column_stack(
        (
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([0.2, 0.2], dtype=np.float32),
        )
    )
    rec.audio_queue.put(stereo_chunk)
    stereo_buf = rec.get_buffers()
    assert stereo_buf.shape == (2, 2)
    data = rec.create_flac_bytes(stereo_buf)
    assert data.startswith(b"fLaC")
    cap.save_flac(data, str(tmp_path), "t")
    files = list(tmp_path.rglob("*_t.flac"))
    assert files
    hb = tmp_path / "health" / "hear.up"
    assert hb.is_file()
