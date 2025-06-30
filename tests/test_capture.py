import importlib
from queue import Queue

import numpy as np


def test_get_buffer_and_flac(tmp_path):
    cap = importlib.import_module("hear.capture")
    rec = cap.AudioRecorder(journal=str(tmp_path))
    q = Queue()
    q.put(np.array([0.1, -0.1], dtype=np.float32))
    buf = rec.get_buffer(q)
    assert len(buf) == 2
    data = rec.create_flac_bytes(buf)
    assert data.startswith(b"fLaC")
    rec.save_flac(data, "t")
    files = list(tmp_path.rglob("*_t.flac"))
    assert files
