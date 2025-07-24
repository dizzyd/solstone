import importlib
import os
import time


def test_check_health(tmp_path):
    mod = importlib.import_module("think.supervisor")
    health = tmp_path / "health"
    health.mkdir()
    for name in ("see.up", "hear.up"):
        (health / name).write_text("x")
    assert mod.check_health(str(tmp_path), threshold=90) == []

    old = time.time() - 100
    for hb in health.iterdir():
        os.utime(hb, (old, old))
    stale = mod.check_health(str(tmp_path), threshold=90)
    assert sorted(stale) == ["hear", "see"]


def test_send_notification(monkeypatch):
    mod = importlib.import_module("think.supervisor")
    called = []

    def fake_run(cmd, check=False):
        called.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.send_notification("msg", command="notify-send")
    assert called


def test_start_runners(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")

    started = []

    class DummyProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=False):
        started.append(
            (cmd, getattr(stdout, "name", None), getattr(stderr, "name", None))
        )
        return DummyProc()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    procs = mod.start_runners(str(tmp_path))
    log_path = tmp_path / "health" / "supervisor.log"
    assert log_path.is_file()
    assert len(procs) == 2
    assert any("hear.runner" in c[0] for c in started)
    assert any("see.runner" in c[0] for c in started)
