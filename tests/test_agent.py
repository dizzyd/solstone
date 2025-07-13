import importlib
import sys
import types
from types import SimpleNamespace


def test_agent_run(monkeypatch):
    agents_stub = types.ModuleType("agents")

    class DummyAgent:
        def __init__(self, *a, **k):
            pass

    class DummyRunner:
        @staticmethod
        def run_sync(*a, **k):
            return "done"

    def decorator(fn=None, **_):
        if fn is None:
            return lambda f: f
        return fn

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.function_tool = decorator
    agents_stub.RunConfig = lambda **k: SimpleNamespace()
    agents_stub.ModelSettings = lambda **k: SimpleNamespace()
    agents_stub.set_default_openai_key = lambda k: None
    sys.modules["agents"] = agents_stub

    mod = importlib.import_module("think.agent")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", "/tmp")

    monkeypatch.setattr(mod, "tool_search_ponder", lambda query: "ponder")
    monkeypatch.setattr(mod, "tool_search_occurrences", lambda query: "occ")
    monkeypatch.setattr(mod, "tool_read_markdown", lambda d, f: "md")

    monkeypatch.setattr(mod, "Agent", DummyAgent)
    monkeypatch.setattr(mod, "Runner", DummyRunner)
    monkeypatch.setattr(mod, "RunConfig", lambda **k: SimpleNamespace())
    monkeypatch.setattr(mod, "ModelSettings", lambda **k: SimpleNamespace())
    monkeypatch.setattr(mod, "set_default_openai_key", lambda k: None)

    agent, cfg = mod.build_agent("gpt-4", 100)
    result = mod.Runner.run_sync(agent, "Hello", run_config=cfg)
    assert result == "done"
