import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


async def run_main(mod, argv):
    sys.argv = argv
    await mod.main_async()


def test_agent_main(monkeypatch, tmp_path, capsys):
    agents_stub = types.ModuleType("agents")

    last_kwargs = {}

    class DummyAgent:
        def __init__(self, *a, **k):
            last_kwargs.update(k)

    class DummyRunner:
        called = False

        @staticmethod
        async def run(agent, prompt, run_config=None):
            DummyRunner.called = True
            return SimpleNamespace(final_output="ok")

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.RunConfig = lambda **k: SimpleNamespace()
    agents_stub.ModelSettings = lambda **k: SimpleNamespace()
    agents_stub.set_default_openai_key = lambda k: None

    agents_mcp_stub = types.ModuleType("agents.mcp")

    class DummyMCP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_stub.MCPServerStdio = lambda **k: DummyMCP()

    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub

    mod = importlib.import_module("think.agent")

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agent", str(task)]))

    out = capsys.readouterr().out
    assert "ok" in out
    assert DummyRunner.called
    assert last_kwargs.get("mcp_servers") is not None
