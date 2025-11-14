import importlib
import os


def test_chat_page_renders(tmp_path):
    """Test chat app page loads successfully."""
    from apps.chat.routes import chat_page

    convey = importlib.import_module("convey")
    convey.journal_root = str(tmp_path)
    with convey.app.test_request_context("/app/chat/"):
        html = chat_page()
    assert "Chat" in html or "app" in html  # Page renders with app template


def test_send_message_no_key(monkeypatch, tmp_path):
    """Test send message API endpoint returns error when API key not set."""
    from apps.chat.routes import send_message

    convey = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with convey.app.test_request_context(
        "/app/chat/api/send",
        method="POST",
        json={"message": "hi", "backend": "google"},
    ):
        resp = send_message()
    assert resp.status_code == 500
    assert resp.json == {"error": "GOOGLE_API_KEY not set"}


def test_send_message_success(monkeypatch, tmp_path):
    """Test send message spawns agent successfully."""
    from apps.chat.routes import send_message

    convey = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    # Create the agents directory that spawn_agent expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    def dummy_spawn_agent(prompt, persona="default", backend="openai", config=None):
        dummy_spawn_agent.called = (prompt, persona, backend, config)
        # Return agent_id directly (as the real spawn_agent does)
        return "test_agent_123"

    monkeypatch.setattr("convey.utils.spawn_agent", dummy_spawn_agent)

    with convey.app.test_request_context(
        "/app/chat/api/send",
        method="POST",
        json={"message": "hi", "backend": "google"},
    ):
        resp = send_message()
    assert resp.json == {"agent_id": "test_agent_123"}
    assert dummy_spawn_agent.called[0] == "hi"
    assert dummy_spawn_agent.called[2] == "google"


def test_send_message_openai(monkeypatch, tmp_path):
    """Test send message with OpenAI backend."""
    from apps.chat.routes import send_message

    convey = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # Create the agents directory that spawn_agent expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    called = {}

    def dummy_spawn_agent(prompt, persona="default", backend="openai", config=None):
        called["backend"] = backend
        called["persona"] = persona
        called["config"] = config
        # Return agent_id directly (as the real spawn_agent does)
        return "test_agent_456"

    monkeypatch.setattr("convey.utils.spawn_agent", dummy_spawn_agent)

    with convey.app.test_request_context(
        "/app/chat/api/send", method="POST", json={"message": "hi", "backend": "openai"}
    ):
        resp = send_message()
    assert resp.json["agent_id"] == "test_agent_456"
    assert called["backend"] == "openai"


def test_send_message_anthropic(monkeypatch, tmp_path):
    """Test send message with Anthropic backend."""
    from apps.chat.routes import send_message

    convey = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    # Create the agents directory that spawn_agent expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    called = {}

    def dummy_spawn_agent(prompt, persona="default", backend="openai", config=None):
        called["backend"] = backend
        called["persona"] = persona
        called["config"] = config
        # Return agent_id directly (as the real spawn_agent does)
        return "test_agent_789"

    monkeypatch.setattr("convey.utils.spawn_agent", dummy_spawn_agent)

    with convey.app.test_request_context(
        "/app/chat/api/send",
        method="POST",
        json={"message": "hi", "backend": "anthropic"},
    ):
        resp = send_message()
    assert resp.json["agent_id"] == "test_agent_789"
    assert called["backend"] == "anthropic"


def test_history_and_clear(monkeypatch):
    """Test history and clear API endpoints."""
    from apps.chat.routes import chat_history, clear_history

    convey = importlib.import_module("convey")

    with convey.app.test_request_context("/app/chat/api/history"):
        resp = chat_history()
    assert resp.json == {"history": []}

    with convey.app.test_request_context("/app/chat/api/clear", method="POST"):
        resp = clear_history()
    assert resp.json == {"ok": True}
