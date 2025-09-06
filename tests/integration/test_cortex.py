"""Integration test for Cortex service with cortex_client."""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.cortex_client import CortexClient

# from think.models import GPT_5_MINI  # Not needed, using string model name


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_cortex_service_with_client():
    """Test Cortex service with CortexClient for a simple OpenAI request."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Create CortexClient and spawn agent
        client = CortexClient(journal_path=journal_path)

        # Collect all events
        events = []

        async def handle_event(event):
            events.append(event)
            print(f"Event: {event.get('event')} - {event}")

        # Spawn agent with simple math question
        agent_id = await client.spawn(
            prompt="what is 2+2, just return the number nothing else",
            backend="openai",
            persona="default",
            config={
                "model": "gpt-4o-mini",  # Use cheap model for testing
                "max_tokens": 100,
                "disable_mcp": True,  # Disable MCP for simple test
            },
        )

        # Wait for completion with timeout
        result = await asyncio.wait_for(
            client.wait_for_completion(agent_id, timeout=30), timeout=35
        )

        # Verify result contains "4"
        assert "4" in result, f"Expected '4' in result, got: {result}"

        # Get all events for verification
        all_events = await client.get_agent_events(agent_id)

        # Verify event structure
        assert (
            len(all_events) >= 3
        ), f"Expected at least 3 events, got {len(all_events)}"

        # Check request event (first)
        request_event = all_events[0]
        assert request_event["event"] == "request"
        assert (
            request_event["prompt"]
            == "what is 2+2, just return the number nothing else"
        )
        assert request_event["backend"] == "openai"

        # Find start event
        start_events = [e for e in all_events if e.get("event") == "start"]
        assert len(start_events) > 0, "No start event found"
        start_event = start_events[0]
        assert start_event["model"] == "gpt-4o-mini"

        # Find finish event
        finish_events = [e for e in all_events if e.get("event") == "finish"]
        assert len(finish_events) > 0, "No finish event found"
        finish_event = finish_events[0]
        assert "4" in str(finish_event["result"])

        # Verify agent status (may need to wait for file rename)
        # Wait up to 5 seconds for status to change to completed
        for _ in range(10):
            status = await client.get_agent_status(agent_id)
            if status == "completed":
                break
            await asyncio.sleep(0.5)

        assert status == "completed", f"Expected 'completed' status, got: {status}"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_cortex_streaming_events():
    """Test streaming events from Cortex service."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Create CortexClient
        client = CortexClient(journal_path=journal_path)

        # Track events as they stream
        streamed_events = []
        event_types_seen = set()

        async def event_handler(event):
            streamed_events.append(event)
            event_types_seen.add(event.get("event"))
            print(f"Streamed: {event.get('event')} at {event.get('ts')}")

        # Spawn and read events
        agent_id = await client.spawn(
            prompt="What is the capital of France? Just say the city name.",
            backend="openai",
            persona="default",
            config={
                "model": "gpt-4o-mini",  # Use cheap model for testing
                "max_tokens": 100,
                "disable_mcp": True,
            },
        )

        # Read events with streaming
        await client.read_events(agent_id, event_handler, follow=True)

        # Verify we got the expected event types
        expected_events = {"request", "start", "finish"}
        assert expected_events.issubset(
            event_types_seen
        ), f"Missing events. Expected: {expected_events}, Got: {event_types_seen}"

        # Verify finish event contains Paris
        finish_events = [e for e in streamed_events if e.get("event") == "finish"]
        assert (
            len(finish_events) == 1
        ), f"Expected 1 finish event, got {len(finish_events)}"
        assert "Paris" in finish_events[0].get(
            "result", ""
        ), "Expected 'Paris' in result"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cortex_agent_list():
    """Test listing agents through CortexClient."""
    # Use fixtures journal for this test
    journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    client = CortexClient(journal_path=journal_path)

    # List all agents
    result = await client.list_agents(limit=5, agent_type="all")

    # Verify structure
    assert "agents" in result
    assert "pagination" in result
    assert "live_count" in result
    assert "historical_count" in result

    # Check pagination structure
    pagination = result["pagination"]
    assert "limit" in pagination
    assert "offset" in pagination
    assert "total" in pagination
    assert "has_more" in pagination

    # If there are agents, verify their structure
    if result["agents"]:
        agent = result["agents"][0]
        required_fields = [
            "id",
            "status",
            "is_live",
            "persona",
            "backend",
            "prompt",
            "ts",
        ]
        for field in required_fields:
            assert field in agent, f"Missing field '{field}' in agent"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_cortex_simple_math_streaming():
    """Test Cortex service with simple math question and verify streaming."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Create CortexClient and test the exact scenario requested
        client = CortexClient(journal_path=journal_path)

        # Track events to verify streaming
        event_sequence = []
        result_value = None
        error_message = None

        async def stream_handler(event):
            event_type = event.get("event")
            event_sequence.append(event_type)
            print(f"Event: {event_type} - {event}")

            if event_type == "finish":
                nonlocal result_value
                result_value = event.get("result", "")
                print(f"Got result: {result_value}")
            elif event_type == "error":
                nonlocal error_message
                error_message = event.get("error", "Unknown error")
                print(f"Got error: {error_message}")

        # Make the exact request: "what is 2+2, just return the number nothing else"
        agent_id = await client.spawn(
            prompt="what is 2+2, just return the number nothing else",
            backend="openai",
            persona="default",
            config={
                "model": "gpt-4o-mini",  # Use cheap model for testing  # Use default model or cheap one for testing
                "max_tokens": 100,  # Minimum is 16 for OpenAI
                "disable_mcp": True,
            },
        )

        # Stream events until completion
        await client.read_events(agent_id, stream_handler, follow=True)

        # Check if we got an error
        if error_message:
            print(f"Test failed with error: {error_message}")
            print(f"Event sequence: {event_sequence}")
            pytest.fail(f"Agent failed with error: {error_message}")

        # Verify we got events in correct order
        assert "request" in event_sequence, "Missing request event"
        assert "start" in event_sequence, "Missing start event"
        assert (
            "finish" in event_sequence
        ), f"Missing finish event. Got events: {event_sequence}"

        # Verify the final event has the correct answer
        assert result_value is not None, "No result received"
        assert "4" in result_value, f"Expected '4' in result, got: {result_value}"

        print(f"Success! Got answer: {result_value}")
        print(f"Event sequence: {event_sequence}")

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_cortex_default_model():
    """Test Cortex service using default model (no model specified)."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Create CortexClient
        client = CortexClient(journal_path=journal_path)

        # Track events
        events = []

        async def event_handler(event):
            events.append(event)
            print(f"Event: {event.get('event')}")

        # Spawn with NO model specified - use backend defaults
        agent_id = await client.spawn(
            prompt="what is 2+2, just return the number nothing else",
            backend="openai",
            persona="default",
            config={
                # No model specified - use default
                "disable_mcp": True,
            },
        )

        # Read events
        await client.read_events(agent_id, event_handler, follow=True)

        # Check we got the key events
        event_types = [e.get("event") for e in events]
        assert "request" in event_types
        assert "start" in event_types
        assert "finish" in event_types or "error" in event_types

        # Find the result
        finish_events = [e for e in events if e.get("event") == "finish"]
        if finish_events:
            result = finish_events[0].get("result", "")
            print(f"Got result with default model: {result}")
            # Result should contain 4
            assert "4" in result, f"Expected '4' in result, got: {result}"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cortex_error_handling():
    """Test Cortex error handling with invalid request."""
    journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Create CortexClient
        client = CortexClient(journal_path=journal_path)

        # Spawn agent with empty prompt (should fail)
        agent_id = await client.spawn(
            prompt="",  # Empty prompt should cause error
            backend="openai",
            persona="default",
        )

        # Try to wait for completion
        with pytest.raises(RuntimeError) as exc_info:
            await asyncio.wait_for(
                client.wait_for_completion(agent_id, timeout=10), timeout=15
            )

        # Verify error message
        assert "Agent error" in str(exc_info.value)

        # Check agent status
        status = await client.get_agent_status(agent_id)
        assert status in [
            "failed",
            "completed",
        ], f"Expected error status, got: {status}"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()
