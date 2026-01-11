# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for DigitalOcean Gradient provider with real API calls."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import DO_LLAMA_8B


def get_env():
    """Load environment from fixtures/.env or main .env file."""
    root = Path(__file__).parent.parent.parent

    # Try fixtures/.env first, then main .env
    fixtures_env = root / "fixtures" / ".env"
    main_env = root / ".env"

    if fixtures_env.exists():
        load_dotenv(fixtures_env, override=True)
    elif main_env.exists():
        load_dotenv(main_env, override=True)

    api_key = os.getenv("DO_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_digitalocean_generate_basic():
    """Test basic text generation with DigitalOcean provider."""
    api_key, journal_path = get_env()

    if not api_key:
        pytest.skip("DO_API_KEY not found in fixtures/.env file")

    os.environ["DO_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_digitalocean import DigitalOceanProvider

    provider = DigitalOceanProvider()
    response = provider.generate(
        "What is 2+2? Reply with just the number.",
        model=DO_LLAMA_8B,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "4" in response or "four" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_digitalocean_generate_with_system_instruction():
    """Test DigitalOcean provider with system instruction."""
    api_key, journal_path = get_env()

    if not api_key:
        pytest.skip("DO_API_KEY not found in fixtures/.env file")

    os.environ["DO_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_digitalocean import DigitalOceanProvider

    provider = DigitalOceanProvider()
    response = provider.generate(
        "Tell me about Python",
        model=DO_LLAMA_8B,
        system_instruction="You are a helpful assistant. Keep responses under 50 words.",
        temperature=0.3,
        max_output_tokens=500,
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "python" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_digitalocean_generate_json_output():
    """Test DigitalOcean provider with JSON output mode."""
    api_key, journal_path = get_env()

    if not api_key:
        pytest.skip("DO_API_KEY not found in fixtures/.env file")

    os.environ["DO_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    import json
    import re

    from think.providers_digitalocean import DigitalOceanProvider

    provider = DigitalOceanProvider()
    response = provider.generate(
        "Generate a JSON object with keys 'name' and 'age' for a person named Alice who is 30.",
        model=DO_LLAMA_8B,
        json_output=True,
        temperature=0.1,
        max_output_tokens=200,
    )

    assert response is not None
    assert isinstance(response, str)

    # Try to parse JSON - some models wrap in markdown code blocks
    json_str = response
    # Extract from ```json ... ``` blocks if present
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)

    data = json.loads(json_str)
    assert "name" in data or "Name" in data
    assert "age" in data or "Age" in data


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_digitalocean_agenerate_basic():
    """Test async generation with DigitalOcean provider."""
    api_key, journal_path = get_env()

    if not api_key:
        pytest.skip("DO_API_KEY not found in fixtures/.env file")

    os.environ["DO_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_digitalocean import DigitalOceanProvider

    provider = DigitalOceanProvider()
    response = await provider.agenerate(
        "What is 3+3? Reply with just the number.",
        model=DO_LLAMA_8B,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "6" in response or "six" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_digitalocean_provider_capabilities():
    """Test that DigitalOcean provider reports capabilities correctly."""
    api_key, journal_path = get_env()

    if not api_key:
        pytest.skip("DO_API_KEY not found in fixtures/.env file")

    os.environ["DO_API_KEY"] = api_key

    from think.providers_digitalocean import DigitalOceanProvider

    provider = DigitalOceanProvider()

    # GPT-OSS models don't support these features
    assert provider.supports_vision() is False
    assert provider.supports_thinking() is False
    assert provider.supports_caching() is False
