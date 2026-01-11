# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for Amazon Bedrock provider with real API calls."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Default test models
BEDROCK_CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_LLAMA_8B = "meta.llama3-8b-instruct-v1:0"


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

    # Check for AWS credentials (various methods)
    has_aws_creds = (
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
    )
    region = os.getenv("AWS_REGION", "us-east-1")
    journal_path = os.getenv("JOURNAL_PATH")

    return has_aws_creds, region, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_bedrock_generate_basic_claude():
    """Test basic text generation with Bedrock Claude model."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()
    response = provider.generate(
        "What is 2+2? Reply with just the number.",
        model=BEDROCK_CLAUDE_HAIKU,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "4" in response or "four" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_bedrock_generate_with_system_instruction():
    """Test Bedrock Claude provider with system instruction."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()
    response = provider.generate(
        "Tell me about Python",
        model=BEDROCK_CLAUDE_HAIKU,
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
def test_bedrock_generate_llama():
    """Test Bedrock with Llama model."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()
    response = provider.generate(
        "What is the capital of France? Reply with just the city name.",
        model=BEDROCK_LLAMA_8B,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "paris" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.asyncio
async def test_bedrock_agenerate_basic():
    """Test async generation with Bedrock provider."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()
    response = await provider.agenerate(
        "What is 3+3? Reply with just the number.",
        model=BEDROCK_CLAUDE_HAIKU,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "6" in response or "six" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_bedrock_provider_capabilities():
    """Test that Bedrock provider reports capabilities correctly."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()

    # Bedrock supports vision (via Claude) and thinking (via Claude 3.5)
    assert provider.supports_vision() is True
    assert provider.supports_thinking() is True
    assert provider.supports_caching() is False


@pytest.mark.integration
@pytest.mark.requires_api
def test_bedrock_context_windows():
    """Test that Bedrock provider returns correct context windows."""
    has_aws_creds, region, journal_path = get_env()

    if not has_aws_creds:
        pytest.skip("AWS credentials not found")

    os.environ["AWS_REGION"] = region

    from think.providers_bedrock import BedrockProvider

    provider = BedrockProvider()

    # Claude models have 200K context
    assert (
        provider.get_context_window("anthropic.claude-3-5-sonnet-20241022-v2:0")
        == 200000
    )
    assert (
        provider.get_context_window("anthropic.claude-3-haiku-20240307-v1:0") == 200000
    )

    # Llama 3.1 models have 128K context
    assert provider.get_context_window("meta.llama3-1-70b-instruct-v1:0") == 128000

    # Llama 3 models have 8K context
    assert provider.get_context_window("meta.llama3-8b-instruct-v1:0") == 8000
