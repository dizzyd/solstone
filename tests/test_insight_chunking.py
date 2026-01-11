# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for insight.py chunking functionality."""

from unittest.mock import MagicMock, patch

import pytest

from think.utils import day_path


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing chunking logic."""
    provider = MagicMock()
    provider.check_content_fits.return_value = (True, 1000, 100000)
    provider.estimate_tokens.return_value = 1000
    return provider


@pytest.fixture
def mock_config():
    """Create a mock provider config."""
    config = MagicMock()
    config.provider = "google"
    config.model = "gemini-3-flash-preview"
    return config


class TestSendMarkdownWithChunking:
    """Tests for send_markdown_with_chunking function."""

    def test_returns_early_when_no_files(self, tmp_path, monkeypatch):
        """Should return early message when no files to process."""
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        # Create empty day directory
        day_path("20240101")

        from think.insight import send_markdown_with_chunking

        with patch("think.insight.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.insight.get_provider") as mock_get:
                mock_get.return_value = MagicMock()

                result, count = send_markdown_with_chunking(
                    day="20240101",
                    prompt="test prompt",
                    api_key="test-key",
                    model="gemini-3-flash-preview",
                    insight_key="test",
                )

        assert count == 0

    def test_uses_standard_path_when_content_fits(self, tmp_path, monkeypatch):
        """Should use send_markdown directly when content fits."""
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        day_dir = day_path("20240101")

        # Create a small segment
        (day_dir / "100000_300").mkdir()
        (day_dir / "100000_300" / "audio.jsonl").write_text(
            '{"raw": "audio.flac"}\n{"text": "small content"}\n'
        )

        from think.insight import send_markdown_with_chunking

        mock_provider = MagicMock()
        mock_provider.check_content_fits.return_value = (True, 100, 100000)

        with patch("think.insight.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.insight.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.insight.send_markdown") as mock_send:
                    mock_send.return_value = "Generated insight"

                    result, count = send_markdown_with_chunking(
                        day="20240101",
                        prompt="test prompt",
                        api_key="test-key",
                        model="gemini-3-flash-preview",
                        insight_key="test",
                    )

        # Should use the standard send_markdown path
        assert mock_send.called
        assert result == "Generated insight"
        assert count == 1

    def test_chunks_when_content_too_large(self, tmp_path, monkeypatch):
        """Should chunk content when it exceeds context window."""
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        day_dir = day_path("20240101")

        # Create segments across different hours
        for hour in [9, 10, 14]:
            segment = day_dir / f"{hour:02d}0000_300"
            segment.mkdir()
            (segment / "audio.jsonl").write_text(
                f'{{"raw": "audio.flac"}}\n{{"text": "content at {hour}"}}\n'
            )

        from think.insight import send_markdown_with_chunking

        mock_provider = MagicMock()
        # First check: full content doesn't fit
        # Subsequent checks: individual hours fit
        mock_provider.check_content_fits.side_effect = [
            (False, 200000, 100000),  # Full content too large
            (True, 30000, 100000),  # Hour 9 fits
            (True, 60000, 100000),  # Hour 9+10 fits
            (False, 150000, 100000),  # Hour 9+10+14 doesn't fit
            (True, 30000, 100000),  # Hour 14 alone fits
        ]

        with patch("think.insight.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.insight.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.insight.send_markdown") as mock_send:
                    mock_send.return_value = "Chunk result"
                    with patch("think.insight.gemini_generate") as mock_gen:
                        mock_gen.return_value = "Merged result"

                        result, count = send_markdown_with_chunking(
                            day="20240101",
                            prompt="test prompt",
                            api_key="test-key",
                            model="gemini-3-flash-preview",
                            insight_key="test",
                        )

        # Should have called send_markdown for chunks and gemini_generate for merge
        assert mock_send.call_count >= 1
        assert count == 3  # 3 audio entries

    def test_no_merge_for_single_chunk(self, tmp_path, monkeypatch):
        """Should not call merge when only one chunk is needed."""
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        day_dir = day_path("20240101")

        # Create a single segment
        (day_dir / "100000_300").mkdir()
        (day_dir / "100000_300" / "audio.jsonl").write_text(
            '{"raw": "audio.flac"}\n{"text": "content"}\n'
        )

        from think.insight import send_markdown_with_chunking

        mock_provider = MagicMock()
        # Content doesn't fit initially but single hour does
        mock_provider.check_content_fits.side_effect = [
            (False, 200000, 100000),  # Full content too large
            (True, 30000, 100000),  # Single hour fits
        ]

        with patch("think.insight.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.insight.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.insight.send_markdown") as mock_send:
                    mock_send.return_value = "Single chunk result"
                    with patch("think.insight.gemini_generate") as mock_gen:
                        result, count = send_markdown_with_chunking(
                            day="20240101",
                            prompt="test prompt",
                            api_key="test-key",
                            model="gemini-3-flash-preview",
                            insight_key="test",
                        )

        # Should not call gemini_generate for merge (only one chunk)
        assert not mock_gen.called
        assert result == "Single chunk result"


class TestDynamicWindowPacking:
    """Tests for the dynamic window packing behavior."""

    def test_packs_multiple_hours_when_space_available(self, tmp_path, monkeypatch):
        """Should pack multiple consecutive hours into one batch when possible."""
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        day_dir = day_path("20240101")

        # Create three segments in consecutive hours
        for hour in [9, 10, 11]:
            segment = day_dir / f"{hour:02d}0000_300"
            segment.mkdir()
            (segment / "audio.jsonl").write_text(
                f'{{"raw": "audio.flac"}}\n{{"text": "hour {hour}"}}\n'
            )

        from think.insight import send_markdown_with_chunking

        mock_provider = MagicMock()
        # All hours fit together
        mock_provider.check_content_fits.side_effect = [
            (False, 200000, 100000),  # Full markdown doesn't fit (triggers chunking)
            (True, 20000, 100000),  # Hour 9 fits
            (True, 40000, 100000),  # Hour 9+10 fits
            (True, 60000, 100000),  # Hour 9+10+11 fits
        ]

        with patch("think.insight.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.insight.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.insight.send_markdown") as mock_send:
                    mock_send.return_value = "Combined result"
                    with patch("think.insight.gemini_generate") as mock_gen:
                        result, count = send_markdown_with_chunking(
                            day="20240101",
                            prompt="test prompt",
                            api_key="test-key",
                            model="gemini-3-flash-preview",
                            insight_key="test",
                        )

        # Should only call send_markdown once (all hours in one batch)
        assert mock_send.call_count == 1
        # Should not need merge
        assert not mock_gen.called
