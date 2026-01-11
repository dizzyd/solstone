# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for importer.py progressive summarization functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


class TestCreateTranscriptSummary:
    """Tests for create_transcript_summary function."""

    def test_skips_when_no_audio_files(self, tmp_path, monkeypatch):
        """Should skip when no audio files provided."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        from think.importer import create_transcript_summary

        with patch("think.importer.gemini_generate") as mock_gen:
            create_transcript_summary(
                import_dir=tmp_path,
                audio_json_files=[],
                input_filename="test.m4a",
                timestamp="2024-01-01 10:00:00",
            )

        # Should not call gemini_generate
        assert not mock_gen.called

    def test_skips_when_no_api_key(self, tmp_path, monkeypatch):
        """Should skip when no API key set."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from think.importer import create_transcript_summary

        with patch("think.importer.gemini_generate") as mock_gen:
            create_transcript_summary(
                import_dir=tmp_path,
                audio_json_files=[tmp_path / "audio.jsonl"],
                input_filename="test.m4a",
                timestamp="2024-01-01 10:00:00",
            )

        # Should not call gemini_generate
        assert not mock_gen.called

    def test_uses_standard_path_when_content_fits(self, tmp_path, monkeypatch):
        """Should use single API call when content fits."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Create a transcript file
        audio_file = tmp_path / "audio.jsonl"
        audio_file.write_text(
            '{"raw": "audio.flac", "model": "whisper"}\n'
            '{"start": "00:00:01", "text": "test content"}\n'
        )

        from think.importer import create_transcript_summary

        mock_provider = MagicMock()
        mock_provider.check_content_fits.return_value = (True, 1000, 100000)

        with patch("think.importer.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.importer.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.importer.gemini_generate") as mock_gen:
                    mock_gen.return_value = "Summary content"
                    with patch("think.importer.load_prompt") as mock_prompt:
                        mock_prompt.return_value = MagicMock(text="Prompt template")

                        create_transcript_summary(
                            import_dir=tmp_path,
                            audio_json_files=[audio_file],
                            input_filename="test.m4a",
                            timestamp="2024-01-01 10:00:00",
                        )

        # Should call gemini_generate once (no chunking)
        assert mock_gen.call_count == 1
        # Verify summary was saved
        summary_path = tmp_path / "summary.md"
        assert summary_path.exists()

    def test_uses_progressive_summarization_when_too_large(
        self, tmp_path, monkeypatch
    ):
        """Should use progressive summarization when content exceeds limit."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Create multiple transcript files
        audio_files = []
        for i in range(3):
            audio_file = tmp_path / f"audio_{i}.jsonl"
            audio_file.write_text(
                f'{{"raw": "audio_{i}.flac", "model": "whisper"}}\n'
                f'{{"start": "00:00:01", "text": "chunk {i} content"}}\n'
            )
            audio_files.append(audio_file)

        from think.importer import create_transcript_summary

        mock_provider = MagicMock()
        # First check: content doesn't fit
        mock_provider.check_content_fits.return_value = (False, 200000, 100000)
        # Token estimates for chunking
        mock_provider.estimate_tokens.return_value = 40000  # Each chunk ~40K

        with patch("think.importer.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.importer.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.importer.gemini_generate") as mock_gen:
                    mock_gen.return_value = "Summarized content"
                    with patch("think.importer.load_prompt") as mock_prompt:
                        mock_prompt.return_value = MagicMock(text="Prompt template")

                        create_transcript_summary(
                            import_dir=tmp_path,
                            audio_json_files=audio_files,
                            input_filename="test.m4a",
                            timestamp="2024-01-01 10:00:00",
                        )

        # Should call gemini_generate multiple times (chunks + merge)
        assert mock_gen.call_count > 1

    def test_no_merge_for_single_chunk(self, tmp_path, monkeypatch):
        """Should not merge when only one chunk needed."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Create a single transcript file
        audio_file = tmp_path / "audio.jsonl"
        audio_file.write_text(
            '{"raw": "audio.flac", "model": "whisper"}\n'
            '{"start": "00:00:01", "text": "content"}\n'
        )

        from think.importer import create_transcript_summary

        mock_provider = MagicMock()
        # Content doesn't fit but only one file
        mock_provider.check_content_fits.return_value = (False, 200000, 100000)
        mock_provider.estimate_tokens.return_value = 50000  # Fits in one batch

        with patch("think.importer.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.importer.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.importer.gemini_generate") as mock_gen:
                    mock_gen.return_value = "Single batch result"
                    with patch("think.importer.load_prompt") as mock_prompt:
                        mock_prompt.return_value = MagicMock(text="Prompt template")

                        create_transcript_summary(
                            import_dir=tmp_path,
                            audio_json_files=[audio_file],
                            input_filename="test.m4a",
                            timestamp="2024-01-01 10:00:00",
                        )

        # Should call gemini_generate once (single chunk, no merge)
        assert mock_gen.call_count == 1


class TestProgressiveSummarizationBatching:
    """Tests for the batching behavior of progressive summarization."""

    def test_batches_by_token_count(self, tmp_path, monkeypatch):
        """Should batch transcripts based on estimated token count."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Create multiple transcript files
        audio_files = []
        for i in range(5):
            audio_file = tmp_path / f"audio_{i}.jsonl"
            audio_file.write_text(
                f'{{"raw": "audio_{i}.flac", "model": "whisper"}}\n'
                f'{{"start": "00:00:01", "text": "transcript {i}"}}\n'
            )
            audio_files.append(audio_file)

        from think.importer import create_transcript_summary

        mock_provider = MagicMock()
        mock_provider.check_content_fits.return_value = (False, 300000, 100000)
        # Each transcript is ~30K tokens, so 3 per batch (~90K < 100K available)
        mock_provider.estimate_tokens.return_value = 30000

        call_contexts = []

        def track_generate(*args, **kwargs):
            call_contexts.append(kwargs.get("context"))
            return "Batch result"

        with patch("think.importer.resolve_provider") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                provider="google", model="gemini-3-flash-preview"
            )
            with patch("think.importer.get_provider") as mock_get:
                mock_get.return_value = mock_provider
                with patch("think.importer.gemini_generate") as mock_gen:
                    mock_gen.side_effect = track_generate
                    with patch("think.importer.load_prompt") as mock_prompt:
                        mock_prompt.return_value = MagicMock(text="Prompt template")

                        create_transcript_summary(
                            import_dir=tmp_path,
                            audio_json_files=audio_files,
                            input_filename="test.m4a",
                            timestamp="2024-01-01 10:00:00",
                        )

        # Should have chunk calls and a merge call
        chunk_calls = [c for c in call_contexts if c and "chunk" in c]
        merge_calls = [c for c in call_contexts if c and "merge" in c]

        assert len(chunk_calls) >= 2  # At least 2 batches
        assert len(merge_calls) == 1  # One merge call
