# Observe Module

Multimodal capture and AI-powered analysis of desktop activity.

## Commands

| Command | Purpose |
|---------|---------|
| `observer` | Screen and audio capture (auto-detects platform) |
| `observe-gnome` | Screen and audio capture on Linux/GNOME (direct) |
| `observe-macos` | Screen and audio capture on macOS (direct) |
| `observe-transcribe` | Audio transcription with speaker diarization |
| `observe-describe` | Visual analysis of screen recordings |
| `observe-sense` | Unified observation coordination |

## Architecture

```
observer (platform-detected capture)
       ↓
   Raw media files (*.flac, *.webm, tmux_*.jsonl)
       ↓
observe-sense (coordination)
   ├── observe-transcribe → audio.jsonl
   └── observe-describe → screen.jsonl
```

## Observer State Machine

The GNOME observer operates in three modes based on activity:

```
          SCREENCAST
         ↗         ↘
    (screen)    (screen idle)
       ↑            ↓
     IDLE ←----→ TMUX
         (tmux active)
```

**Mode priority**: Screen activity always wins over tmux (user is physically present).

| Mode | Trigger | Captures |
|------|---------|----------|
| SCREENCAST | Screen active (not idle/locked/power-save) | Video + Audio |
| TMUX | Screen idle but tmux has recent client activity | Terminal content + Audio |
| IDLE | Both screen and tmux inactive | Audio only (if threshold met) |

Mode transitions trigger segment boundaries (like mute transitions do).

## Key Components

- **observer.py** - Unified entry point with platform detection
- **gnome/observer.py**, **macos/observer.py** - Platform-specific capture using native APIs
- **tmux/capture.py** - Tmux capture library (integrated into GNOME observer for fallback capture)
- **sense.py** - File watcher that dispatches transcription and description jobs
- **transcribe.py** - Audio processing with Whisper/Rev.ai and pyannote diarization
- **describe.py** - Vision analysis with Gemini, category-based prompts
- **categories/** - Category-specific prompts for screen content (see [SCREEN_CATEGORIES.md](SCREEN_CATEGORIES.md))

## Output Formats

See [JOURNAL.md](JOURNAL.md) for detailed extract schemas:
- Audio transcripts: `audio.jsonl` with speaker turns and timestamps
- Screen analysis: `screen.jsonl` with frame-by-frame categorization

## Configuration

Requires `JOURNAL_PATH` environment variable. API keys for transcription/vision services configured in `.env`.
