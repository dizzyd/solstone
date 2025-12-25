# Screen Description Categories

This directory contains category prompts and formatters for vision analysis of screencast frames.

## Adding a New Category

Each category requires 2-3 files:

### 1. `<category>.json` (required)

Metadata specifying the output format:

```json
{
  "output": "markdown"
}
```

Set `"output": "json"` if the prompt produces structured JSON.

### 2. `<category>.txt` (required)

The vision prompt template sent to the model. Should instruct the model to:
- Analyze the screenshot for this specific category
- Return content in the format specified by `.json` (markdown or JSON)

### 3. `<category>.py` (optional)

Custom formatter for rich markdown output. If not provided, default formatting applies:
- Markdown content: displayed with category header
- JSON content: displayed in a code block

To add a custom formatter, create a `format` function:

```python
def format(content: Any, context: dict) -> str:
    """Format category content to markdown.

    Args:
        content: The category content (str for markdown, dict for JSON)
        context: Dict with:
            - frame: Full frame dict from JSONL
            - file_path: Path to JSONL file
            - timestamp_str: Formatted time like "14:30:22"

    Returns:
        Formatted markdown string (empty string to skip)
    """
    # Your formatting logic here
    return "**Header:**\n\nFormatted content..."
```

## Current Categories

| Category | Output | Formatter | Description |
|----------|--------|-----------|-------------|
| meeting | json | ✓ | Video conferencing with participants |
| messaging | markdown | - | Chat and email apps |
| browsing | markdown | - | Web browsing content |
| reading | markdown | - | Documents and PDFs |
| productivity | markdown | - | Spreadsheets, calendars, etc. |

## How It Works

1. `observe/describe.py` runs initial categorization to identify primary/secondary categories
2. For categories with prompts here, a follow-up request extracts detailed content
3. Results are stored in JSONL under the category name (e.g., `"meeting": {...}`)
4. `observe/screen.py` formats JSONL to markdown, using custom formatters when available
