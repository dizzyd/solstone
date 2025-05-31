# sunstone
Navigate Intelligently

This repository contains a collection of utilities for multimodal navigation.
The `see` package provides screenshot capture and comparison via the
`screen-watch` command. Screenshots and metadata are cached under
`~/.cache/sunstone/see` between runs. See `see/README.md` for details.

The `think` package includes helpers for analysing captured data. The
`ponder-day` command clusters a day's JSON files into Markdown and sends the
result to Gemini. Use `-f` to specify a custom prompt file and `-p` to switch
from the default flash model to the pro model.
