import argparse
import os
import sys
import threading
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .cluster_day import cluster_day

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "ponder_day.txt")

FLASH_MODEL = "gemini-2.5-flash-preview-05-20"
PRO_MODEL = "gemini-2.5-pro-preview-05-14"


def send_markdown(markdown: str, prompt: str, api_key: str, model: str) -> str:
    client = genai.Client(api_key=api_key)

    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed", file=sys.stderr)

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                system_instruction=prompt,
            ),
        )
        return response.text
    finally:
        done.set()
        t.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a day's clustered Markdown to Gemini for analysis."
    )
    parser.add_argument(
        "folder",
        help="Directory containing the day's folder or its parent",
    )
    parser.add_argument(
        "day",
        nargs="?",
        help="Day folder (YYYYMMDD). If omitted, derived from folder path",
    )
    parser.add_argument(
        "-f",
        "--prompt",
        default=DEFAULT_PROMPT_PATH,
        help="Prompt file to use",
    )
    parser.add_argument(
        "-p",
        "--pro",
        action="store_true",
        help="Use the gemini 2.5 pro model",
    )
    args = parser.parse_args()

    markdown, file_count = cluster_day(args.folder, args.day)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GOOGLE_API_KEY not found in environment")

    try:
        with open(args.prompt, "r") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        parser.error(f"Prompt file not found: {args.prompt}")

    model = PRO_MODEL if args.pro else FLASH_MODEL
    day = args.day if args.day else os.path.basename(os.path.normpath(args.folder))
    size_kb = len(markdown.encode("utf-8")) / 1024
    print(
        f"Prompt: {args.prompt} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB",
        file=sys.stderr,
    )

    result = send_markdown(markdown, prompt, api_key, model)
    print(result)


if __name__ == "__main__":
    main()
