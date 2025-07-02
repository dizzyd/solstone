import argparse
import asyncio
import io
import logging
import os

import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from silero_vad import load_silero_vad

from hear.audio_utils import SAMPLE_RATE, detect_speech

MODEL = "gemini-2.5-flash"#-lite-preview-06-17"

from google.genai import types

USER_PROMPT = "Please transcribe any spoken words or utterances you hear in this audio clip, accuracy is important."


def transcribe_light(client, model: str, audio_bytes: bytes) -> str:

    parts = [
        types.Part.from_text(text=USER_PROMPT),
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac"),
    ]

    contents = [types.Content(role="user", parts=parts)]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
            system_instruction="You are an expert transcriptionist. Your task is to transcribe all spoken words or utterances you hear in the audio clip into simple text. Do not include any additional commentary or formatting, if you heard no words or utterances then return a single newline.",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
        ),
    )
    result = response.text.strip()
    logging.info("Transcription result: %s", result)
    return result


whisper_model = None


def transcribe_whisper(audio_bytes: bytes) -> str:
    """Transcribe using faster-whisper."""
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    with io.BytesIO(audio_bytes) as buf:
        segments, _ = whisper_model.transcribe(buf, language="en", task="transcribe")
        text = " ".join(segment.text for segment in segments).strip()
    logging.info("Whisper result: %s", text)
    return text


async def handle_audio_message(msg: bytes, vad, stash: np.ndarray, client) -> np.ndarray:
    """Handle a single audio message from WebSocket."""
    try:
        chunk = np.frombuffer(msg, dtype=np.float32).reshape(-1, 2)
        mono = chunk.mean(axis=1)
        stash = np.concatenate((stash, mono))
        segments, stash = detect_speech(vad, "live", stash)
        for seg in segments:
            audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
            try:
                g_text = transcribe_light(
                    client,
                    MODEL,
                    buf.getvalue(),
                )
                w_text = transcribe_whisper(buf.getvalue())
                print(f"G: {g_text}\nW: {w_text}")
            except Exception as e:
                logging.error("Transcription error: %s", e)
        return stash
    except Exception as e:
        logging.error("Error processing audio chunk: %s", e)
        return stash


async def live_loop(ws_url: str, client) -> None:
    vad = load_silero_vad()
    stash = np.array([], dtype=np.float32)
    processed_seconds = 0.0
    max_retries = 5
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            logging.info(f"Connecting to WebSocket (attempt {attempt + 1}/{max_retries})")
            async with websockets.connect(ws_url) as ws:
                logging.info("WebSocket connected successfully")
                async for msg in ws:
                    stash = await handle_audio_message(msg, vad, stash, client)
                    processed_seconds += (len(msg) // 8) / SAMPLE_RATE  # Approximate calculation
                
                # If we reach here, connection closed normally
                logging.info("WebSocket connection closed normally")
                break
                
        except (websockets.exceptions.ConnectionClosedError, 
                websockets.exceptions.ConnectionClosedOK,
                ConnectionError, 
                BrokenPipeError) as e:
            logging.warning(f"WebSocket connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logging.error("Max retries reached, giving up")
                raise
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down gracefully")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                raise


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Live transcription from WebSocket")
    parser.add_argument("--ws-url", required=True, help="WebSocket URL from gemini-mic")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    client = genai.Client(api_key=api_key)

    asyncio.run(live_loop(args.ws_url, client))


if __name__ == "__main__":
    main()
