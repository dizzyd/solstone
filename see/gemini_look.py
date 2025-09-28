import json
from pathlib import Path

from PIL import ImageDraw

from think.models import GEMINI_FLASH, GEMINI_LITE, gemini_generate
from think.utils import PromptNotFoundError, load_prompt

# Module-level global to store the system instruction
_system_instruction = None


def initialize():
    """
    Load the system instruction for Gemini.
    Sets up the module-level global.
    """
    global _system_instruction

    # Skip if already initialized
    if _system_instruction is not None:
        return True

    # Load system instruction from external file
    try:
        prompt_content = load_prompt("gemini_look", base_dir=Path(__file__).parent)
        _system_instruction = prompt_content.text
    except PromptNotFoundError as exc:
        print(exc)
        return False

    return _system_instruction is not None


def gemini_describe_region(image, box, models=None, entities_text=None):
    """
    Crops the image using native pixel coordinates from box,
    computes normalized coordinates once for the Gemini call, and then
    sends both full image and crop to Gemini.
    Retries with a fallback model if the primary model fails.
    """
    # Ensure the module is initialized
    if _system_instruction is None:
        initialize()

    native_y_min, native_x_min, native_y_max, native_x_max = box
    im_with_box = image.copy()
    draw = ImageDraw.Draw(im_with_box)
    draw.rectangle(
        ((native_x_min, native_y_min), (native_x_max, native_y_max)),
        outline="red",
        width=3,
    )
    cropped = im_with_box.crop((native_x_min, native_y_min, native_x_max, native_y_max))

    prompt = "Here is the latest screenshot with the cropped region of interest, please return the complete JSON as instructed."
    if not entities_text:
        entities_text = ""  # Use empty string if no entities provided
    contents = [entities_text, prompt, im_with_box, cropped]

    models_to_try = (
        models
        if models is not None
        else [
            GEMINI_FLASH,
            "gemini-2.0-flash",
            GEMINI_LITE,
            "gemini-2.0-flash-lite",
        ]
    )

    for model_name in models_to_try:
        try:
            response_text = gemini_generate(
                contents=contents,
                model=model_name,
                temperature=0.5,
                max_output_tokens=8192 * 4,
                system_instruction=_system_instruction,
                json_output=True,
            )
            if not response_text:
                print(
                    f"Bad response from Gemini API with model {model_name}: empty response"
                )
            print(response_text)
            return {"result": json.loads(response_text), "model_used": model_name}
        except Exception as e:
            print(f"Error from Gemini API with model {model_name}: {e}")
            if model_name == models_to_try[-1]:  # If it's the last model in the list
                return None  # All retries failed
            # Otherwise, loop will continue to the next model
            print(
                f"Retrying with next model: {models_to_try[models_to_try.index(model_name) + 1]}"
            )
    return None
