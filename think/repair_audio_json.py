import argparse
import json
import os


def find_broken(folder, day):
    day_dir = os.path.join(folder, day)
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")

    broken = []
    for name in sorted(os.listdir(day_dir)):
        if not name.endswith("_audio.json"):
            continue

        path = os.path.join(day_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"{name}: failed to parse JSON ({e})")
            continue

        if "text" not in data:
            print(f"{name}: missing 'text' key")
            continue

        text = data["text"]
        if not isinstance(text, str):
            print(f"{name}: already correct - 'text' is {type(text).__name__}")
            continue

        stripped = text.strip()
        if not stripped.startswith("["):
            print(f"{name}: 'text' string does not start with '['")
            continue

        try:
            arr = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"{name}: failed to decode text as JSON ({e})")
            continue

        if not isinstance(arr, list):
            print(f"{name}: decoded text is not a list")
            continue

        print(f"{name}: needs repair ({len(arr)} entries)")
        broken.append((path, arr))

    return broken


def repair_files(entries):
    for path, arr in entries:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": arr}, f, indent=2)
        print(f"Repaired {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser(description="Repair escaped JSON in audio transcription files")
    parser.add_argument("folder", help="Base directory containing day folders")
    parser.add_argument("day", help="Day folder (YYYYMMDD)")
    args = parser.parse_args()

    try:
        broken = find_broken(args.folder, args.day)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not broken:
        print("No files requiring repair.")
        return

    print(f"Found {len(broken)} file(s) needing repair.")
    proceed = input("Continue with repair? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Aborted")
        return

    repair_files(broken)


if __name__ == "__main__":
    main()

