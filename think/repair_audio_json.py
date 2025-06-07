import argparse
import json
import os


def find_broken(day_dir):
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")

    broken = []
    error_count = 0
    files_checked = 0
    for name in sorted(os.listdir(day_dir)):
        if not name.endswith("_audio.json"):
            continue

        files_checked += 1
        path = os.path.join(day_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"{name}: failed to parse JSON ({e})")
            error_count += 1
            continue

        # Check if file is already a plain array (already repaired)
        if isinstance(data, list):
            continue

        if "text" not in data:
            print(f"{name}: missing 'text' key")
            error_count += 1
            continue

        text = data["text"]
        if not isinstance(text, str):
            # text is already an array - needs repair to move to root
            if isinstance(text, list):
                print(f"{name}: needs repair (text is already array with {len(text)} entries)")
                broken.append((path, text))
            else:
                print(f"{name}: 'text' is {type(text).__name__}, expected str or list")
                error_count += 1
            continue

        stripped = text.strip()
        if not stripped.startswith("["):
            print(f"{name}: 'text' string does not start with '['")
            error_count += 1
            continue

        try:
            arr = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"{name}: failed to decode text as JSON ({e})")
            error_count += 1
            continue

        if not isinstance(arr, list):
            print(f"{name}: decoded text is not a list")
            error_count += 1
            continue

        print(f"{name}: needs repair ({len(arr)} entries)")
        broken.append((path, arr))

    return broken, error_count, files_checked


def repair_files(entries):
    for path, arr in entries:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(arr, f)
        print(f"Repaired {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser(description="Repair escaped JSON in audio transcription files")
    parser.add_argument("day_path", help="Path to day directory (e.g., ~/Pictures/Eri/20250522/)")
    args = parser.parse_args()

    # Expand user path if needed
    day_path = os.path.expanduser(args.day_path)

    try:
        broken, error_count, files_checked = find_broken(day_path)
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"\nChecked {files_checked} audio JSON file(s).")
    
    if error_count > 0:
        print(f"Found {error_count} file(s) with errors.")

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

