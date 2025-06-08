import argparse
import json
import os
from dotenv import load_dotenv


def find_broken(day_dir):
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")

    broken = []
    remove_files = []
    error_count = 0
    files_checked = 0
    repaired_count = 0
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
            remove_files.append(path)
            continue

        # Check if file is already a plain array (already repaired)
        if isinstance(data, list):
            repaired_count += 1
            #print(f"{name}: already repaired ({len(data)} entries)")
            continue

        if "text" not in data:
            print(f"{name}: missing 'text' key")
            error_count += 1
            continue

        text = data["text"]

        # if text is empty string or None, mark for removal
        if text == "" or text is None:
            #print(f"{name}: 'text' is empty/None, marking for removal")
            remove_files.append(path)
            continue

        if not isinstance(text, str):
            # text is already an array - needs repair to move to root
            if isinstance(text, list):
                print(f"{name}: needs repair (text is already array with {len(text)} entries)")
                broken.append((path, text))
            else:
                print(f"{name}: 'text' is {type(text).__name__}, expected str or list from {data}")
                error_count += 1
            continue

        stripped = text.strip()
        if not stripped.startswith("["):
            print(f"{name}: 'text' string does not start with '[', got '{stripped[:30]!r}...'")
            error_count += 1
            continue

        try:
            arr = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"{name}: failed to decode text as JSON ({e}), marking for removal")
            remove_files.append(path)
            continue

        if not isinstance(arr, list):
            print(f"{name}: decoded text is not a list, got {type(arr).__name__} from '{stripped[:30]!r}...'")
            error_count += 1
            continue

        print(f"{name}: needs repair ({len(arr)} entries)")
        broken.append((path, arr))

    return broken, remove_files, error_count, files_checked, repaired_count


def repair_files(entries, remove_files):
    for path, arr in entries:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(arr, f)
        print(f"Repaired {os.path.basename(path)}")
    
    for path in remove_files:
        os.remove(path)
        print(f"Removed file {os.path.basename(path)}")


def main():
    # Load environment for API key
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Repair escaped JSON in audio transcription files")
    parser.add_argument("day_path", help="Path to day directory (e.g., ~/Pictures/Eri/20250522/)")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-accept repairs without prompting")
    args = parser.parse_args()

    # Expand user path if needed
    day_path = os.path.expanduser(args.day_path)

    try:
        broken, remove_files, error_count, files_checked, repaired_count = find_broken(day_path)
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"\nChecked {files_checked} audio JSON file(s).")
    
    if error_count > 0:
        print(f"Found {error_count} file(s) with errors.")

    if not broken and not remove_files:
        print(f"No files requiring repair, {repaired_count} already repaired.")
        return

    if broken:
        print(f"Found {len(broken)} file(s) needing standard repair.")
    if remove_files:
        print(f"Found {len(remove_files)} file(s) to remove.")
    
    if not args.yes:
        proceed = input("Continue with repair? [y/N]: ").strip().lower()
        if proceed != "y":
            print("Aborted")
            return

    repair_files(broken, remove_files)


if __name__ == "__main__":
    main()

