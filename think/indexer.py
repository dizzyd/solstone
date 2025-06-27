import json
import os
import re
from typing import Dict, List, Optional, Tuple

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

TOP_KEY = "__top__"
CHROMA_DIR = "chroma"

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")


def find_day_dirs(journal: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entity_line(line: str) -> Optional[Tuple[str, str, str]]:
    cleaned = line.replace("**", "")
    match = ITEM_RE.match(cleaned)
    if not match:
        return None

    text = match.group(1).strip()
    if ":" not in text:
        return None

    etype, rest = text.split(":", 1)
    rest = rest.strip()
    if " - " in rest:
        name, desc = rest.split(" - ", 1)
    else:
        name, desc = rest, ""

    return etype.strip(), name.strip(), desc.strip()


def parse_entities(path: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    valid_types = {"Person", "Company", "Project", "Tool"}

    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not ITEM_RE.match(line.replace("**", "")):
                continue
            parsed = parse_entity_line(line)
            if not parsed:
                continue
            etype, name, desc = parsed
            if etype not in valid_types:
                continue
            items.append((etype, name, desc))

    return items


def load_cache(journal: str) -> Dict[str, dict]:
    cache_path = os.path.join(journal, "indexer.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(journal: str, cache: Dict[str, dict]) -> None:
    cache_path = os.path.join(journal, "indexer.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def build_entities(cache: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    index: Dict[str, Dict[str, dict]] = {}
    for key, info in cache.items():
        is_top = info.get("top", False)
        for etype, name, desc in info.get("entries", []):
            type_map = index.setdefault(etype, {})
            entry = type_map.setdefault(name, {"dates": [], "descriptions": {}, "top": False})
            if is_top:
                entry["top"] = True
                if desc:
                    entry["top_desc"] = desc
            else:
                if key not in entry["dates"]:
                    entry["dates"].append(key)
                if desc:
                    entry["descriptions"][key] = desc

    for type_map in index.values():
        for info in type_map.values():
            if info.get("top"):
                info["primary"] = info.get("top_desc", "")
            elif info["descriptions"]:
                latest = max(info["descriptions"].keys())
                info["primary"] = info["descriptions"].get(latest, "")
            else:
                info["primary"] = ""
            info.pop("top_desc", None)

    return index


def scan_entities(journal: str, cache) -> Dict[str, Dict[str, dict]]:
    days = find_day_dirs(journal)
    changed = False

    # handle top entities file in parent directory
    top_path = os.path.join(journal, "entities.md")
    if os.path.isfile(top_path):
        mtime = int(os.path.getmtime(top_path))
        info = cache.get(TOP_KEY)
        if info is None or info.get("mtime") != mtime:
            cache[TOP_KEY] = {
                "file": os.path.relpath(top_path, journal),
                "mtime": mtime,
                "entries": parse_entities(journal),
                "top": True,
            }
            changed = True
    elif TOP_KEY in cache:
        del cache[TOP_KEY]
        changed = True

    # remove days no longer present (ignore top key)
    for day in list(cache.keys()):
        if day == TOP_KEY:
            continue
        if day not in days:
            del cache[day]
            changed = True

    for day, path in days.items():
        md_path = os.path.join(path, "entities.md")
        if not os.path.isfile(md_path):
            if day in cache:
                del cache[day]
                changed = True
            continue

        mtime = int(os.path.getmtime(md_path))
        day_info = cache.get(day)
        if day_info is None or day_info.get("mtime") != mtime:
            entries = parse_entities(path)
            cache[day] = {
                "file": os.path.relpath(md_path, journal),
                "mtime": mtime,
                "entries": entries,
            }
            changed = True

    return changed


def get_entities(journal: str) -> Dict[str, Dict[str, dict]]:
    cache = load_cache(journal)
    if scan_entities(journal, cache):
        save_cache(journal, cache)

    return build_entities(cache)


def get_ponder_collection(journal: str):
    """Return the Chroma collection for ponder files."""
    db_path = os.path.join(journal, CHROMA_DIR)
    os.makedirs(db_path, exist_ok=True)
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = PersistentClient(path=db_path)
    return client.get_or_create_collection("ponders", embedding_function=embed)


def find_ponder_files(journal: str) -> Dict[str, str]:
    """Map relative ponder file path to full path."""
    files: Dict[str, str] = {}
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            if name.startswith("ponder_") and name.endswith(".md"):
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def scan_ponders(journal: str, cache: Dict[str, dict]) -> bool:
    """Index ponder markdown files into Chroma if they changed."""
    collection = get_ponder_collection(journal)
    p_cache = cache.setdefault("ponders", {})
    files = find_ponder_files(journal)
    changed = False

    for rel, path in files.items():
        mtime = int(os.path.getmtime(path))
        if p_cache.get(rel) != mtime:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            day = os.path.basename(os.path.dirname(path))
            collection.upsert(
                documents=[text],
                ids=[rel],
                metadatas=[{"day": day, "ponder": os.path.basename(path)}],
            )
            p_cache[rel] = mtime
            changed = True

    for rel in list(p_cache.keys()):
        if rel not in files:
            collection.delete(ids=[rel])
            del p_cache[rel]
            changed = True

    return changed


def search_ponders(journal: str, query: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Search the ponder collection and return results."""
    collection = get_ponder_collection(journal)
    res = collection.query(query_texts=[query], n_results=n_results)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    results = []
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        results.append({"id": i, "text": doc, "metadata": meta, "distance": dist})
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index ponder markdown files")
    parser.add_argument("journal", help="Path to the journal directory")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    args = parser.parse_args()

    cache = load_cache(args.journal)
    if args.rescan:
        changed = scan_entities(args.journal, cache)
        changed |= scan_ponders(args.journal, cache)
        if changed:
            save_cache(args.journal, cache)

    while True:
        try:
            query = input("search> ").strip()
        except EOFError:
            break
        if not query:
            break
        results = search_ponders(args.journal, query, 5)
        for idx, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            snippet = r["text"].splitlines()[0][:80]
            print(f"{idx}. {meta.get('day')} {meta.get('ponder')}: {snippet}")


if __name__ == "__main__":
    main()
