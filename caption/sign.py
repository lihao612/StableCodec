import argparse
import json
from pathlib import Path
from typing import Any


PUNCTUATION_MAP = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def replace_chinese_punctuation(text: str) -> str:
    updated = text
    for source, target in PUNCTUATION_MAP.items():
        updated = updated.replace(source, target)
    return updated


def normalize_value(value: Any):
    if isinstance(value, str):
        updated = replace_chinese_punctuation(value)
        return updated, updated != value

    if isinstance(value, list):
        changed = False
        updated_items = []
        for item in value:
            updated_item, item_changed = normalize_value(item)
            updated_items.append(updated_item)
            changed = changed or item_changed
        return updated_items, changed

    if isinstance(value, dict):
        changed = False
        updated_dict = {}
        for key, item in value.items():
            updated_item, item_changed = normalize_value(item)
            updated_dict[key] = updated_item
            changed = changed or item_changed
        return updated_dict, changed

    return value, False


def process_json_file(json_path: Path, dry_run: bool) -> int:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    total_changes = 0

    if isinstance(data, dict):
        updated_data = {}
        for entry_id, entry in data.items():
            updated_entry, changed = normalize_value(entry)
            updated_data[entry_id] = updated_entry
            if changed:
                print(f"[{entry_id}]")
                print(f"before: {entry}")
                print(f"after : {updated_entry}")
                print()
                total_changes += 1
    elif isinstance(data, list):
        updated_data = []
        for entry_id, entry in enumerate(data):
            updated_entry, changed = normalize_value(entry)
            updated_data.append(updated_entry)
            if changed:
                payload_before = json.dumps(entry, ensure_ascii=False)
                payload_after = json.dumps(updated_entry, ensure_ascii=False)
                print(f"[{entry_id}]")
                print(f"before: {payload_before}")
                print(f"after : {payload_after}")
                print()
                total_changes += 1
    else:
        raise TypeError(f"Unsupported top-level JSON type: {type(data).__name__}")

    if total_changes == 0:
        print(f"== {json_path} ==")
        print("No entries with Chinese punctuation found.")
        print()
        return 0

    if not dry_run:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(updated_data, handle, ensure_ascii=False, indent=2)

    print(f"== {json_path} ==")
    print(f"Updated entries: {total_changes}")
    print("Mode: dry-run" if dry_run else "Mode: file updated")
    print()
    return total_changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Replace Chinese double/single quotes in JSON entries with ASCII quotes.'
    )
    parser.add_argument("json_files", nargs="+", help="One or more JSON files to process.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing them back to the file.",
    )
    args = parser.parse_args()

    total_changes = 0
    for json_file in args.json_files:
        total_changes += process_json_file(Path(json_file), dry_run=args.dry_run)

    print(f"Total updated entries: {total_changes}")


if __name__ == "__main__":
    main()
