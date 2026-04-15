import argparse
import json
import re
from pathlib import Path
from typing import Any


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_chinese(value: Any) -> bool:
    if isinstance(value, str):
        return bool(CHINESE_RE.search(value))
    if isinstance(value, dict):
        return any(contains_chinese(key) or contains_chinese(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_chinese(item) for item in value)
    return False


def iter_entries(data: Any):
    if isinstance(data, dict):
        return data.items()
    if isinstance(data, list):
        return enumerate(data)
    raise TypeError(f"Unsupported top-level JSON type: {type(data).__name__}")


def print_match(entry_id: Any, entry: Any) -> None:
    if isinstance(entry, str):
        print(f"[{entry_id}] {entry}")
        return

    payload = json.dumps(entry, ensure_ascii=False, indent=2)
    print(f"[{entry_id}]")
    print(payload)


def check_json_file(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    matches = 0
    print(f"== {json_path} ==")
    for entry_id, entry in iter_entries(data):
        if contains_chinese(entry_id) or contains_chinese(entry):
            print_match(entry_id, entry)
            matches += 1

    if matches == 0:
        print("No entries with Chinese characters found.")
    print(f"Matched entries: {matches}")
    print()
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Print JSON entries that contain Chinese characters.")
    parser.add_argument("json_files", nargs="+", help="One or more JSON files to inspect.")
    args = parser.parse_args()

    total_matches = 0
    for json_file in args.json_files:
        total_matches += check_json_file(Path(json_file))

    print(f"Total matched entries: {total_matches}")


if __name__ == "__main__":
    main()
