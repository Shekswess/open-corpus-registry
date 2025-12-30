"""Expand dataset information by fetching metadata from Hugging Face Hub."""

import json
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import dataset_info

INPUT_PATH = Path("data/datasets_all.jsonl")
OUTPUT_PATH = Path("data/datasets_all.jsonl")
SLEEP_SECONDS = 0.05
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3


def serialize_dt(value):
    """Serialize datetime-like values to strings.

    Args:
        value: Input value to serialize.

    Returns:
        ISO-8601 string for datetimes, stringified value otherwise.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def main():
    """Update dataset rows with Hugging Face Hub metadata.

    Returns:
        None.
    """
    lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    rows = []
    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        dataset_id = row.get("dataset_id")
        if not dataset_id:
            rows.append(row)
            continue
        info = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                info = dataset_info(dataset_id, timeout=REQUEST_TIMEOUT)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {dataset_id} (attempt {attempt}/{MAX_RETRIES}): {exc}")
                if attempt == MAX_RETRIES:
                    rows.append(row)
                    info = None
                else:
                    time.sleep(1)
        if info is None:
            continue

        card = info.cardData or {}
        row.update(
            {
                "hf_id": info.id,
                "downloads": info.downloads,
                "likes": info.likes,
                "license": card.get("license"),
                "languages": card.get("language"),
                "task_categories": card.get("task_categories"),
                "created_at": serialize_dt(info.created_at),
                "last_modified": serialize_dt(info.last_modified),
                "citation": card.get("citation"),
            }
        )
        rows.append(row)
        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
