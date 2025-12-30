"""Pipeline to update datasets_all.jsonl with new datasets from list_datasets.json."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, dataset_info
from huggingface_hub.errors import HfHubHTTPError, HFValidationError

import extract_readme_data as readme_extract
import standardize_data as llm_enrich

DEFAULT_GROUND_TRUTH = Path("data/datasets_all.jsonl")
DEFAULT_LIST = Path("data/list_datasets.json")
DEFAULT_READMES = Path("data/readmes")
DEFAULT_PROMPT = Path("src/prompts/system_prompt.txt")
DEFAULT_CACHE = Path("outputs/enrichment_cache.jsonl")
DEFAULT_PIPELINE_DIR = Path("outputs/pipeline")

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
SLEEP_SECONDS = 0.05


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: JSONL file path.

    Returns:
        Parsed JSON objects.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to a JSONL file.

    Args:
        path: Output JSONL path.
        rows: Row dictionaries to write.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _unique_ids(ids: Iterable[str]) -> list[str]:
    """Deduplicate dataset ids while preserving order.

    Args:
        ids: Iterable of dataset ids.

    Returns:
        Ordered, case-insensitive unique ids.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for rid in ids:
        key = rid.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(rid)
    return ordered


def compute_new_ids(
    list_ids: list[str], existing_rows: list[dict[str, Any]]
) -> list[str]:
    """Compute dataset ids not yet present in existing rows.

    Args:
        list_ids: Candidate dataset ids from the list source.
        existing_rows: Existing dataset rows.

    Returns:
        Dataset ids absent from existing rows.
    """
    existing = {
        str(row.get("dataset_id", "")).strip().lower()
        for row in existing_rows
        if row.get("dataset_id")
    }
    return [rid for rid in list_ids if rid.lower() not in existing]


def serialize_dt(value: Any) -> Any:
    """Serialize datetime-like values to strings.

    Args:
        value: Value to serialize.

    Returns:
        ISO-8601 string for datetimes, stringified value otherwise.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def enrich_metadata(rows: list[dict[str, Any]], target_ids: set[str]) -> None:
    """Enrich rows with metadata from the Hugging Face Hub.

    Args:
        rows: Dataset rows to update in place.
        target_ids: Dataset ids to refresh.

    Returns:
        None.
    """
    for row in rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id or dataset_id not in target_ids:
            continue
        info = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                info = dataset_info(dataset_id, timeout=REQUEST_TIMEOUT)
                break
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] {dataset_id} (attempt {attempt}/{MAX_RETRIES}): {exc}",
                    file=sys.stderr,
                )
                if attempt == MAX_RETRIES:
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
        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)


def build_base_rows(
    dataset_ids: list[str],
    readmes_root: Path,
    *,
    refresh_readmes: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build base dataset rows from README extraction.

    Args:
        dataset_ids: Dataset ids to process.
        readmes_root: README root directory.
        refresh_readmes: Whether to re-download README files.

    Returns:
        Tuple of base rows and README error messages.
    """
    texts, card_datas, errors = readme_extract.ensure_readmes(
        dataset_ids, readmes_root, refresh=refresh_readmes
    )
    api = HfApi()
    rows: list[dict[str, Any]] = []
    for repo_id in dataset_ids:
        readme_text = texts.get(repo_id, "")
        card_data = card_datas.get(repo_id, {})
        stage = readme_extract.classify_stage(repo_id, readme_text)
        nature = readme_extract.infer_nature(readme_text)
        content_types = readme_extract.infer_content_types(
            repo_id, readme_text, card_data.get("_hf_card_data", {}) or card_data
        )
        tokens = readme_extract.extract_tokens(readme_text)
        description = readme_extract.extract_description(readme_text)
        author = str(card_data.get("_hf_author") or "")
        if not author:
            try:
                author = str(api.dataset_info(repo_id).author or "")
            except (HfHubHTTPError, HFValidationError):
                author = ""
        rows.append(
            {
                "dataset_id": repo_id,
                "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
                "stage": stage,
                "nature": nature,
                "content_types": content_types,
                "tokens": tokens,
                "description": description,
                "author": author,
            }
        )
    return rows, errors


def merge_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    list_order: list[str],
    prune: bool,
) -> list[dict[str, Any]]:
    """Merge new rows into existing rows with optional pruning.

    Args:
        existing_rows: Existing dataset rows.
        new_rows: Newly generated dataset rows.
        list_order: Ordered list of dataset ids from the list source.
        prune: Whether to drop datasets not in list_order.

    Returns:
        Merged dataset rows.
    """
    existing_by_id = {
        str(r.get("dataset_id", "")).strip().lower(): r
        for r in existing_rows
        if r.get("dataset_id")
    }
    new_by_id = {
        str(r.get("dataset_id", "")).strip().lower(): r
        for r in new_rows
        if r.get("dataset_id")
    }

    if prune:
        merged: list[dict[str, Any]] = []
        for dataset_id in list_order:
            key = dataset_id.lower()
            row = existing_by_id.get(key) or new_by_id.get(key)
            if row:
                merged.append(row)
            else:
                print(f"[WARN] Missing row for {dataset_id}", file=sys.stderr)
        return merged

    merged = list(existing_rows)
    for row in new_rows:
        key = str(row.get("dataset_id", "")).strip().lower()
        if key and key not in existing_by_id:
            merged.append(row)
    return merged


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the update pipeline.

    Args:
        argv: CLI arguments (excluding program name).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Unified pipeline to add datasets from data/list_datasets.json into data/datasets_all.jsonl."
    )
    parser.add_argument(
        "--list", type=Path, default=DEFAULT_LIST, help="Dataset list JSON path"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Ground-truth JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Final JSONL output path",
    )
    parser.add_argument(
        "--readmes-root",
        type=Path,
        default=DEFAULT_READMES,
        help="Root folder for README.md files",
    )
    parser.add_argument(
        "--prompt", type=Path, default=DEFAULT_PROMPT, help="LLM system prompt path"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="LLM enrichment cache JSONL path",
    )
    parser.add_argument(
        "--model",
        default="bedrock/us.amazon.nova-2-lite-v1:0",
        help="LiteLLM model name",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Max concurrent LLM requests"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=40_000,
        help="Max README characters sent to the model",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Retries per dataset"
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Intermediate outputs folder",
    )
    parser.add_argument(
        "--refresh-readmes", action="store_true", help="Re-download README.md files"
    )
    parser.add_argument(
        "--llm-force", action="store_true", help="Force LLM enrichment even if cached"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM enrichment step"
    )
    parser.add_argument(
        "--metadata",
        choices=("none", "new", "all"),
        default="new",
        help="Metadata refresh scope via HF API (none, new, all)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Drop datasets not present in data/list_datasets.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without writing output"
    )
    return parser.parse_args(argv)


def main() -> int:
    """Run the update pipeline.

    Returns:
        Process exit code.
    """
    args = parse_args(sys.argv[1:])

    list_ids = _unique_ids(readme_extract.load_dataset_ids(args.list))
    existing_rows = read_jsonl(args.ground_truth)
    new_ids = compute_new_ids(list_ids, existing_rows)

    print(f"[info] list ids: {len(list_ids)}", file=sys.stderr)
    print(f"[info] existing rows: {len(existing_rows)}", file=sys.stderr)
    print(f"[info] new ids: {len(new_ids)}", file=sys.stderr)

    if args.dry_run:
        if new_ids:
            for rid in new_ids[: min(10, len(new_ids))]:
                print(f"[dry-run] would add: {rid}", file=sys.stderr)
        return 0

    new_rows: list[dict[str, Any]] = []
    if new_ids:
        base_rows, errors = build_base_rows(
            new_ids, args.readmes_root, refresh_readmes=args.refresh_readmes
        )
        if errors:
            err_path = args.pipeline_dir / "readme_errors.json"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(
                json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[warn] README errors written to {err_path}", file=sys.stderr)

        base_path = args.pipeline_dir / "datasets_new.base.jsonl"
        write_jsonl(base_path, base_rows)
        new_rows = base_rows

        if not args.skip_llm:
            enriched_path = args.pipeline_dir / "datasets_new.enriched.jsonl"
            llm_args = argparse.Namespace(
                input=str(base_path),
                output=str(enriched_path),
                readmes_root=str(args.readmes_root),
                prompt=str(args.prompt),
                cache=str(args.cache),
                model=args.model,
                concurrency=args.concurrency,
                max_chars=args.max_chars,
                max_retries=args.max_retries,
                force=args.llm_force,
                strict=False,
                dry_run=False,
                overwrite_unknown=False,
            )
            asyncio.run(llm_enrich.run(llm_args))
            new_rows = read_jsonl(enriched_path)
    else:
        print("[info] no new datasets to add", file=sys.stderr)

    merged_rows = merge_rows(
        existing_rows, new_rows, list_order=list_ids, prune=args.prune
    )

    if args.metadata != "none":
        if args.metadata == "all":
            target_ids = {
                str(r.get("dataset_id", "")).strip()
                for r in merged_rows
                if r.get("dataset_id")
            }
        else:
            target_ids = {
                str(r.get("dataset_id", "")).strip()
                for r in new_rows
                if r.get("dataset_id")
            }
        if target_ids:
            print(
                f"[info] refreshing metadata for {len(target_ids)} datasets",
                file=sys.stderr,
            )
            enrich_metadata(merged_rows, target_ids)

    write_jsonl(args.output, merged_rows)
    print(f"[info] wrote {len(merged_rows)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
