"""Standardize enriched dataset information."""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ALLOWED_STAGE = {"pretraining", "midtraining", "post-training", "unknown"}
ALLOWED_NATURE = {"real", "synthetic", "mixed", "unknown"}
ALLOWED_CONTENT_TYPES = {
    "code",
    "instruction-following",
    "math",
    "reasoning",
    "preference",
    "conversation",
    "web",
    "books",
    "wikipedia",
    "multilingual",
    "vision",
    "audio",
    "speech",
    "safety",
    "evaluation",
    "qa",
    "tool-use",
    "other",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        Parsed JSON objects.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dicts to a JSONL file.

    Args:
        path: Output JSONL path.
        rows: Iterable of row objects.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def dataset_id_to_readme_path(readmes_root: Path, dataset_id: str) -> Path:
    """Build the README.md path for a dataset id.

    Args:
        readmes_root: Root folder containing README directories.
        dataset_id: Dataset repo id.

    Returns:
        Path to the dataset README.md file.
    """
    return readmes_root / dataset_id.replace("/", "__") / "README.md"


KEYWORDS = re.compile(
    r"\b(token|tokens|billion|million|trillion|size|bytes|gb|tb|samples|examples|instances)\b",
    re.I,
)


def select_readme_context(md: str, max_chars: int = 40_000) -> str:
    """Select relevant README text with keyword-focused snippets.

    Args:
        md: Full README markdown.
        max_chars: Max characters to return.

    Returns:
        Reduced README context string.
    """
    md = md.replace("\x00", "")
    if len(md) <= max_chars:
        return md

    lines = md.splitlines()
    head = "\n".join(lines[:260])

    hits: list[str] = []
    for i, line in enumerate(lines):
        if KEYWORDS.search(line):
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            hits.append("\n".join(lines[start:end]))
            if sum(len(x) for x in hits) > 14_000:
                break

    tail = "\n".join(lines[-120:])

    combined = "\n\n".join([head, *hits, tail])
    return combined[:max_chars]


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output text.

    Args:
        text: Raw model output, possibly fenced with backticks.

    Returns:
        Parsed JSON object.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}\s*$", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def normalize_content_types(value: Any) -> list[str]:
    """Normalize content type values into a list of strings.

    Args:
        value: Raw content_types value.

    Returns:
        Normalized list of content type strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    return [str(value).strip()]


def normalize_result(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw model result into strict schema fields.

    Args:
        obj: Raw result dictionary from the model.

    Returns:
        Normalized result dictionary.
    """
    description = str(obj.get("description", "")).strip()
    stage = str(obj.get("stage", "unknown")).strip()
    nature = str(obj.get("nature", "unknown")).strip()
    tokens = str(obj.get("tokens", "unknown")).strip()
    content_types = normalize_content_types(obj.get("content_types"))

    if stage not in ALLOWED_STAGE:
        stage = "unknown"
    if nature not in ALLOWED_NATURE:
        nature = "unknown"

    norm_ct: list[str] = []
    for ct in content_types:
        ct_norm = ct.strip()
        if not ct_norm:
            continue
        if ct_norm not in ALLOWED_CONTENT_TYPES:
            ct_norm = "other"
        norm_ct.append(ct_norm)
    norm_ct = sorted(set(norm_ct))

    evidence = obj.get("evidence", {}) if isinstance(obj.get("evidence"), dict) else {}
    norm_evidence: dict[str, list[str]] = {}
    for k in ["description", "stage", "nature", "content_types", "tokens"]:
        v = evidence.get(k, [])
        if isinstance(v, str):
            norm_evidence[k] = [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            norm_evidence[k] = [str(x).strip() for x in v if str(x).strip()][:3]
        else:
            norm_evidence[k] = []

    return {
        "description": description or "unknown",
        "stage": stage,
        "nature": nature,
        "content_types": norm_ct,
        "tokens": tokens or "unknown",
        "evidence": norm_evidence,
    }


@dataclass(frozen=True)
class Job:
    dataset_id: str
    readme_path: Path
    existing: dict[str, Any]


async def call_model(
    *,
    model: str,
    system_prompt: str,
    dataset_id: str,
    readme_text: str,
    max_retries: int,
) -> dict[str, Any]:
    """Call the LLM to enrich a dataset README.

    Args:
        model: LiteLLM model name.
        system_prompt: System prompt content.
        dataset_id: Dataset identifier.
        readme_text: README context passed to the model.
        max_retries: Max retry attempts on failure.

    Returns:
        Normalized enrichment result.
    """
    user = f"Dataset ID: {dataset_id}\n\n" f"README.md (markdown):\n" f"{readme_text}\n"

    attempt = 0
    while True:
        attempt += 1
        try:
            from litellm import acompletion

            resp = await acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            content = resp["choices"][0]["message"]["content"]
            obj = extract_json_from_text(content)
            return normalize_result(obj)
        except Exception as e:  # noqa: BLE001
            if attempt > max_retries:
                raise
            delay = min(45.0, (2**attempt) + random.random())
            sys.stderr.write(
                f"[retry] {dataset_id}: {type(e).__name__}: {e} (sleep {delay:.1f}s)\n"
            )
            await asyncio.sleep(delay)


def merge_row(
    existing: dict[str, Any],
    enriched: dict[str, Any],
    *,
    keep_existing_on_unknown: bool,
) -> dict[str, Any]:
    """Merge enriched fields into an existing dataset row.

    Args:
        existing: Existing dataset row.
        enriched: Enriched fields from the model.
        keep_existing_on_unknown: Keep existing fields when enrichment is unknown.

    Returns:
        Merged dataset row.
    """
    out = dict(existing)

    def take(key: str) -> Any:
        val = enriched.get(key, "unknown")
        if keep_existing_on_unknown and (val == "unknown" or val == []):
            return existing.get(key)
        return val

    out["description"] = take("description")
    out["stage"] = take("stage")
    out["nature"] = take("nature")

    content_types = take("content_types")
    if isinstance(content_types, list):
        out["content_types"] = (
            ", ".join(content_types)
            if content_types
            else existing.get("content_types", "")
        )
    else:
        out["content_types"] = str(content_types)

    out["tokens"] = take("tokens")
    out.pop("source", None)
    return out


async def run(args: argparse.Namespace) -> int:
    """Run the enrichment pipeline with provided arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    input_path = Path(args.input)
    readmes_root = Path(args.readmes_root)
    output_path = Path(args.output)
    cache_path = Path(args.cache)
    prompt_path = Path(args.prompt)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    rows = read_jsonl(input_path)

    if not args.dry_run:
        try:
            import litellm  # noqa: F401
        except ModuleNotFoundError as e:
            raise SystemExit(
                "litellm is not installed in this Python environment. "
                "Activate your project venv or run via `uv run`."
            ) from e

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for obj in read_jsonl(cache_path):
            did = str(obj.get("dataset_id", "")).strip()
            if did:
                cache[did] = obj

    jobs: list[Job] = []
    for row in rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        if not args.force and dataset_id in cache:
            continue
        readme_path = dataset_id_to_readme_path(readmes_root, dataset_id)
        if not readme_path.exists():
            if args.strict:
                raise FileNotFoundError(readme_path)
            sys.stderr.write(f"[skip] missing README: {dataset_id} -> {readme_path}\n")
            continue
        jobs.append(Job(dataset_id=dataset_id, readme_path=readme_path, existing=row))

    if args.dry_run:
        sys.stderr.write(
            f"Dry run: would enrich {len(jobs)} datasets (cached={len(cache)})\n"
        )
        for j in jobs[: min(10, len(jobs))]:
            sys.stderr.write(f"  - {j.dataset_id}: {j.readme_path}\n")
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_write_lock = asyncio.Lock()

    async def run_one(job: Job) -> None:
        """Enrich a single dataset and append to cache.

        Args:
            job: Dataset enrichment job.

        Returns:
            None.
        """
        async with sem:
            md = job.readme_path.read_text(encoding="utf-8", errors="replace")
            context = select_readme_context(md, max_chars=args.max_chars)
            enriched = await call_model(
                model=args.model,
                system_prompt=system_prompt,
                dataset_id=job.dataset_id,
                readme_text=context,
                max_retries=args.max_retries,
            )
            record = {
                "dataset_id": job.dataset_id,
                "enriched": enriched,
                "readme_path": str(job.readme_path),
            }
            cache[job.dataset_id] = record
            async with cache_write_lock:
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
            sys.stderr.write(f"[ok] {job.dataset_id}\n")

    if jobs:
        sys.stderr.write(
            f"Enriching {len(jobs)} datasets with {args.model} (concurrency={args.concurrency})\n"
        )
        await asyncio.gather(*(run_one(j) for j in jobs))
    else:
        sys.stderr.write("No datasets to enrich (already cached or missing README).\n")

    merged: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if dataset_id and dataset_id in cache:
            enriched = cache[dataset_id].get("enriched", {})
            if isinstance(enriched, dict):
                merged.append(
                    merge_row(
                        row,
                        enriched,
                        keep_existing_on_unknown=not args.overwrite_unknown,
                    )
                )
                continue
        merged.append(dict(row))

    write_jsonl(output_path, merged)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the enrichment script.

    Args:
        argv: CLI arguments (excluding program name).

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description="Enrich datasets_all.jsonl from local README.md files via LiteLLM."
    )
    p.add_argument(
        "--input", default="data/datasets_all.jsonl", help="Input JSONL path"
    )
    p.add_argument(
        "--output",
        default="outputs/datasets_all.enriched.jsonl",
        help="Output JSONL path",
    )
    p.add_argument(
        "--readmes-root",
        default="data/readmes",
        help="Root folder containing extracted README.md files",
    )
    p.add_argument(
        "--prompt",
        default="src/prompts/system_prompt.txt",
        help="System prompt file path",
    )
    p.add_argument(
        "--cache",
        default="outputs/enrichment_cache.jsonl",
        help="Append-only cache JSONL path",
    )
    p.add_argument(
        "--model",
        default="bedrock/us.amazon.nova-2-lite-v1:0",
        help="LiteLLM model name",
    )
    p.add_argument("--concurrency", type=int, default=3, help="Max concurrent requests")
    p.add_argument(
        "--max-chars",
        type=int,
        default=40_000,
        help="Max README characters sent to the model",
    )
    p.add_argument("--max-retries", type=int, default=3, help="Retries per dataset")
    p.add_argument("--force", action="store_true", help="Re-enrich even if cached")
    p.add_argument("--strict", action="store_true", help="Fail if README is missing")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be processed (no model calls)",
    )
    p.add_argument(
        "--overwrite-unknown",
        action="store_true",
        help="Overwrite fields even when the model returns unknown (default keeps existing values on unknown).",
    )
    return p.parse_args(argv)


def main() -> int:
    """CLI entrypoint for dataset enrichment.

    Returns:
        Process exit code.
    """
    args = parse_args(sys.argv[1:])
    if not os.getenv("AWS_REGION") and args.model.startswith("bedrock/"):
        sys.stderr.write(
            "[warn] AWS_REGION not set; Bedrock auth/region may be required.\n"
        )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
