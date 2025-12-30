"""Extract and classify dataset information from Hugging Face dataset READMEs."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import DatasetCard, HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import (
    HfHubHTTPError,
    disable_progress_bars,
    validate_repo_id,
)

STAGES = ("pretraining", "midtraining", "post-training")


@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    dataset_url: str
    stage: str  # pretraining | midtraining | post-training
    nature: str  # real | synthetic | mixed | unknown
    content_types: str  # comma-separated
    tokens: str  # best-effort string (e.g. "6T", "350B", "unknown")
    description: str
    author: str
    source: str  # json | discovered


def _normalize_repo_id(repo_id: str) -> str:
    """Normalize a repo id by trimming whitespace.

    Args:
        repo_id: Raw repository id.

    Returns:
        Normalized repository id.
    """
    return repo_id.strip()


def _sanitize_repo_id(repo_id: str) -> str:
    """Sanitize a repo id by trimming punctuation and wrappers.

    Args:
        repo_id: Raw repository id.

    Returns:
        Sanitized repository id.
    """
    repo_id = _normalize_repo_id(repo_id)
    repo_id = repo_id.strip("`\"'")
    repo_id = repo_id.strip("()[]{}<>")
    repo_id = repo_id.strip(".,;:!?")
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        owner = owner.strip("-.")
        name = name.strip("-.")
        repo_id = f"{owner}/{name}"
    return repo_id


def _is_valid_repo_id(repo_id: str) -> bool:
    """Validate a Hugging Face dataset repo id.

    Args:
        repo_id: Repository id to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        validate_repo_id(repo_id)
        return True
    except HFValidationError:
        return False


def load_dataset_ids(path: Path) -> list[str]:
    """Load dataset ids from a JSON list file.

    Args:
        path: Path to the list JSON file.

    Returns:
        Sanitized dataset ids.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [
        _sanitize_repo_id(x)
        for x in data.get("datasets", [])
        if isinstance(x, str) and x.strip()
    ]
    return ids


def _dataset_dirname(repo_id: str) -> str:
    """Convert a repo id to its README folder name.

    Args:
        repo_id: Dataset repository id.

    Returns:
        Directory name for the dataset README.
    """
    return repo_id.replace("/", "__")


def fetch_dataset_readme(repo_id: str) -> tuple[str, dict]:
    """Fetch dataset README and metadata from the Hub.

    Args:
        repo_id: Dataset repository id.

    Returns:
        Tuple of README text and card data dict.
    """
    card = DatasetCard.load(repo_id)
    text = card.text or ""
    data = card.data or {}
    return text, data


_HF_DATASET_LINK_RE = re.compile(
    r"""(?xi)
    https?://(?:www\.)?huggingface\.co/datasets/
    (?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)
    """
)
_HF_DATASET_SHORTLINK_RE = re.compile(
    r"""(?xi)
    (?:^|[\s(])
    datasets/
    (?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)
    (?=$|[)\s'".,;:!?])
    """
)
_LOAD_DATASET_RE = re.compile(
    r"""(?xi)
    load_dataset\(\s*["']
    (?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)
    ["']
    """
)


def extract_linked_dataset_ids(text: str) -> set[str]:
    """Extract linked dataset ids from README text.

    Args:
        text: README markdown text.

    Returns:
        Set of valid dataset ids.
    """
    ids: set[str] = set()
    for match in _HF_DATASET_LINK_RE.finditer(text):
        ids.add(match.group("id"))
    for match in _HF_DATASET_SHORTLINK_RE.finditer(text):
        ids.add(match.group("id"))
    for match in _LOAD_DATASET_RE.finditer(text):
        ids.add(match.group("id"))

    out: set[str] = set()
    for i in ids:
        cand = _sanitize_repo_id(i)
        if _is_valid_repo_id(cand):
            out.add(cand)
        else:
            # Try a last pass to remove trailing punctuation from markdown links.
            cand2 = cand.rstrip(".,;:!?")
            if _is_valid_repo_id(cand2):
                out.add(cand2)
    return out


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """Return True if any needle exists in the haystack.

    Args:
        haystack: Text to search.
        needles: Substrings to find.

    Returns:
        True if any substring is present.
    """
    haystack_l = haystack.lower()
    return any(n in haystack_l for n in needles)


def _keyword_regex(keyword: str) -> re.Pattern[str]:
    """Compile a regex for a keyword with boundary handling.

    Args:
        keyword: Keyword to compile.

    Returns:
        Compiled regex pattern.
    """
    kw = keyword.lower()
    # For short acronyms like "ppo"/"sft"/"dpo", require word-ish boundaries to avoid
    # accidental matches (e.g. "support" contains "ppo").
    if " " not in kw and "-" not in kw and len(kw) <= 4:
        return re.compile(rf"(?i)(?<![A-Za-z0-9_.-]){re.escape(kw)}(?![A-Za-z0-9_.-])")
    return re.compile(re.escape(kw), re.IGNORECASE)


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """Count keyword occurrences in text.

    Args:
        text: Input text.
        keywords: Keywords to match.

    Returns:
        Total number of keyword matches.
    """
    total = 0
    for kw in keywords:
        total += len(_keyword_regex(kw).findall(text))
    return total


def _extract_stage_hints(readme_text: str) -> set[str]:
    """Extract explicit stage hints from README text.

    Args:
        readme_text: README text content.

    Returns:
        Set of hinted stages.
    """
    t = readme_text.lower()
    hints: set[str] = set()
    if re.search(r"\bpre[- ]?training\b|\bpretrain(?:ing)?\b", t):
        hints.add("pretraining")
    if re.search(
        r"\bmid[- ]?training\b|\bmidtrain(?:ing)?\b|\bcontinued pre[- ]?training\b", t
    ):
        hints.add("midtraining")
    if re.search(r"\bpost[- ]?training\b|\balignment\b|\binstruction[- ]?tuning\b", t):
        hints.add("post-training")
    if re.search(
        r"(?i)(?<![A-Za-z0-9_.-])(?:sft|dpo|rlhf|rlaif|ppo)(?![A-Za-z0-9_.-])",
        readme_text,
    ):
        hints.add("post-training")
    return hints


def _stage_scores(text: str) -> dict[str, int]:
    """Score pre/mid/post-training hints in text.

    Args:
        text: Text to score.

    Returns:
        Mapping of stage to score.
    """
    post_kw = (
        "instruction tuning",
        "instruction-tuning",
        "instruction",
        "instruct",
        "chat",
        "reward model",
        "preference",
        "alignment",
        "post-training",
        "post training",
        "rlvr",
        "ultrafeedback",
        "wildchat",
        "safety",
        "jailbreak",
        "guard",
        "rlhf",
        "rlaif",
        "sft",
        "dpo",
        "ppo",
    )
    pre_kw = (
        "pretrain",
        "pre-training",
        "pre training",
        "corpus",
        "crawl",
        "common crawl",
        "web corpus",
        "dedup",
        "the-stack",
        "fineweb",
        "dolma",
        "dclm",
        "cc-",
    )
    mid_kw = (
        "midtrain",
        "mid-training",
        "mid training",
        "continued pretraining",
        "continued pre-training",
        "domain adaptation",
        "curriculum",
        "domain-specific",
        "specialized",
        "reasoning",
        "cot",
    )
    return {
        "pretraining": _count_keyword_hits(text, pre_kw),
        "midtraining": _count_keyword_hits(text, mid_kw),
        "post-training": _count_keyword_hits(text, post_kw),
    }


def classify_stage(repo_id: str, readme_text: str) -> str:
    """Classify dataset stage based on repo id and README text.

    Args:
        repo_id: Dataset repository id.
        readme_text: README text content.

    Returns:
        Stage classification string.
    """
    text = f"{repo_id}\n{readme_text}".lower()
    rid = repo_id.lower()

    # Strong repo-id signals (prefer explicit naming over keyword heuristics in README).
    if "pretrain" in rid or "pre-training" in rid:
        return "pretraining"
    if "midtrain" in rid or "mid-training" in rid:
        return "midtraining"
    if "post" in rid or "post-training" in rid or "post_training" in rid:
        return "post-training"
    if _contains_any(
        rid,
        (
            "-sft",
            "sft-",
            "instruct",
            "instruction",
            "chat",
            "dpo",
            "rlhf",
            "rlaif",
            "rlvr",
        ),
    ):
        return "post-training"

    # Explicit stage hints in README (high priority).
    hints = _extract_stage_hints(readme_text)
    if "post-training" in hints and (
        "pretraining" not in hints and "midtraining" not in hints
    ):
        return "post-training"
    if "pretraining" in hints and "post-training" not in hints:
        return "pretraining"
    if (
        "midtraining" in hints
        and "post-training" not in hints
        and "pretraining" not in hints
    ):
        return "midtraining"

    scores = _stage_scores(text)
    post_score = scores["post-training"]
    pre_score = scores["pretraining"]
    mid_score = scores["midtraining"]

    # If hints include both pre+mid (common for general corpora), bias to pretraining.
    if (
        "pretraining" in hints
        and "midtraining" in hints
        and "post-training" not in hints
    ):
        return "pretraining"

    if post_score >= pre_score + 2 and post_score >= mid_score + 2:
        return "post-training"
    if pre_score >= post_score and pre_score >= mid_score:
        return "pretraining"
    if mid_score >= post_score and mid_score >= pre_score:
        return "midtraining"
    return "midtraining"


def infer_nature(readme_text: str) -> str:
    """Infer dataset nature (real/synthetic/mixed/unknown) from README.

    Args:
        readme_text: README text content.

    Returns:
        Nature classification string.
    """
    text = readme_text.lower()
    syn = (
        "synthetic",
        "generated",
        "llm-generated",
        "model-generated",
        "distill",
        "self-instruct",
        "teacher model",
        "gpt-",
        "deepseek",
    )
    real = (
        "web",
        "common crawl",
        "github",
        "stackexchange",
        "wikipedia",
        "arxiv",
        "pubmed",
        "books",
        "papers",
        "collected",
        "scraped",
        "crawl",
    )
    has_syn = _contains_any(text, syn)
    has_real = _contains_any(text, real)
    if has_syn and has_real:
        return "mixed"
    if has_syn:
        return "synthetic"
    if has_real:
        return "real"
    return "unknown"


def infer_content_types(repo_id: str, readme_text: str, card_data: dict) -> str:
    """Infer content types from README and card tags.

    Args:
        repo_id: Dataset repository id.
        readme_text: README text content.
        card_data: Dataset card data.

    Returns:
        Comma-separated content type string.
    """
    text = f"{repo_id}\n{readme_text}".lower()
    tags = []
    if isinstance(card_data, dict):
        tags = [str(t).lower() for t in card_data.get("tags", []) if t is not None]
    joined_tags = " ".join(tags)

    categories: set[str] = set()
    if _contains_any(text, ("code",)) or "code" in joined_tags:
        categories.add("code")
    if _contains_any(text, ("math", "gsm", "aime")) or "math" in joined_tags:
        categories.add("math")
    if _contains_any(text, ("instruction", "instruct", "sft", "chat")):
        categories.add("instruction-following")
    if _contains_any(text, ("reasoning", "chain-of-thought", "cot")):
        categories.add("reasoning")
    if _contains_any(text, ("preference", "dpo", "rlhf", "reward model", "ranking")):
        categories.add("preference")
    if _contains_any(text, ("safety", "jailbreak", "guardrail", "guard")):
        categories.add("safety")

    if not categories:
        categories.add("general")

    return ", ".join(sorted(categories))


_TOKENS_RE = re.compile(
    r"""(?xi)
    (?:
        (?P<num1>\d+(?:\.\d+)?)\s*(?P<unit1>t|trillion|b|billion|m|million)\s+tokens
      |
        (?P<num2>\d[\d,]{2,})\s+tokens
    )
    """
)


def extract_tokens(readme_text: str) -> str:
    """Extract a best-effort token count from README text.

    Args:
        readme_text: README text content.

    Returns:
        Token count string or "unknown".
    """
    # Prefer dataset-size mentions ("contains/includes/comprising ... tokens") over
    # training-recipe mentions ("trained on ... tokens").
    size_positive = (
        "contains",
        "includes",
        "including",
        "comprising",
        "consists",
        "consisting",
        "total",
        "overall",
        "in total",
        "dataset",
        "corpus",
        "size",
        "amount",
        "tokens",
    )
    size_negative = (
        "trained on",
        "training",
        "we train",
        "we trained",
        "fine-tuning",
        "finetuning",
        "convergence",
        "steps",
        "epochs",
        "model",
    )

    best: tuple[int, int, float, str] | None = (
        None  # (context_score, rank, value, rendered)
    )

    def _to_rank_and_value(num: float, unit: str | None) -> tuple[int, float, str]:
        """Normalize a token count to ranking and rendered form.

        Args:
            num: Numeric token count.
            unit: Token unit (e.g. billion/million).

        Returns:
            Tuple of rank, numeric value, and rendered string.
        """
        if unit is None:
            return 0, num, f"{num:g}" if num.is_integer() else f"{num:g}"
        unit_l = unit.lower()
        if unit_l in ("t", "trillion"):
            return 3, num, f"{num:g}T"
        if unit_l in ("b", "billion"):
            return 2, num, f"{num:g}B"
        if unit_l in ("m", "million"):
            return 1, num, f"{num:g}M"
        return 0, num, f"{num:g}"

    for m in _TOKENS_RE.finditer(readme_text):
        start, end = m.span(0)
        ctx = readme_text[
            max(0, start - 120) : min(len(readme_text), end + 120)
        ].lower()
        context_score = 0
        for w in size_positive:
            if w in ctx:
                context_score += 1
        for w in size_negative:
            if w in ctx:
                context_score -= 2

        if m.group("num1") and m.group("unit1"):
            num = float(m.group("num1"))
            rank, value, rendered = _to_rank_and_value(num, m.group("unit1"))
        elif m.group("num2"):
            raw = m.group("num2").replace(",", "")
            try:
                num = float(raw)
            except ValueError:
                continue
            rank, value, rendered = _to_rank_and_value(num, None)
        else:
            continue

        cand = (context_score, rank, value, rendered)
        if best is None or cand > best:
            best = cand

    return best[3] if best else "unknown"


def extract_description(readme_text: str) -> str:
    """Extract a short description from README text.

    Args:
        readme_text: README text content.

    Returns:
        Short description string.
    """
    text = readme_text.strip()
    if not text:
        return ""
    # Drop YAML front-matter if present.
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].lstrip()

    lines = [ln.rstrip() for ln in text.splitlines()]
    paragraph: list[str] = []
    for ln in lines:
        if not ln.strip():
            if paragraph:
                break
            continue
        if ln.lstrip().startswith("#"):
            continue
        if ln.lstrip().startswith("!["):
            continue
        if "<img" in ln.lower():
            continue
        if ln.lstrip().startswith("<"):
            continue
        if ln.lstrip().startswith("[!["):
            continue
        if ln.lstrip().startswith("![]("):
            continue
        if ln.lstrip().startswith(">"):
            # Skip callouts/quotes (often license badges or notices).
            if not paragraph:
                continue
        paragraph.append(ln.strip())
        if sum(len(x) for x in paragraph) > 400:
            break
    desc = " ".join(paragraph).strip()
    desc = re.sub(r"\s+", " ", desc)
    # Strip remaining HTML tags.
    desc = re.sub(r"<[^>]+>", "", desc).strip()
    # Simplify markdown formatting.
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
    desc = desc.replace("**", "").replace("__", "").replace("`", "")
    desc = desc.lstrip("- ").strip()
    return desc


def ensure_readmes(
    repo_ids: list[str],
    readme_root: Path,
    *,
    refresh: bool = False,
) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
    """Ensure README.md files exist for the given repo ids.

    Args:
        repo_ids: Dataset repository ids.
        readme_root: Root directory for README files.
        refresh: Whether to re-download existing README files.

    Returns:
        Tuple of README texts, card data, and error messages.
    """
    readme_root.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    card_datas: dict[str, dict] = {}
    errors: dict[str, str] = {}

    api = HfApi()

    for repo_id in repo_ids:
        repo_id = _normalize_repo_id(repo_id)
        out_dir = readme_root / _dataset_dirname(repo_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "README.md"
        if out_path.exists() and not refresh:
            texts[repo_id] = out_path.read_text(encoding="utf-8")
            card_datas[repo_id] = {}
        else:
            try:
                text, card_data = fetch_dataset_readme(repo_id)
                out_path.write_text(text, encoding="utf-8")
                texts[repo_id] = text
                card_datas[repo_id] = card_data if isinstance(card_data, dict) else {}
            except Exception as e:  # noqa: BLE001 - best-effort scrape
                errors[repo_id] = f"{type(e).__name__}: {e}"
                if not out_path.exists():
                    out_path.write_text("", encoding="utf-8")
                texts[repo_id] = out_path.read_text(encoding="utf-8")
                card_datas[repo_id] = {}

        # Best-effort: cache author/metadata via API call (useful even when card is sparse)
        try:
            info = api.dataset_info(repo_id)
            if repo_id not in card_datas:
                card_datas[repo_id] = {}
            if hasattr(info, "card_data") and info.card_data is not None:
                card_datas[repo_id].setdefault(
                    "_hf_card_data", info.card_data.to_dict()
                )
            if hasattr(info, "author") and info.author:
                card_datas[repo_id].setdefault("_hf_author", str(info.author))
        except (HfHubHTTPError, HFValidationError):
            pass

    return texts, card_datas, errors


def write_markdown_tables(rows: list[DatasetRow], out_path: Path) -> None:
    """Write dataset rows into stage-grouped markdown tables.

    Args:
        rows: Dataset rows to write.
        out_path: Output markdown path.

    Returns:
        None.
    """
    df = pd.DataFrame([asdict(r) for r in rows])
    df = df[
        [
            "dataset_id",
            "dataset_url",
            "stage",
            "nature",
            "content_types",
            "tokens",
            "description",
            "author",
            "source",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    def _md_escape(value: object) -> str:
        """Escape markdown table cell values.

        Args:
            value: Cell value.

        Returns:
            Escaped string.
        """
        s = "" if value is None else str(value)
        s = s.replace("|", "\\|")
        s = s.replace("\n", " ").strip()
        return s

    def _df_to_markdown(sdf: pd.DataFrame) -> str:
        """Convert a DataFrame to markdown table text.

        Args:
            sdf: DataFrame to render.

        Returns:
            Markdown table string.
        """
        cols = list(sdf.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for _, row in sdf.iterrows():
            lines.append("| " + " | ".join(_md_escape(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    for stage in STAGES:
        sdf = df[df["stage"] == stage].copy()
        sdf = sdf.sort_values(["source", "dataset_id"])
        parts.append(f"## {stage}\n")
        parts.append(_df_to_markdown(sdf))
        parts.append("\n")
    out_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    """Run README extraction and dataset collation.

    Returns:
        Process exit code.
    """
    disable_progress_bars()
    parser = argparse.ArgumentParser(
        description="Collate HF datasets into pre/mid/post-training tables."
    )
    parser.add_argument("--input", type=Path, default=Path("data/list_datasets.json"))
    parser.add_argument("--readmes", type=Path, default=Path("data/readmes"))
    parser.add_argument(
        "--out-md", type=Path, default=Path("outputs/datasets_tables.md")
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("outputs/datasets_all.csv")
    )
    parser.add_argument(
        "--out-jsonl", type=Path, default=Path("data/datasets_all.jsonl")
    )
    parser.add_argument(
        "--out-audit-csv", type=Path, default=Path("outputs/datasets_audit.csv")
    )
    parser.add_argument("--max-discovered", type=int, default=200)
    parser.add_argument("--refresh-readmes", action="store_true")
    args = parser.parse_args()

    json_ids = load_dataset_ids(args.input)
    sources: dict[str, str] = {rid: "json" for rid in json_ids}

    all_ids: list[str] = list(json_ids)
    known_lower: set[str] = {rid.lower() for rid in all_ids}
    texts, card_datas, errors = ensure_readmes(
        all_ids, args.readmes, refresh=args.refresh_readmes
    )

    # Discover additional datasets referenced in READMEs (iterate once; avoids runaway expansion).
    discovered: list[str] = []
    for repo_id, text in texts.items():
        for found in extract_linked_dataset_ids(text):
            found_l = found.lower()
            if found_l in known_lower:
                continue
            known_lower.add(found_l)
            discovered.append(found)
            sources[found] = "discovered"
            if len(discovered) >= args.max_discovered:
                break
        if len(discovered) >= args.max_discovered:
            break

    if discovered:
        more_texts, more_card_datas, more_errors = ensure_readmes(
            discovered, args.readmes, refresh=args.refresh_readmes
        )
        texts.update(more_texts)
        card_datas.update(more_card_datas)
        errors.update(more_errors)
        all_ids.extend(discovered)

    rows: list[DatasetRow] = []
    api = HfApi()
    for repo_id in all_ids:
        readme_text = texts.get(repo_id, "")
        card_data = card_datas.get(repo_id, {})
        stage = classify_stage(repo_id, readme_text)
        nature = infer_nature(readme_text)
        content_types = infer_content_types(
            repo_id, readme_text, card_data.get("_hf_card_data", {}) or card_data
        )
        tokens = extract_tokens(readme_text)
        description = extract_description(readme_text)
        author = str(card_data.get("_hf_author") or "")
        if not author:
            try:
                author = str(api.dataset_info(repo_id).author or "")
            except (HfHubHTTPError, HFValidationError):
                author = ""
        rows.append(
            DatasetRow(
                dataset_id=repo_id,
                dataset_url=f"https://huggingface.co/datasets/{repo_id}",
                stage=stage,
                nature=nature,
                content_types=content_types,
                tokens=tokens,
                description=description,
                author=author,
                source=sources.get(repo_id, "json"),
            )
        )

    # Persist outputs
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(args.out_csv, index=False)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    write_markdown_tables(rows, args.out_md)

    if errors:
        err_path = Path("outputs/readme_errors.json")
        err_path.write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Audit stage hints vs classifier.
    stage_by_id = {r.dataset_id: r.stage for r in rows}
    tokens_by_id = {r.dataset_id: r.tokens for r in rows}
    audit_rows: list[dict] = []
    for repo_id in all_ids:
        readme_text = texts.get(repo_id, "")
        hints = sorted(_extract_stage_hints(readme_text))
        stage = stage_by_id.get(repo_id, "")
        scores = _stage_scores(f"{repo_id}\n{readme_text}".lower())
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        confidence = (
            (sorted_scores[0][1] - sorted_scores[1][1])
            if len(sorted_scores) > 1
            else sorted_scores[0][1]
        )
        mismatch = bool(hints) and stage not in hints
        audit_rows.append(
            {
                "dataset_id": repo_id,
                "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
                "stage": stage,
                "readme_stage_hints": ", ".join(hints),
                "readme_stage_hints_count": len(hints),
                "ambiguous_hints": len(hints) > 1,
                "stage_mismatch": mismatch,
                "pre_score": scores.get("pretraining", 0),
                "mid_score": scores.get("midtraining", 0),
                "post_score": scores.get("post-training", 0),
                "score_confidence": confidence,
                "tokens": tokens_by_id.get(repo_id, "unknown"),
                "source": sources.get(repo_id, "json"),
            }
        )
    args.out_audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(args.out_audit_csv, index=False)
    review_df = audit_df[
        (audit_df["stage_mismatch"])
        | (audit_df["ambiguous_hints"])
        | (audit_df["score_confidence"] <= 1)
        | (audit_df["tokens"] == "unknown")
    ].copy()
    review_df_path = Path("outputs/datasets_review_candidates.csv")
    review_df.to_csv(review_df_path, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
