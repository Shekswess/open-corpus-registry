<p align="center">
  <img
    src="templates/assets/icon.png"
    alt="Open Corpus Registry logo"
    width="220"
  />
</p>

<h1 align="center">Open Corpus Registry</h1>

<p align="center">
  An open catalog of datasets used to train, adapt, and align Large Language Models (LLMs).
</p>

---

## Why Open Corpus Registry

Data is the real bottleneck of modern LLMs. As Ilya Sutskever famously put it, *“Data is the fossil fuel of AI.”* While models, architectures, and infrastructure evolve rapidly, high-quality training data remains scarce, fragmented, and difficult to navigate.

Today, many of the most impactful datasets live across papers, README files, blog posts, and repositories, often without clear structure around *how* they are used in practice. Discovering reliable datasets for pretraining, mid-training, or post-training frequently requires deep domain knowledge or significant manual exploration.

Open Corpus Registry exists to close this gap. It provides a transparent, open view into datasets that are already being used successfully, making the LLM data landscape easier to explore, compare, and reuse.

Open Corpus Registry is a curated navigation layer over open-source datasets actively used across the LLM training lifecycle. It is not a dataset hosting platform or a replacement for Hugging Face, but a signal-focused index designed to make dataset discovery faster and more intentional.

---

## What You’ll Find Here

The registry currently catalogs **300+ open datasets** used for:
- foundation model pretraining  
- continued / mid-training  
- post-training, instruction tuning, and alignment  

Each entry focuses on practical metadata such as training stage, data nature (real, synthetic, mixed), scale (when available), popularity signals, licensing, and direct links to the original source.

The datasets come from widely trusted open research and engineering efforts, including teams and organizations such as:
- [Hugging Face](https://huggingface.co/datasets)  
- [AllenAI](https://x.com/allen_ai)  
- [Nous Research](https://x.com/NousResearch)  
- [NVIDIA Research](https://x.com/NVIDIAAI)  
- [Google DeepMind](https://x.com/GoogleDeepMind)  
- [OpenAI (open datasets and research artifacts)](https://x.com/OpenAI)  
---

## Who This Is For

Open Corpus Registry is designed for researchers, engineers, educators, and practitioners who want to work with LLM data more deliberately — whether that means training smaller models, extending existing ones, studying alignment techniques, or simply understanding what datasets are commonly used in the field.

---

## Open and Community-Driven

This project is fully open and intentionally simple.  
Contributions, corrections, and dataset suggestions are welcome.

If you know a dataset that should be included, feel free to open an issue or submit a pull request. The registry is a living resource and will continue to evolve with the community.

---

## Disclaimer

All datasets listed in Open Corpus Registry remain governed by their original licenses and terms. This project does not host dataset files; it only indexes and references publicly available sources.

---

## Quick start

### Install `uv` (recommended)

`uv` is a fast Python package manager used by this repo for reproducible installs via `uv.lock`.

- macOS / Linux:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Windows (PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Then install dependencies:

```bash
uv sync
```

### View the registry (web UI)

Serve the repo root:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/templates/`.

## Add a dataset (pipeline)

OCR treats [`data/datasets_all.jsonl`](data/datasets_all.jsonl) as the ground-truth registry.

1. Add the Hugging Face dataset ID(s) to [`data/list_datasets.json`](data/list_datasets.json) (`{"datasets":[...]}`).
2. Run the pipeline:
   ```bash
   uv run src/update_datasets_pipeline.py
   ```
3. The registry is written/updated at [`data/datasets_all.jsonl`](data/datasets_all.jsonl).

### Common options

- `--skip-llm`: heuristics only (no LLM calls)
- `--refresh-readmes`: re-download dataset READMEs
- `--metadata none|new|all`: refresh Hugging Face metadata for none, just new rows, or all rows
- `--llm-force`: re-run LLM enrichment even if cached
- `--prune`: drop datasets not present in `data/list_datasets.json`
- `--dry-run`: show what would be added without writing output
- `--output <path>`: write the merged JSONL to a different file

Examples:

```bash
uv run src/update_datasets_pipeline.py --skip-llm --metadata new
uv run src/update_datasets_pipeline.py --metadata all
uv run src/update_datasets_pipeline.py --prune --dry-run
uv run src/update_datasets_pipeline.py --output outputs/datasets_all.preview.jsonl
```

## Repo structure

```plain
.
├── .github/
│   ├── workflows/
│   │   ├── pages.yaml              # deploy web UI to GitHub Pages
│   │   └── uv-ci.yaml              # CI for pipeline
│   └── dependabot.yaml             # dependency update config
├── data/
│   ├── datasets_all.jsonl          # ground-truth registry (generated/updated)
│   ├── list_datasets.json          # input dataset ids (you edit this)
│   └── readmes/                    # cached dataset cards (generated)
├── outputs/                        # pipeline outputs/cache (generated)
├── src/
│   ├── prompts/
│   │   └── system_prompt.txt       # system prompt used for LLM enrichment
│   ├── expand_dataset_info.py      # fetch Hugging Face Hub metadata (utility)
│   ├── extract_readme_data.py      # download/read README.md + stage classification
│   ├── standardize_data.py         # LLM enrichment + schema normalization
│   └── update_datasets_pipeline.py # pipeline entrypoint (end-to-end updater)
├── templates/
│   ├── assets/
│   └── index.html                  # static UI (single-page app)
├── .pre-commit-config.yaml
├── .python-version
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```
