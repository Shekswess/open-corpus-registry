# Open Corpus Registry (OCR)

Open catalog of datasets used to train and align LLMs across pretraining, mid-training, and post-training.

## Why Open Corpus Registry

Data is the fuel of AI. As Ilya Sutskever puts it: "Data is the fossil fuel of AI." High-quality data is often scarce, expensive, or restricted by privacy. Much of the public web has already been mined, and access to open, reliable datasets is uneven. OCR exists to make the data landscape visible and shareable: a transparent, open registry that documents what is used, where it comes from, and how it is described.

Synthetic data is part of the answer. In my SynthGenAI work, synthetic data helps expand coverage, reduce privacy risk, and support both LLMs and smaller, efficient SLMs. OCR embraces this reality by tracking real and synthetic sources together so researchers can reason about provenance, coverage, and gaps.

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
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/templates/`.

## Add a dataset (pipeline)

OCR treats [`data/datasets_all.jsonl`](data/datasets_all.jsonl) as the ground-truth registry.

1. Add the Hugging Face dataset ID(s) to [`data/list_datasets.json`](data/list_datasets.json) (`{"datasets":[...]}`).
2. Run the pipeline:
   ```bash
   python3 src/update_datasets_pipeline.py
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
python3 src/update_datasets_pipeline.py --skip-llm --metadata new
python3 src/update_datasets_pipeline.py --metadata all
python3 src/update_datasets_pipeline.py --prune --dry-run
python3 src/update_datasets_pipeline.py --output outputs/datasets_all.preview.jsonl
```

## Repo structure

```text
open-corpus-registry/
├── data/
│   ├── datasets_all.jsonl
│   ├── list_datasets.json
│   └── readmes/                    # cached dataset cards (generated)
├── src/
│   ├── update_datasets_pipeline.py # pipeline entrypoint
│   └── ...                         # helpers
├── templates/
│   ├── index.html                  # static UI (single-page app)
│   └── assets/
└── .github/
```
