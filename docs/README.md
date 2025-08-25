# Project documentation

This document describes the purpose of the repository's top-level folders and the types of files you should keep in each. Use this as a quick reference when adding new components to the AI trading system.

## Top-level folders

- `src/` — Source code packages
	- Purpose: Primary Python application code for data pipelines, model definitions, trading logic, and utilities.
	- Typical files: Python packages and modules (`__init__.py`, `pipeline.py`), subpackages like `src/data`, `src/models`, `src/trading`, `src/backtest`, `src/utils`.
	- Notes: Keep public APIs, reusable functions, and lightweight adapters here. Heavy experiments and notebooks should live elsewhere.

- `data/` — Datasets and raw inputs
	- Purpose: Store raw and processed datasets used for training and backtesting.
	- Typical files: `raw/` CSV/Parquet files, `processed/` cleaned datasets, small example fixtures, `.gitkeep` for empty directories, and README(s) describing provenance and licenses.
	- Notes: Do not commit large raw datasets to git; use external storage and store pointers/manifest files instead.

- `models/` — Trained model artifacts and checkpoints
	- Purpose: Persisted model weights, serialized pipelines, and model metadata for reproducibility.
	- Typical files: checkpoints (`.pt`, `.pkl`, `.onnx`), model cards/metadata (YAML/JSON), versioned folders, README describing format and how to load.
	- Notes: Treat as binary/artifact storage; prefer an artifact registry or object store for large models.

- `backtest/` — Backtest results and archives
	- Purpose: Stores outputs from backtest runs: PnL timeseries, trade logs, performance reports, and plots.
	- Typical files: `results/` CSVs, `reports/` HTML or Markdown, plots (`.png`, `.pdf`), serialized run configs, and `archive/` subfolders.

- `notebooks/` — Research and experiment notebooks
	- Purpose: Interactive analysis, prototyping, and explorations.
	- Typical files: Jupyter notebooks (`.ipynb`) with clear narrative, small datasets or links, and companion Python scripts for reproducible steps.
	- Notes: Keep notebooks tidy and include a short README describing each notebook's purpose.

- `configs/` — Configuration files
	- Purpose: Centralized YAML/JSON/TOML configuration for experiments, backtests, training runs, and deploys.
	- Typical files: `default.yaml`, environment-specific configs, schema files, and a `README` describing configuration conventions.

- `scripts/` — Small convenience scripts and entrypoints
	- Purpose: CLI helpers and one-off scripts for common tasks (run backtest, prepare data, evaluate models).
	- Typical files: executable Python scripts (`#!/usr/bin/env python3`), shell scripts, and small orchestration helpers.

- `experiments/` — Experiment tracking and logs
	- Purpose: Local experiment outputs, hyperparameter sweeps, and experiment metadata.
	- Typical files: per-experiment folders containing logs, metrics (`metrics.json`), and links to model artifacts.
	- Notes: For scale, use experiment tracking tools (MLflow, Weights & Biases) and store pointers here.

- `tests/` — Unit and integration tests
	- Purpose: Automated tests to validate code correctness and prevent regressions.
	- Typical files: `pytest` test modules (`test_*.py`), small test fixtures, and CI config pointers.

- `logs/` — Runtime logs and monitoring
	- Purpose: Store log outputs, diagnostic traces, and rotated logs from local runs.
	- Typical files: `YYYY-MM-DD.log`, structured logs (`.jsonl`), and diagnostic summaries.

- `docs/` — Documentation and design artifacts
	- Purpose: System design docs, architecture diagrams, runbooks, and contribution guidelines.
	- Typical files: Markdown guides (`.md`), diagrams (`.drawio`, `.png`), and architecture notes. This file (`docs/README.md`) should explain the repo layout.

## Best practices

- Keep code in `src/` small and well-tested; heavy data or binary artifacts belong in `data/` or `models/` and ideally in external storage.
- Use `configs/` to avoid hardcoding parameters in code. Version control the configs used for experiments.
- Keep `notebooks/` reproducible by pairing them with scripts in `scripts/` or `src/`.
- Add README files in subfolders where domain-specific context is required (for example `data/README.md`, `models/README.md`).

If you'd like, I can also generate README stubs for each major folder (e.g. `data/README.md`, `models/README.md`) to standardize onboarding. 
