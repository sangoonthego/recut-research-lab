# Recut Research Lab 01: Advertising Consistency Classification

[![CI](https://github.com/recut-ai/recut-research-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset License](https://img.shields.io/badge/dataset%20license-CC%20BY--NC--SA%204.0-orange.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A research-first Python and Jupyter workspace for **Advertising Consistency Classification**, empirical model evaluation, systematic error analysis, and domain transfer preparation for the Recut ad-generation product.

---

## 1. Research Context

- **Task**: Binary consistency classification between Japanese advertising headlines (`title`) and landing page text (`lp_text`).
- **Target Labels**: `consistent` vs. `inconsistent`.
- **Primary Benchmark**: CyberAgent AdTEC (`ad-consistency` configuration) from NAACL 2025: *AdTEC: A Unified Benchmark for Evaluating Text Quality in Search Engine Advertising*.
- **Domain**: Search engine advertising in Japanese.
- **Critical Distinction**: This benchmark evaluates thematic and claim consistency between an advertisement and its destination page. It is **not** a three-class factual claim-verification task, and does not evaluate clinical trials or skincare medical efficacy claims.

---

## 2. Repository Layout

```text
recut-research-lab/
├── .github/workflows/ci.yml    # Continuous integration workflow
├── configs/                    # Experiment and baseline configuration JSONs
│   ├── baseline.json           # TF-IDF + Logistic Regression configuration
│   └── experiment.example.json # Template for baseline experiments
├── data/                       # Dataset directory (raw data excluded from Git)
│   ├── README.md
│   ├── raw/                    # Downloaded benchmark splits (adtec_consistency/)
│   ├── processed/              # Processed / cached representations
│   └── recut_eval/             # Gold-standard evaluation splits for domain transfer
├── notebooks/                  # Jupyter research notebooks
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline_comparison.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_feature_interaction.ipynb
│   ├── 05_evidence_selection.ipynb
│   └── 06_domain_transfer.ipynb
├── src/recut_lab/              # Core reusable Python package
│   ├── __init__.py
│   ├── data.py                 # Split loading, schema validation, persistence
│   ├── audit.py                # Quality audit, duplicates, lengths, cross-split overlap
│   ├── preprocessing.py        # Japanese text normalization, pair combination
│   ├── features.py             # Character n-gram TF-IDF, lexical overlap extraction
│   ├── baseline.py             # Majority and TF-IDF LogReg baseline classifiers
│   ├── evaluation.py           # Multi-metric evaluation, latency benchmark
│   ├── analysis.py             # Error extraction, slice analysis, feature weights
│   └── experiment.py           # Experiment tracking, seeds, run metadata
├── scripts/                    # Command-line entry points
│   ├── download_adtec.py       # Download and verify official AdTEC splits
│   ├── run_audit.py            # Run data audit and export Markdown/JSON reports
│   ├── train_baseline.py       # Train baseline on train, evaluate on valid
│   └── evaluate_model.py       # Evaluate checkpoint with frozen-test guardrail
├── experiments/                # Local experiment runs and tracked parameters
│   ├── README.md
│   └── runs/                   # Timestamped run directories
├── models/                     # Serialized model checkpoints (.pkl)
├── reports/                    # Generated analytical outputs
│   ├── README.md
│   └── generated/              # Audit reports, metric summaries, figures
├── docs/                       # Research documentation
│   ├── research_questions.md   # Research questions and formal hypotheses
│   ├── dataset_card.md         # AdTEC dataset card and licensing guardrails
│   ├── experiment_protocol.md  # Split discipline, metrics, leakage controls
│   ├── error_taxonomy.md       # Error categories (E1–E6)
│   └── research_log.md         # Chronological research engineering log
├── tests/                      # Automated test suite
│   ├── test_data.py
│   ├── test_audit.py
│   ├── test_baseline.py
│   └── test_evaluation.py
├── .env.example                # Template environment variables
├── .gitignore                  # Git exclusions for datasets and binaries
├── pyproject.toml              # Build configuration and dependency declarations
├── LICENSE                     # Software license (MIT)
└── README.md                   # Project documentation
```

---

## 3. Setup and Installation

Compatible with Python 3.11+.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Linux & macOS (Bash / Zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 4. Research Workflow

### Step 1: Download Benchmark Splits
Acquires the official AdTEC `ad-consistency` split from Hugging Face Hub and records SHA-256 checksums in `metadata.json`:
```bash
python scripts/download_adtec.py
```

### Step 2: Run Dataset Quality Audit
Computes sample counts, null values, exact and field-level duplicates, text length statistics (median, IQR), and cross-split overlap to detect potential data leakage:
```bash
python scripts/run_audit.py
```
Outputs are saved to `reports/generated/audit_report.json` and `reports/generated/audit_report.md`.

### Step 3: Train Baseline Models
Fits baseline models exclusively on the official `train` split and evaluates performance on the `valid` split:
```bash
# Character n-gram TF-IDF + Logistic Regression
python scripts/train_baseline.py --config configs/baseline.json

# Majority-class baseline (empirical lower bound)
python scripts/train_baseline.py --model-type majority
```
Metrics and misclassification predictions are persisted locally under `experiments/runs/` and `reports/generated/`.

### Step 4: Evaluate Checkpoints (With Split Discipline Guardrail)
Evaluate a trained checkpoint against validation:
```bash
python scripts/evaluate_model.py --model models/tfidf_logreg_baseline.pkl --split valid
```
> [!IMPORTANT]
> The official test set (`test`) is **strictly frozen**. The evaluation CLI rejects evaluation against the test set unless explicitly confirmed with `--confirm-test`.

### Step 5: Research Notebooks
Launch Jupyter to explore audit visualizations and baseline comparisons:
```bash
jupyter lab notebooks/
```
- `01_dataset_audit.ipynb`: Visual audit of distributions, text lengths, and cross-split leakage.
- `02_baseline_comparison.ipynb`: Comparative baseline evaluation and error slices.
- `03_error_analysis.ipynb` through `06_domain_transfer.ipynb`: Protocol skeletons for downstream research phases.

---

## 5. Verification & Testing

Execute unit tests and source validation:
```bash
# Run test suite
python -m pytest -q

# Verify source compilation
python -m compileall -q src scripts
```

---

## 6. Licensing & Ethical Guidelines

1. **Tooling & Code**: The software and scaffolding in this repository are licensed under the [MIT License](LICENSE).
2. **Benchmark Dataset**: The CyberAgent AdTEC dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is strictly restricted to non-commercial research use.
3. **Data Protection**: Raw dataset files, processed text, model checkpoints, and generated run logs are excluded from Git commits via `.gitignore`.

