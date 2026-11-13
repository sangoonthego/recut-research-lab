# Research Log

## Project: Recut Research Lab 01
**Topic**: Advertising Consistency Classification  
**Target Benchmark**: CyberAgent AdTEC `ad-consistency`  
**License Context**: Tooling (MIT), Benchmark (CC BY-NC-SA 4.0)  

---

### Entry 001 — Repository Inception & Environment Scaffolding
- **Date**: 2026-11-13
- **Author**: Senior Machine Learning Engineer / Research Software Architect
- **Action**:
  - Inspected initial empty workspace.
  - Initialized src-layout architecture: `src/recut_lab/` containing data loading, auditing, preprocessing, features, baseline modeling, evaluation, analysis, and experiment tracking.
  - Created root packaging configuration `pyproject.toml` supporting Python 3.11+.
  - Added strict `.gitignore` protecting raw data, processed files, models, and experiment runs from repository commits.
  - Added GitHub Actions continuous integration workflow (`.github/workflows/ci.yml`).

### Entry 002 — Benchmark Specifications & Upstream Integrity
- **Date**: 2026-11-13
- **Decision**: Target the `ad-consistency` config of `cyberagent/AdTEC` (NAACL 2025).
- **Split Distribution**:
  - `train`: 10,635 rows
  - `valid`: 945 rows
  - `test`: 970 rows (frozen)
- **Features**: `title` (Japanese search ad headline), `lp_text` (landing page text), `label` (`consistent` / `inconsistent`).
- **Policy**: Maintain strict separation between research tooling license (MIT) and dataset terms (CC BY-NC-SA 4.0). Exclude dataset binaries and text from Git commits.

### Entry 003 — Baseline Model Architecture & Evaluation Protocol
- **Date**: 2026-11-13
- **Decision**: Implement two initial baselines:
  1. `MajorityBaseline`: Dummy classifier predicting the most frequent training label (establishes trivial baseline threshold).
  2. `TfidfLogRegBaseline`: Character n-gram TF-IDF vectorizer (ngram_range=(1, 3)) + Logistic Regression with class-weight balancing.
- **Protocol**:
  - Optimization is performed solely on `train`.
  - Hyperparameter tuning and error analysis occur on `valid`.
  - The `test` split remains strictly untouched until final model selection is frozen.
  - Primary evaluation metric: Macro-F1.

### Entry 004 — Next Planned Steps
- Execute automated audit script `scripts/run_audit.py` once data is loaded.
- Run baseline model training (`scripts/train_baseline.py`) and log benchmark results in `reports/generated/`.
- Prepare Notebooks 01 and 02 with live audit and baseline evaluation, and skeletonize Notebooks 03–06.

