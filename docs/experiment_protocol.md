# Experiment Protocol and Evaluation Standards

**Project**: Recut Research Lab 01  
**Dataset**: CyberAgent AdTEC `ad-consistency`  

---

## 1. Split Discipline

To guarantee scientific validity and prevent data leakage:

1. **Training Split (`train`)**:
   - Used exclusively for model parameter estimation (fitting TF-IDF vectorizers, training classifiers).
   - Never exposed to validation or test evaluation data during fitting.

2. **Validation Split (`valid`)**:
   - Used for hyperparameter tuning, model comparison, feature selection, and error taxonomy development.
   - All iteration decisions and notebook explorations must report on the validation split.

3. **Frozen Test Split (`test`)**:
   - **Strictly frozen.** Never used for exploratory analysis, feature engineering, or hyperparameter selection.
   - Evaluated strictly once when candidate models are frozen to report final benchmark metrics.
   - Any script execution against the test split requires explicit confirmation (`--confirm-test`).

---

## 2. Evaluation Metrics

Because advertising consistency datasets may exhibit label imbalance, metric evaluation relies on multi-dimensional reporting:

- **Macro-F1 (Primary Metric)**:
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{consistent}} + \text{F1}_{\text{inconsistent}}}{2}$$
  Ensures that performance is balanced across both minority and majority classes.

- **Per-Class Metrics**:
  - Precision, Recall, and F1 for `consistent`.
  - Precision, Recall, and F1 for `inconsistent`.

- **Accuracy**:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total Samples}}$$

- **Confusion Matrix**:
  Full 2x2 matrix capturing False Positives (inconsistent ad classified as consistent) and False Negatives (consistent ad classified as inconsistent).

- **Inference Latency**:
  - Measured in milliseconds per sample (averaged over 3 repeated inference passes).
  - Critical constraint for high-throughput ad generation in Recut pipelines.

---

## 3. Leakage Prevention and Data Audit Checks

Before running modeling pipelines, the data audit script (`scripts/run_audit.py`) validates:
1. **Exact Pair Duplication**: Checks if any `(title, lp_text)` pair in `valid` or `test` appeared in `train`.
2. **Landing Page Cluster Leakage**: Tracks the percentage of landing pages in the validation set that share identical `lp_text` with training rows.
3. **Empty or Corrupted Inputs**: Verifies that no whitespace-only or NaN fields contaminate evaluation.

---

## 4. Reproducibility Standards

- **Deterministic Seeds**: All scripts and models accept a `--seed` parameter (default `42`), which is propagated to Python's `random`, `numpy`, and `os.environ["PYTHONHASHSEED"]`.
- **Environment Logging**: Every experiment run logs system hardware, Python version, platform string, and timestamps to `metadata.json`.
- **Artifact Isolation**: Model checkpoints are stored in `models/`, run logs in `experiments/runs/`, and summary reports in `reports/generated/`.

