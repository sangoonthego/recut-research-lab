# Experiments Directory

This directory stores configuration templates and local experiment run records.

## Structure

- `runs/`: Contains per-run artifacts (`run_<timestamp>_<model>/`).
  - `config.json`: The frozen configuration parameters for the run.
  - `metadata.json`: Environment and execution metadata (platform, python version, random seed, timestamp).
  - `metrics.json`: Evaluated metrics on validation and/or test splits.
  - `predictions.csv`: Model predictions without copying raw source corpus text.

## Reproducibility Guidelines

- Every experiment must define a deterministic random seed.
- Configurations should be saved or referenced from `configs/`.
- Training data is used exclusively for model fitting; validation data is used for tuning and model selection.
- The official test split remains untouched until final frozen evaluation.

