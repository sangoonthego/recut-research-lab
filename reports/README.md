# Reports Directory

This directory stores generated analytical reports, charts, and audit summaries.

## Structure

- `generated/`: Output directory for scripts and notebooks.
  - `audit_report.json` & `audit_report.md`: Dataset audit findings, label balances, and cross-split checks.
  - `baseline_metrics.json`: Comparative evaluation tables for baseline models.
  - Figures and plots (e.g. confusion matrix plots, text length distributions).

All generated files in `generated/` are excluded from git commits by default.

