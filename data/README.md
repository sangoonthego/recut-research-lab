# Data Directory

This directory manages datasets for advertising consistency research.

## Structure

- `raw/`: Raw downloaded benchmark data (e.g. `raw/adtec_consistency/`). Excluded from version control.
- `processed/`: Normalized, tokenized, or split data files for local experimentation. Excluded from version control.
- `recut_eval/`: Evaluation datasets for domain transfer experiments representing Recut advertising generation scenarios. Excluded from version control.

## Upstream Dataset: CyberAgent AdTEC (Ad-Consistency)

- **Source**: [HuggingFace: cyberagent/AdTEC](https://huggingface.co/datasets/cyberagent/AdTEC)
- **Task**: Binary consistency classification (`consistent` vs. `inconsistent`) between ad title and landing page text (`lp_text`).
- **Domain**: Japanese search engine advertising.
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

## Usage & Security Rules

1. **Do not commit dataset contents** to Git. All raw and processed data paths are ignored by `.gitignore`.
2. To download and verify the AdTEC dataset locally, run:
   ```bash
   python scripts/download_adtec.py
   ```
3. To audit dataset integrity, distributions, and split leakage, run:
   ```bash
   python scripts/run_audit.py
   ```

