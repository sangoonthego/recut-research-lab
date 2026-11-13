# Dataset Card: CyberAgent AdTEC (Ad-Consistency)

## 1. Summary

- **Dataset Name**: CyberAgent AdTEC (Ad Text Evaluation and Quality Benchmark)
- **Task Configuration**: `ad-consistency`
- **Language**: Japanese (`ja`)
- **Domain**: Search engine advertising
- **Task Type**: Binary classification
- **Input Fields**:
  - `title` (string): Search advertisement headline / title.
  - `lp_text` (string): Text content extracted from the corresponding landing page.
- **Target Field**:
  - `label` (string): `consistent` or `inconsistent`.

## 2. Citation and Reference

- **Paper**: *AdTEC: A Unified Benchmark for Evaluating Text Quality in Search Engine Advertising* (NAACL 2025 Main Track)
- **Authors**: CyberAgent AI Lab
- **Hugging Face Hub**: [https://huggingface.co/datasets/cyberagent/AdTEC](https://huggingface.co/datasets/cyberagent/AdTEC)
- **Repository**: [https://github.com/CyberAgentAILab/AdTEC](https://github.com/CyberAgentAILab/AdTEC)

## 3. Official Split Distribution

| Split | Number of Rows | Percentage | Role in Laboratory |
|---|---|---|---|
| `train` | 10,635 | 84.7% | Dedicated model fitting. |
| `valid` | 945 | 7.5% | Hyperparameter tuning, model selection, error analysis. |
| `test` | 970 | 7.7% | **Frozen test set.** Evaluated strictly once for final benchmark. |
| **Total** | **12,550** | **100.0%** | |

## 4. Licensing and Commercial Restrictions

- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Non-Commercial Restriction**: This dataset cannot be used for commercial training or commercial deployment.
- **Laboratory Separation**: The software tooling, evaluation harnesses, and experiment scripts in `recut-research-lab` are licensed under the MIT License. The dataset remains strictly under CC BY-NC-SA 4.0.
- **Git Policy**: All raw, processed, and split files must be kept out of version control (`.gitignore`).

## 5. Domain Distinction and Guardrails

- **Ad Consistency vs. Fact Verification**:
  The benchmark measures whether an ad headline aligns with the subject matter and claims presented on its destination landing page. It is **not** a three-class factual claim-verification task (e.g. `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`).
- **Skincare and Medical Claims**:
  The dataset contains diverse consumer advertising verticals. It does **not** evaluate clinical trials, skincare medical efficacy, or pharmaceutical compliance claims.

