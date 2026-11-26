"""Execute Milestone M2.1 (RQ1): Generalization to Unseen Landing Pages.

Compares Protocol A (StratifiedKFold) vs. Protocol B (StratifiedGroupKFold grouped by
NFKC-normalized lp_text) exclusively within data/raw/adtec_consistency/train.tsv.
Official valid.tsv and test.tsv are strictly never loaded or accessed.

Usage:
    python scripts/run_generalization.py [--config configs/m2_generalization.json]
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.generalization import M2_1_DEFAULT_CONFIG, run_rq1_generalization_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute M2.1 RQ1 Generalization Experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m2_generalization.json",
        help="Path to RQ1 JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else deepcopy(M2_1_DEFAULT_CONFIG)

    print("[*] Running Milestone M2.1 — RQ1: Generalization to Unseen Landing Pages...")
    print("    [PROTOCOL CHECK] Using ONLY official train.tsv; valid.tsv and test.tsv remain untouched.")

    res = run_rq1_generalization_study(config=config)

    print(f"\n[+] RQ1 study completed! Run artifacts: {Path(res['run_dir']).resolve()}")
    print(f"[+] Report written to: {(Path(res['reports_dir']) / 'rq1_generalization_report.md').resolve()}")
    print("\n" + res["report_md"])


if __name__ == "__main__":
    main()
