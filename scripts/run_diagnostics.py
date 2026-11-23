"""Run Milestone M2.0.1 — Baseline Failure Diagnostics (D1, D2, D3, Seen/Unseen LP).

Uses official train and valid splits exclusively; official test split is never touched.

Usage:
    python scripts/run_diagnostics.py [--config configs/m2_diagnostics.json]
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.data import load_adtec_train_valid
from recut_lab.diagnostics import M2_0_1_DEFAULT_CONFIG, run_m2_failure_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute M2.0.1 baseline failure diagnostics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m2_diagnostics.json",
        help="Path to M2.0.1 JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        config = deepcopy(M2_0_1_DEFAULT_CONFIG)

    data_dir = config["data"]["raw_dir"]
    print(f"[*] Loading ONLY official train and valid splits from '{data_dir}'...")
    print("    [PROTOCOL CHECK] Official test split is strictly protected and will NOT be loaded.")
    train_df, valid_df, metadata = load_adtec_train_valid(data_dir)
    print(f"[+] Loaded train ({len(train_df)}) and valid ({len(valid_df)}).")

    print("[*] Running M2.0.1 diagnostic suite (D1, D2, D3, Seen/Unseen LP)...")
    res = run_m2_failure_diagnostics(
        train_df=train_df,
        valid_df=valid_df,
        dataset_metadata=metadata,
        config=config,
        m2_run_dir=config["m2_reference_run"],
    )

    print(f"\n[+] Diagnostics completed! Run artifacts: {Path(res['run_dir']).resolve()}")
    print(f"[+] Report written to: {(Path(res['reports_dir']) / 'baseline_failure_diagnostics.md').resolve()}")
    print("\n" + res["summary_md"])


if __name__ == "__main__":
    main()
