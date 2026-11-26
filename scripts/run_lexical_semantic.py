"""Execute Milestone M2.2 (RQ2): Lexical vs Semantic Modeling.

Evaluates L0 (frozen M2.1 B3_pair lexical baseline), L1 (lexical interaction),
S1 (frozen intfloat/multilingual-e5-small + LogReg), and S2 (hybrid lexical + semantic)
on the exact M2.1 Protocol B LP-group-disjoint folds within train.tsv ONLY.

Official valid.tsv and test.tsv are strictly never loaded or accessed.

Usage:
    python scripts/run_lexical_semantic.py [--config configs/m2_lexical_semantic.json] [--smoke-test]
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.generalization import load_adtec_train_only
from recut_lab.lexical_semantic import M2_2_DEFAULT_CONFIG, run_rq2_lexical_semantic_study
from recut_lab.semantic import run_semantic_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute M2.2 RQ2 Lexical vs Semantic Experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m2_lexical_semantic.json",
        help="Path to RQ2 JSON config file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small technical smoke test on 10 training rows (NOT a research result).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = (
        json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_path.is_file()
        else deepcopy(M2_2_DEFAULT_CONFIG)
    )

    if args.smoke_test:
        print("[*] Running technical smoke test for frozen intfloat/multilingual-e5-small (NOT a research result)...")
        train_df, _ = load_adtec_train_only(config["data"]["raw_dir"])
        sample_df = train_df.head(10).copy()
        sem_cfg = config["semantic"]
        smoke = run_semantic_smoke_test(
            sample_df=sample_df,
            model_id=sem_cfg["model_id"],
            revision=sem_cfg.get("revision"),
            batch_size=min(8, int(sem_cfg.get("batch_size", 32))),
            device=str(sem_cfg.get("device", "cpu")),
        )
        print(json.dumps(smoke, indent=2))
        return

    print("[*] Running Milestone M2.2 — RQ2: Lexical vs Semantic Modeling...")
    print("    [PROTOCOL CHECK] Using ONLY official train.tsv + M2.1 Protocol B LP-group-disjoint folds.")
    print("    [PROTOCOL CHECK] Official valid.tsv and test.tsv remain strictly untouched.")

    res = run_rq2_lexical_semantic_study(config=config)

    print(f"\n[+] RQ2 study completed! Run artifacts: {Path(res['run_dir']).resolve()}")
    print(
        f"[+] Report written to: {(Path(res['reports_dir']) / 'rq2_lexical_semantic_report.md').resolve()}"
    )
    print("\n" + res["report_md"])


if __name__ == "__main__":
    main()
