"""Train and evaluate the M2.0 four-baseline suite (B0–B3) on AdTEC ad-consistency.

Fits all models exclusively on the official train split and evaluates exclusively
on the official validation split. The official test split is never loaded or accessed.

Usage:
    python scripts/train_baseline.py [--config configs/m2_baseline.json]
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

# Add src to sys.path when executed directly
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.baseline import M2_DEFAULT_CONFIG, run_m2_baseline_benchmark
from recut_lab.data import load_adtec_train_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible M2.0 baseline comparison (B0–B3) on AdTEC train/valid splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m2_baseline.json",
        help="Path to JSON benchmark configuration file.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Override experiment run identifier (default from config: m2_baseline_v001).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory (defaults to config or data/raw/adtec_consistency).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="experiments/runs",
        help="Base directory for experiment runs.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Base directory for serialized model checkpoints.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/generated/m2_baseline",
        help="Directory for generated M2.0 benchmark reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"[!] Config file '{config_path}' not found; using M2_DEFAULT_CONFIG.")
        config = deepcopy(M2_DEFAULT_CONFIG)

    if args.experiment_id:
        config["experiment_id"] = args.experiment_id
    if args.seed is not None:
        config["random_seed"] = args.seed

    data_dir = args.data_dir or config.get("data", {}).get("raw_dir", "data/raw/adtec_consistency")

    print(f"[*] Loading official train and valid splits ONLY from '{data_dir}'...")
    print("    [PROTOCOL CHECK] Official test split is strictly protected and will NOT be loaded.")
    try:
        train_df, valid_df, dataset_metadata = load_adtec_train_valid(data_dir)
    except FileNotFoundError as exc:
        print(f"[!] Error: {exc}", file=sys.stderr)
        print("Run 'python scripts/download_adtec.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Loaded train ({len(train_df)} rows) and valid ({len(valid_df)} rows).")
    if dataset_metadata:
        print(f"    Dataset revision: {dataset_metadata.get('upstream_revision', 'unknown')}")

    print(f"[*] Executing M2.0 four-baseline benchmark (experiment_id={config['experiment_id']})...")
    result = run_m2_baseline_benchmark(
        train_df=train_df,
        valid_df=valid_df,
        dataset_metadata=dataset_metadata,
        config=config,
        runs_base_dir=args.runs_dir,
        models_base_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )

    benchmark_df = result["benchmark_df"]
    print(f"\n[+] Benchmark completed successfully!")
    print(f"    Run artifacts:     {Path(result['run_dir']).resolve()}")
    print(f"    Reports directory: {Path(result['reports_dir']).resolve()}")
    print("\n" + "=" * 90)
    print("M2.0 VALIDATION BENCHMARK SUMMARY (OFFICIAL VALID SPLIT, N = 945)")
    print("=" * 90)
    print(benchmark_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
