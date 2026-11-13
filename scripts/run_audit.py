"""Run comprehensive data quality and integrity audit on CyberAgent AdTEC splits.

Generates:
    - reports/generated/dataset_audit.json
    - reports/generated/dataset_audit.md
    - reports/generated/qualitative_samples.csv (20 manual review items from train/valid)

Usage:
    python scripts/run_audit.py [--data-dir data/raw/adtec_consistency] [--output-dir reports/generated]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.audit import (
    format_audit_markdown,
    run_dataset_audit,
    sample_qualitative_audit,
)
from recut_lab.data import load_adtec_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Audit AdTEC dataset splits.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/adtec_consistency",
        help="Directory containing dataset splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/generated",
        help="Directory to save audit reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for qualitative sample generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset splits from {data_dir.resolve()}...")
    try:
        dataset = load_adtec_dataset(data_dir)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        print("Run 'python scripts/download_adtec.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loaded splits: {dataset.split_sizes()}")
    split_matches = dataset.verify_split_counts()
    for s_name, res in split_matches.items():
        tag = "[VERIFIED MATCH]" if res["match"] else "[WARNING MISMATCH]"
        print(f"    - {s_name}: {res['actual']} rows (published: {res['expected']}) {tag}")

    print("[*] Running comprehensive audit on all splits and cross-split pairs...")
    summary = run_dataset_audit(dataset)

    # 1. Save primary JSON audit reports
    dataset_json_path = output_dir / "dataset_audit.json"
    audit_report_json_path = output_dir / "audit_report.json"
    summary_dict = summary.to_dict()

    for p in [dataset_json_path, audit_report_json_path]:
        with p.open("w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    # 2. Save primary Markdown audit reports
    md_content = format_audit_markdown(summary)
    dataset_md_path = output_dir / "dataset_audit.md"
    audit_report_md_path = output_dir / "audit_report.md"

    for p in [dataset_md_path, audit_report_md_path]:
        with p.open("w", encoding="utf-8") as f:
            f.write(md_content)

    # 3. Generate and save 20 qualitative inspection samples (train + valid ONLY)
    print("[*] Generating 20 qualitative inspection samples (10 consistent, 10 inconsistent) from train/valid...")
    sample_df = sample_qualitative_audit(dataset, n_per_class=10, random_seed=args.seed)
    samples_csv_path = output_dir / "qualitative_samples.csv"
    samples_json_path = output_dir / "qualitative_samples.json"

    sample_df.to_csv(samples_csv_path, index=False, encoding="utf-8")
    sample_df.to_json(samples_json_path, orient="records", indent=2, force_ascii=False)

    print(f"[+] Audit completed successfully!")
    print(f"    - Audit JSON: {dataset_json_path}")
    print(f"    - Audit Markdown: {dataset_md_path}")
    print(f"    - Qualitative Samples CSV: {samples_csv_path}")
    print(f"    - Qualitative Samples JSON: {samples_json_path}")
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)


if __name__ == "__main__":
    main()
