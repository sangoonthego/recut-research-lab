"""Run data quality and distribution audit on dataset splits.

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

from recut_lab.audit import format_audit_markdown, run_dataset_audit
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

    print(f"[*] Running comprehensive audit on splits: {list(dataset.split_sizes().keys())}...")
    summary = run_dataset_audit(dataset)

    # Save JSON report
    json_path = output_dir / "audit_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    # Save Markdown report
    md_content = format_audit_markdown(summary)
    md_path = output_dir / "audit_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[+] Audit completed successfully!")
    print(f"    - JSON report: {json_path}")
    print(f"    - Markdown report: {md_path}")
    print()
    print(md_content)


if __name__ == "__main__":
    main()
