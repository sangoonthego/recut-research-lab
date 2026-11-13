"""Download and persist the official CyberAgent AdTEC ad-consistency benchmark dataset.

Usage:
    python scripts/download_adtec.py [--output-dir data/raw/adtec_consistency]
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path if running as standalone script
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.data import save_adtec_splits, validate_schema


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CyberAgent AdTEC ad-consistency dataset from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/adtec_consistency",
        help="Destination directory for dataset splits.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="cyberagent/AdTEC",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="ad-consistency",
        help="Dataset configuration name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"[*] Connecting to Hugging Face to load {args.dataset_id} (config: {args.config_name})...")
    try:
        from datasets import load_dataset
        ds_dict = load_dataset(args.dataset_id, args.config_name)
    except Exception as e:
        print(f"[!] Error loading dataset from Hugging Face: {e}", file=sys.stderr)
        print("Please check your internet connection or Hugging Face access.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Successfully retrieved splits: {list(ds_dict.keys())}")
    split_dfs = {}
    for split_name in ["train", "valid", "test"]:
        if split_name not in ds_dict:
            raise ValueError(f"Expected split '{split_name}' not found in loaded dataset.")
        df = ds_dict[split_name].to_pandas()
        validate_schema(df, split_name=split_name)
        split_dfs[split_name] = df
        print(f"    - {split_name}: {len(df)} samples")

    print(f"[*] Persisting dataset splits to {output_dir.resolve()}...")
    metadata = {
        "source_hub": "huggingface",
        "dataset_id": args.dataset_id,
        "config_name": args.config_name,
        "citation": "NAACL 2025: AdTEC: A Unified Benchmark for Evaluating Text Quality in Search Engine Advertising",
    }
    checksums = save_adtec_splits(split_dfs, output_dir, source_metadata=metadata)

    print("[+] Download and persistence completed successfully!")
    print("    SHA-256 Checksums:")
    for split_name, sha in checksums.items():
        print(f"    - {split_name}.jsonl: {sha}")


if __name__ == "__main__":
    main()
