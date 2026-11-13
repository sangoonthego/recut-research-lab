"""Download and persist the official CyberAgent AdTEC ad-consistency benchmark dataset.

Downloads original train.tsv, valid.tsv, and test.tsv files directly from Hugging Face / GitHub.
Preserves original TSV contents, encodings (UTF-8), and records provenance metadata and SHA-256 hashes.

Usage:
    python scripts/download_adtec.py [--output-dir data/raw/adtec_consistency] [--force]
"""

import argparse
import datetime
import json
import shutil
import sys
import urllib.request
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.data import (
    EXPECTED_COLUMNS,
    PUBLISHED_SPLIT_COUNTS,
    compute_file_sha256,
    load_adtec_split,
    validate_schema,
)

REPO_ID = "cyberagent/AdTEC"
CONFIG_SUBDIR = "data/ad-consistency"
SPLIT_FILENAMES = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "test": "test.tsv",
}

# Direct fallback URLs in case huggingface_hub is restricted or offline
RAW_HF_BASE = f"https://huggingface.co/datasets/{REPO_ID}/raw/main/{CONFIG_SUBDIR}"
RAW_GH_BASE = "https://raw.githubusercontent.com/CyberAgentAILab/AdTEC/main/data"
GH_FALLBACK_FILES = {
    "train": "ad-consistency.train.tsv",
    "valid": "ad-consistency.valid.tsv",
    "test": "ad-consistency.test.tsv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download official CyberAgent AdTEC ad-consistency TSV splits."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/adtec_consistency",
        help="Destination directory for raw TSV splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files even if checksums match.",
    )
    return parser.parse_args()


def get_repo_revision(repo_id: str) -> str:
    """Attempt to get current commit SHA for the Hugging Face dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id, repo_type="dataset")
        return info.sha or "main"
    except Exception:
        return "main"


def download_single_split(split_name: str, target_path: Path) -> str:
    """Download single TSV split using huggingface_hub with HTTP fallback."""
    rel_filename = f"{CONFIG_SUBDIR}/{SPLIT_FILENAMES[split_name]}"
    
    # 1. Try huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_filename,
            repo_type="dataset",
        )
        shutil.copyfile(cached_path, target_path)
        return "huggingface_hub"
    except Exception as e_hub:
        print(f"    [i] huggingface_hub download attempt failed ({e_hub}). Trying direct raw download...")

    # 2. Try Hugging Face raw HTTP URL
    hf_url = f"{RAW_HF_BASE}/{SPLIT_FILENAMES[split_name]}"
    try:
        req = urllib.request.Request(hf_url, headers={"User-Agent": "recut-research-lab/0.1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, target_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        return "huggingface_raw_http"
    except Exception as e_hf:
        print(f"    [i] HF raw URL failed ({e_hf}). Trying GitHub raw fallback...")

    # 3. Try GitHub raw fallback
    gh_url = f"{RAW_GH_BASE}/{GH_FALLBACK_FILES[split_name]}"
    try:
        req = urllib.request.Request(gh_url, headers={"User-Agent": "recut-research-lab/0.1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, target_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        return "github_raw_http"
    except Exception as e_gh:
        raise RuntimeError(
            f"Failed to download {split_name} from all official sources: "
            f"HF Hub, HF Raw ({e_hf}), GitHub Raw ({e_gh}). "
            "Please check internet connectivity."
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading official CyberAgent AdTEC ad-consistency dataset...")
    print(f"    Target directory: {output_dir.resolve()}")

    # Check if files already exist
    all_exist = all((output_dir / fname).is_file() for fname in SPLIT_FILENAMES.values())
    if all_exist and not args.force:
        print("[!] All raw TSV files already exist in destination.")
        print("    Verifying integrity without re-downloading (use --force to overwrite)...")
    else:
        revision = get_repo_revision(REPO_ID)
        print(f"    Upstream dataset: {REPO_ID} (commit: {revision})")

    file_metadata = {}
    download_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    repo_revision = get_repo_revision(REPO_ID)

    for split_name, fname in SPLIT_FILENAMES.items():
        target_file = output_dir / fname
        
        if not target_file.is_file() or args.force:
            print(f"[*] Fetching {split_name} split ({fname})...")
            source_used = download_single_split(split_name, target_file)
            print(f"    [+] Saved via {source_used}")
        else:
            print(f"[*] Checking existing {split_name} split ({fname})...")

        # Verify readability, schema, and row count
        df = load_adtec_split(target_file, split_name=split_name)
        validate_schema(df, split_name=split_name)
        row_count = len(df)
        expected_rows = PUBLISHED_SPLIT_COUNTS.get(split_name, row_count)
        sha256 = compute_file_sha256(target_file)
        file_size = target_file.stat().st_size

        if row_count != expected_rows:
            print(
                f"    [WARNING] Row count mismatch for {split_name}: "
                f"got {row_count}, expected {expected_rows} published rows!"
            )
        else:
            print(f"    [VERIFIED] Row count matches official published count: {row_count} rows.")

        file_metadata[split_name] = {
            "filename": fname,
            "format": "tsv",
            "sha256": sha256,
            "file_size_bytes": file_size,
            "row_count": row_count,
            "expected_row_count": expected_rows,
            "verified_readable": True,
        }

    manifest = {
        "dataset_name": REPO_ID,
        "config_name": "ad-consistency",
        "license": "CC BY-NC-SA 4.0",
        "task": "binary_ad_consistency",
        "download_date_utc": download_timestamp,
        "upstream_revision": repo_revision,
        "official_upstream_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "official_github_url": "https://github.com/CyberAgentAILab/AdTEC",
        "columns": EXPECTED_COLUMNS,
        "splits": file_metadata,
    }

    manifest_path = output_dir / "metadata.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Dataset download and verification completed successfully!")
    print(f"    Metadata manifest: {manifest_path.resolve()}")
    for split_name, meta in file_metadata.items():
        print(f"    - {meta['filename']}: {meta['row_count']} rows, {meta['file_size_bytes']} bytes, SHA-256: {meta['sha256']}")


if __name__ == "__main__":
    main()
