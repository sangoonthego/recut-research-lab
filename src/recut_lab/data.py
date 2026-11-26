"""Data loading, schema validation, and persistence for advertising consistency benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

EXPECTED_COLUMNS: List[str] = ["label", "lp_text", "title"]
VALID_LABELS: Set[str] = {"consistent", "inconsistent"}

# Published official benchmark row counts from NAACL 2025 paper / AdTEC repo
PUBLISHED_SPLIT_COUNTS: Dict[str, int] = {
    "train": 10635,
    "valid": 945,
    "test": 970,
}


@dataclass
class ConsistencyDataset:
    """Container holding official splits and dataset metadata."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_split(self, split_name: str) -> pd.DataFrame:
        """Retrieve split DataFrame by name ('train', 'valid', or 'test')."""
        normalized = split_name.lower()
        if normalized == "train":
            return self.train
        if normalized in ("valid", "val", "validation"):
            return self.valid
        if normalized == "test":
            return self.test
        raise ValueError(
            f"Unknown split name '{split_name}'. Expected 'train', 'valid', or 'test'."
        )

    def split_sizes(self) -> Dict[str, int]:
        """Return sample counts per split."""
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
        }

    def verify_split_counts(self) -> Dict[str, Dict[str, Any]]:
        """Verify split row counts against official published AdTEC benchmark counts."""
        results = {}
        for split_name, expected_count in PUBLISHED_SPLIT_COUNTS.items():
            actual_count = len(self.get_split(split_name))
            results[split_name] = {
                "expected": expected_count,
                "actual": actual_count,
                "match": expected_count == actual_count,
                "difference": actual_count - expected_count,
            }
        return results


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute deterministic SHA-256 checksum for a local file."""
    path = Path(file_path)
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_schema(df: pd.DataFrame, split_name: str = "dataset") -> None:
    """Validate that the DataFrame contains expected columns and supported label values.

    Args:
        df: Input DataFrame to check.
        split_name: Name of the split for context in error messages.

    Raises:
        ValueError: If columns are missing or if unsupported labels are present.
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for {split_name}, got {type(df)}")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed for split '{split_name}'. "
            f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Check for unexpected labels
    if "label" in df.columns and len(df) > 0:
        actual_labels = set(df["label"].dropna().unique())
        invalid_labels = actual_labels - VALID_LABELS
        if invalid_labels:
            raise ValueError(
                f"Invalid labels found in split '{split_name}': {invalid_labels}. "
                f"Supported labels are: {VALID_LABELS}"
            )


def load_adtec_split(split_path: Path | str, split_name: str = "") -> pd.DataFrame:
    """Load a single split file from TSV, JSONL, or CSV and validate schema.

    Preserves original text contents without modification, truncation, or normalization.

    Args:
        split_path: Path to the split file.
        split_name: Name of the split.

    Returns:
        Validated pandas DataFrame.
    """
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path.resolve()}. "
            "Run 'python scripts/download_adtec.py' to obtain the dataset."
        )

    # TSV files are the official format for AdTEC
    if path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype={"label": str, "lp_text": str, "title": str})
    elif path.suffix in (".json", ".jsonl"):
        df = pd.read_json(path, dtype={"label": str, "lp_text": str, "title": str})
    elif path.suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Default try TSV then jsonl
        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
        except Exception:
            df = pd.read_json(path, lines=True)

    # Ensure all expected columns exist as string types
    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    validate_schema(df, split_name=split_name or path.stem)
    return df


def load_adtec_dataset(data_dir: Path | str = "data/raw/adtec_consistency") -> ConsistencyDataset:
    """Deterministically load official AdTEC train, valid, and test splits from a directory.

    Prioritizes official TSV format, with fallback to JSONL or CSV.

    Args:
        data_dir: Directory containing train/valid/test split files.

    Returns:
        ConsistencyDataset object with validated train, valid, and test DataFrames.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dir_path.resolve()}' does not exist. "
            "Please download the dataset using 'python scripts/download_adtec.py'."
        )

    splits: Dict[str, pd.DataFrame] = {}
    for split in ["train", "valid", "test"]:
        found_file = None
        # Prioritize TSV (official upstream benchmark format), then jsonl, csv, parquet
        for ext in [".tsv", ".jsonl", ".csv", ".parquet"]:
            candidate = dir_path / f"{split}{ext}"
            if candidate.exists():
                found_file = candidate
                break

        if not found_file:
            raise FileNotFoundError(
                f"Could not find '{split}' split in '{dir_path.resolve()}'. "
                f"Expected {split}.tsv, {split}.jsonl, or {split}.csv. "
                "Run 'python scripts/download_adtec.py' to acquire the dataset."
            )
        splits[split] = load_adtec_split(found_file, split_name=split)

    metadata: Dict[str, Any] = {}
    metadata_path = dir_path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return ConsistencyDataset(
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
        metadata=metadata,
    )


def load_adtec_train_valid(
    data_dir: Path | str = "data/raw/adtec_consistency",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load ONLY the official train and valid splits plus metadata, leaving test untouched.

    This enforces strict split discipline during development and baseline comparison
    by never reading the frozen test file from disk.

    Args:
        data_dir: Directory containing official split files and metadata.json.

    Returns:
        Tuple of (train_df, valid_df, metadata_dict).
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dir_path.resolve()}' does not exist. "
            "Please download the dataset using 'python scripts/download_adtec.py'."
        )

    loaded: Dict[str, pd.DataFrame] = {}
    for split in ["train", "valid"]:
        found_file = None
        for ext in [".tsv", ".jsonl", ".csv", ".parquet"]:
            candidate = dir_path / f"{split}{ext}"
            if candidate.exists():
                found_file = candidate
                break

        if not found_file:
            raise FileNotFoundError(
                f"Could not find '{split}' split in '{dir_path.resolve()}'. "
                f"Expected {split}.tsv, {split}.jsonl, or {split}.csv. "
                "Run 'python scripts/download_adtec.py' to acquire the dataset."
            )
        loaded[split] = load_adtec_split(found_file, split_name=split)

    metadata: Dict[str, Any] = {}
    metadata_path = dir_path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return loaded["train"], loaded["valid"], metadata


def save_adtec_tsv_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path | str,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist splits as UTF-8 TSV files and write metadata manifest with SHA-256 hashes.

    Args:
        splits: Dictionary mapping split names ('train', 'valid', 'test') to DataFrames.
        output_dir: Destination folder.
        source_metadata: Optional provenance metadata.

    Returns:
        Dictionary of split names to computed SHA-256 hashes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checksums: Dict[str, str] = {}
    file_records: Dict[str, Any] = {}

    for split_name, df in splits.items():
        validate_schema(df, split_name=split_name)
        target_file = out_path / f"{split_name}.tsv"
        df.to_csv(target_file, sep="\t", index=False, encoding="utf-8")
        sha = compute_file_sha256(target_file)
        checksums[split_name] = sha
        file_records[split_name] = {
            "filename": target_file.name,
            "sha256": sha,
            "row_count": len(df),
            "file_size_bytes": target_file.stat().st_size,
        }

    manifest = {
        "dataset_name": "cyberagent/AdTEC",
        "config_name": "ad-consistency",
        "license": "CC BY-NC-SA 4.0",
        "task": "binary_ad_consistency",
        "format": "tsv",
        "labels": sorted(list(VALID_LABELS)),
        "splits": file_records,
        "source_metadata": source_metadata or {},
    }

    manifest_path = out_path / "metadata.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return checksums


def save_adtec_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path | str,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist splits as UTF-8 JSONL files (retained for backward compatibility)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checksums: Dict[str, str] = {}
    file_records: Dict[str, Any] = {}

    for split_name, df in splits.items():
        validate_schema(df, split_name=split_name)
        target_file = out_path / f"{split_name}.jsonl"
        df.to_json(target_file, orient="records", lines=True, force_ascii=False)
        sha = compute_file_sha256(target_file)
        checksums[split_name] = sha
        file_records[split_name] = {
            "filename": target_file.name,
            "sha256": sha,
            "row_count": len(df),
            "file_size_bytes": target_file.stat().st_size,
        }

    manifest = {
        "dataset_name": "cyberagent/AdTEC",
        "config_name": "ad-consistency",
        "license": "CC BY-NC-SA 4.0",
        "task": "binary_ad_consistency",
        "labels": sorted(list(VALID_LABELS)),
        "splits": file_records,
        "source_metadata": source_metadata or {},
    }

    manifest_path = out_path / "metadata.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return checksums


def create_synthetic_dataset(num_samples_per_split: int = 20) -> ConsistencyDataset:
    """Generate small synthetic dataset fixture for tests and offline scaffolding."""
    records = []
    labels = ["consistent", "inconsistent"]
    for i in range(num_samples_per_split):
        lbl = labels[i % 2]
        records.append({
            "label": lbl,
            "title": f"Synthetic Ad Title {i} for Search Campaign",
            "lp_text": f"Landing page content describing service and offerings for item {i}."
        })

    train_df = pd.DataFrame(records)
    valid_df = pd.DataFrame(records[: max(4, num_samples_per_split // 2)])
    test_df = pd.DataFrame(records[: max(4, num_samples_per_split // 2)])

    return ConsistencyDataset(
        train=train_df,
        valid=valid_df,
        test=test_df,
        metadata={"is_synthetic": True, "num_samples": num_samples_per_split},
    )
