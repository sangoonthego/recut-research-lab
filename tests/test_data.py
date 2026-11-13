"""Unit tests for dataset loading, schema validation, and persistence."""

from pathlib import Path
import pandas as pd
import pytest

from recut_lab.data import (
    ConsistencyDataset,
    create_synthetic_dataset,
    load_adtec_dataset,
    load_adtec_split,
    save_adtec_splits,
    validate_schema,
)


def test_validate_schema_valid():
    df = pd.DataFrame({
        "label": ["consistent", "inconsistent"],
        "title": ["Ad Title A", "Ad Title B"],
        "lp_text": ["Landing page description A", "Landing page description B"],
    })
    # Should execute without exception
    validate_schema(df, split_name="test_split")


def test_validate_schema_missing_column():
    df = pd.DataFrame({
        "title": ["Ad Title A"],
        "lp_text": ["LP text"],
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_invalid_label():
    df = pd.DataFrame({
        "label": ["unsupported_label"],
        "title": ["Ad Title"],
        "lp_text": ["Landing page"],
    })
    with pytest.raises(ValueError, match="Invalid labels found"):
        validate_schema(df)


def test_save_and_load_splits(tmp_path: Path):
    ds = create_synthetic_dataset(num_samples_per_split=6)
    splits = {"train": ds.train, "valid": ds.valid, "test": ds.test}

    checksums = save_adtec_splits(splits, tmp_path)
    assert "train" in checksums
    assert "valid" in checksums
    assert "test" in checksums
    assert len(checksums["train"]) == 64  # SHA-256 length

    loaded_ds = load_adtec_dataset(tmp_path)
    assert len(loaded_ds.train) == 6
    assert len(loaded_ds.valid) == 4
    assert len(loaded_ds.test) == 4
    assert loaded_ds.get_split("train").shape[0] == 6


def test_load_adtec_dataset_missing_dir(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_adtec_dataset(non_existent)


def test_load_adtec_dataset_missing_split(tmp_path: Path):
    # Directory exists but splits are absent
    with pytest.raises(FileNotFoundError, match="Could not find 'train' split"):
        load_adtec_dataset(tmp_path)

