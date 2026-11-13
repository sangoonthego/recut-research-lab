"""Unit tests for dataset auditing, text length calculation, and overlap detection."""

import pandas as pd
import pytest

from recut_lab.audit import (
    analyze_cross_split_overlap,
    audit_single_split,
    compute_text_length_stats,
    format_audit_markdown,
    run_dataset_audit,
)
from recut_lab.data import ConsistencyDataset, create_synthetic_dataset


def test_compute_text_length_stats():
    series = pd.Series(["a", "ab", "abc", "abcd"])
    stats = compute_text_length_stats(series)
    assert stats.min == 1
    assert stats.max == 4
    assert stats.median == 2.5
    assert stats.mean == 2.5
    assert stats.iqr > 0


def test_audit_single_split():
    df = pd.DataFrame({
        "label": ["consistent", "consistent", "inconsistent"],
        "title": ["Ad Title 1", "Ad Title 1", "Ad Title 2"],
        "lp_text": ["Landing page content 1", "Landing page content 1", "Landing page content 2"],
    })
    report = audit_single_split(df, "train")
    assert report.total_rows == 3
    assert report.exact_duplicate_rows == 1
    assert report.duplicate_titles == 1
    assert report.duplicate_lp_texts == 1
    assert report.label_distribution["consistent"] == 2
    assert report.label_distribution["inconsistent"] == 1


def test_analyze_cross_split_overlap():
    df1 = pd.DataFrame({
        "label": ["consistent", "inconsistent"],
        "title": ["Title A", "Title B"],
        "lp_text": ["Landing A", "Landing B"],
    })
    df2 = pd.DataFrame({
        "label": ["consistent", "consistent"],
        "title": ["Title A", "Title C"],
        "lp_text": ["Landing A", "Landing C"],
    })
    overlap = analyze_cross_split_overlap(df1, df2, "train", "valid")
    assert overlap.exact_pair_overlap_count == 1
    assert overlap.lp_overlap_count == 1
    assert overlap.title_overlap_count == 1


def test_run_dataset_audit_and_markdown():
    ds = create_synthetic_dataset(num_samples_per_split=10)
    summary = run_dataset_audit(ds)
    assert "train" in summary.splits
    assert "valid" in summary.splits
    assert "test" in summary.splits
    assert len(summary.cross_split_overlaps) == 3

    md = format_audit_markdown(summary)
    assert "# Dataset Audit Report" in md
    assert "| Split | Total Rows |" in md
