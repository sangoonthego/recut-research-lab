"""Unit tests for dataset auditing, text length calculation, overlap, and qualitative sampling."""

import pandas as pd
import pytest

from recut_lab.audit import (
    analyze_cross_split_overlap,
    audit_potential_issues,
    audit_single_split,
    compute_label_distribution,
    compute_lexical_relationships,
    compute_text_length_stats,
    find_conflicting_labels,
    format_audit_markdown,
    run_dataset_audit,
    sample_qualitative_audit,
)
from recut_lab.data import ConsistencyDataset, create_synthetic_dataset


def test_compute_text_length_stats():
    series = pd.Series(["a", "ab", "abc", "abcd", "abcde"])
    stats = compute_text_length_stats(series)
    assert stats.min == 1
    assert stats.max == 5
    assert stats.median == 3.0
    assert stats.mean == 3.0
    assert stats.p25 == 2.0
    assert stats.p75 == 4.0
    assert stats.p95 >= 4.0
    assert stats.iqr == 2.0


def test_find_conflicting_labels():
    df = pd.DataFrame({
        "label": ["consistent", "inconsistent", "consistent"],
        "title": ["Ad Title 1", "Ad Title 1", "Ad Title 2"],
        "lp_text": ["Landing page 1", "Landing page 1", "Landing page 2"],
    })
    conf_pairs, conf_rows = find_conflicting_labels(df)
    assert conf_pairs == 1
    assert conf_rows == 2


def test_audit_single_split_with_conflicts_and_whitespace():
    df = pd.DataFrame({
        "label": ["consistent", "inconsistent", "consistent", "consistent"],
        "title": ["Ad 1", "Ad 1", "Ad 2", "   "],
        "lp_text": ["LP 1", "LP 1", "LP 2", ""],
    })
    report = audit_single_split(df, "train")
    assert report.total_rows == 4
    assert report.duplicates.conflicting_label_pairs == 1
    assert report.duplicates.conflicting_label_sample_count == 2
    assert report.empty_strings["lp_text"] == 1
    assert report.whitespace_only["title"] == 1
    assert report.label_distribution.consistent_count == 3
    assert report.label_distribution.inconsistent_count == 1
    assert report.label_distribution.imbalance_ratio == 3.0


def test_compute_lexical_relationships():
    df = pd.DataFrame({
        "label": ["consistent"],
        "title": ["楽天カード 入会"],
        "lp_text": ["楽天カードの新規入会特典"],
    })
    lex = compute_lexical_relationships(df)
    assert 0.0 < lex.char_jaccard_mean <= 1.0
    assert 0.0 < lex.bigram_jaccard_mean <= 1.0
    assert 0.0 < lex.length_ratio_mean < 1.0


def test_audit_potential_issues():
    df = pd.DataFrame({
        "label": ["consistent", "inconsistent"],
        "title": ["Abc", "Normal Title for Ad"],
        "lp_text": ["Short", "Sufficiently long landing page description text for validation"],
    })
    issues = audit_potential_issues(df, min_title_len=5, min_lp_len=10)
    assert issues["short_title_count"] == 1
    assert issues["short_lp_count"] == 1


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
    assert overlap.normalized_pair_overlap_count == 1
    assert overlap.lp_overlap_count == 1
    assert overlap.title_overlap_count == 1


def test_sample_qualitative_audit_protocol():
    ds = create_synthetic_dataset(num_samples_per_split=30)
    sample_df = sample_qualitative_audit(ds, n_per_class=10, random_seed=42)

    assert len(sample_df) == 20
    assert (sample_df["original_label"] == "consistent").sum() == 10
    assert (sample_df["original_label"] == "inconsistent").sum() == 10

    # Ensure test split was NEVER sampled
    assert "test" not in sample_df["split"].unique()
    assert set(sample_df["split"].unique()).issubset({"train", "valid"})

    # Check manual review diagnostic columns are empty initially
    diagnostic_cols = [
        "topic_relevance",
        "direct_support",
        "potential_contradiction",
        "insufficient_evidence",
        "reviewer_notes",
    ]
    for col in diagnostic_cols:
        assert col in sample_df.columns
        assert (sample_df[col] == "").all()


def test_run_dataset_audit_and_markdown():
    ds = create_synthetic_dataset(num_samples_per_split=10)
    summary = run_dataset_audit(ds)
    assert "train" in summary.splits
    assert "valid" in summary.splits
    assert "test" in summary.splits
    assert len(summary.cross_split_overlaps) == 3

    md = format_audit_markdown(summary)
    assert "# Dataset Audit Report" in md
    assert "Structural Integrity" in md
    assert "Cross-Split Overlap" in md
