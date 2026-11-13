"""Dataset audit tooling for advertising consistency benchmarks.

Analyzes null values, duplicates, label balance, text lengths, and cross-split overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from recut_lab.data import ConsistencyDataset, validate_schema


@dataclass
class TextLengthStats:
    """Statistical summary of text length distributions."""

    min: int
    max: int
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    iqr: float


@dataclass
class SplitAuditReport:
    """Audit summary for a single split."""

    split_name: str
    total_rows: int
    null_counts: Dict[str, int]
    empty_strings: Dict[str, int]
    exact_duplicate_rows: int
    duplicate_titles: int
    duplicate_lp_texts: int
    duplicate_title_lp_pairs: int
    label_distribution: Dict[str, int]
    label_proportions: Dict[str, float]
    title_length_stats: TextLengthStats
    lp_length_stats: TextLengthStats


@dataclass
class CrossSplitOverlapReport:
    """Overlap analysis between split pairs to detect potential data leakage."""

    pair_name: str
    split1_size: int
    split2_size: int
    exact_pair_overlap_count: int
    exact_pair_overlap_pct_of_split2: float
    lp_overlap_count: int
    lp_overlap_pct_of_split2: float
    title_overlap_count: int
    title_overlap_pct_of_split2: float


@dataclass
class DatasetAuditSummary:
    """Overall dataset audit compilation."""

    splits: Dict[str, SplitAuditReport]
    cross_split_overlaps: List[CrossSplitOverlapReport]
    leakage_warning: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit summary to serializable dictionary."""
        return asdict(self)


def compute_text_length_stats(series: pd.Series) -> TextLengthStats:
    """Compute character length distribution metrics for a pandas Series."""
    clean_series = series.fillna("").astype(str)
    lengths = clean_series.str.len().to_numpy()
    if len(lengths) == 0:
        return TextLengthStats(
            min=0, max=0, mean=0.0, median=0.0, std=0.0, q25=0.0, q75=0.0, iqr=0.0
        )

    q25 = float(np.percentile(lengths, 25))
    q75 = float(np.percentile(lengths, 75))
    return TextLengthStats(
        min=int(np.min(lengths)),
        max=int(np.max(lengths)),
        mean=round(float(np.mean(lengths)), 2),
        median=round(float(np.median(lengths)), 2),
        std=round(float(np.std(lengths)), 2),
        q25=round(q25, 2),
        q75=round(q75, 2),
        iqr=round(q75 - q25, 2),
    )


def audit_single_split(df: pd.DataFrame, split_name: str) -> SplitAuditReport:
    """Audit data quality, nulls, duplicates, and distributions for one split."""
    validate_schema(df, split_name=split_name)
    total_rows = len(df)

    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
    empty_strings = {
        col: int((df[col].astype(str).str.strip() == "").sum()) for col in ["title", "lp_text"]
    }

    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_titles = int(df.duplicated(subset=["title"]).sum())
    duplicate_lp_texts = int(df.duplicated(subset=["lp_text"]).sum())
    duplicate_title_lp_pairs = int(df.duplicated(subset=["title", "lp_text"]).sum())

    label_counts = df["label"].value_counts().to_dict()
    label_dist = {str(k): int(v) for k, v in label_counts.items()}
    label_props = {
        str(k): round(float(v) / max(total_rows, 1), 4) for k, v in label_counts.items()
    }

    title_stats = compute_text_length_stats(df["title"])
    lp_stats = compute_text_length_stats(df["lp_text"])

    return SplitAuditReport(
        split_name=split_name,
        total_rows=total_rows,
        null_counts=null_counts,
        empty_strings=empty_strings,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_titles=duplicate_titles,
        duplicate_lp_texts=duplicate_lp_texts,
        duplicate_title_lp_pairs=duplicate_title_lp_pairs,
        label_distribution=label_dist,
        label_proportions=label_props,
        title_length_stats=title_stats,
        lp_length_stats=lp_stats,
    )


def analyze_cross_split_overlap(
    df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str
) -> CrossSplitOverlapReport:
    """Check for identical pairs, landing pages, or titles between two splits."""
    pair_name = f"{name1}_vs_{name2}"
    size1 = len(df1)
    size2 = len(df2)

    if size1 == 0 or size2 == 0:
        return CrossSplitOverlapReport(
            pair_name=pair_name,
            split1_size=size1,
            split2_size=size2,
            exact_pair_overlap_count=0,
            exact_pair_overlap_pct_of_split2=0.0,
            lp_overlap_count=0,
            lp_overlap_pct_of_split2=0.0,
            title_overlap_count=0,
            title_overlap_pct_of_split2=0.0,
        )

    # 1. Exact (title, lp_text) pair overlap
    pairs1 = set(zip(df1["title"].astype(str), df1["lp_text"].astype(str)))
    pairs2 = set(zip(df2["title"].astype(str), df2["lp_text"].astype(str)))
    pair_overlap = len(pairs1.intersection(pairs2))

    # 2. LP text overlap
    lp1 = set(df1["lp_text"].astype(str))
    lp2 = set(df2["lp_text"].astype(str))
    lp_overlap = len(lp1.intersection(lp2))

    # 3. Title overlap
    titles1 = set(df1["title"].astype(str))
    titles2 = set(df2["title"].astype(str))
    title_overlap = len(titles1.intersection(titles2))

    return CrossSplitOverlapReport(
        pair_name=pair_name,
        split1_size=size1,
        split2_size=size2,
        exact_pair_overlap_count=pair_overlap,
        exact_pair_overlap_pct_of_split2=round((pair_overlap / max(size2, 1)) * 100, 2),
        lp_overlap_count=lp_overlap,
        lp_overlap_pct_of_split2=round((lp_overlap / max(len(lp2), 1)) * 100, 2),
        title_overlap_count=title_overlap,
        title_overlap_pct_of_split2=round((title_overlap / max(len(titles2), 1)) * 100, 2),
    )


def run_dataset_audit(dataset: ConsistencyDataset) -> DatasetAuditSummary:
    """Run comprehensive audit on all splits and cross-split intersections."""
    split_reports = {
        "train": audit_single_split(dataset.train, "train"),
        "valid": audit_single_split(dataset.valid, "valid"),
        "test": audit_single_split(dataset.test, "test"),
    }

    overlaps = [
        analyze_cross_split_overlap(dataset.train, dataset.valid, "train", "valid"),
        analyze_cross_split_overlap(dataset.train, dataset.test, "train", "test"),
        analyze_cross_split_overlap(dataset.valid, dataset.test, "valid", "test"),
    ]

    notes = []
    leakage_warning = False

    for ov in overlaps:
        if ov.exact_pair_overlap_count > 0:
            notes.append(
                f"CRITICAL: Exact (title, lp_text) pair overlap detected in {ov.pair_name}: "
                f"{ov.exact_pair_overlap_count} identical pairs."
            )
            leakage_warning = True
        if ov.lp_overlap_count > 0:
            notes.append(
                f"NOTE: Landing-page text overlap detected in {ov.pair_name}: "
                f"{ov.lp_overlap_count} unique LPs shared ({ov.lp_overlap_pct_of_split2}% of split2 LPs). "
                "Evaluate whether LP-level clustering leakage affects generalization."
            )

    return DatasetAuditSummary(
        splits=split_reports,
        cross_split_overlaps=overlaps,
        leakage_warning=leakage_warning,
        notes=notes,
    )


def format_audit_markdown(summary: DatasetAuditSummary) -> str:
    """Format audit summary into a clean GitHub Flavored Markdown document."""
    lines = [
        "# Dataset Audit Report: AdTEC Ad-Consistency",
        "",
        "## 1. Summary by Split",
        "",
        "| Split | Total Rows | Nulls (T/LP/L) | Duplicates (Rows) | Title Length (Med [IQR]) | LP Length (Med [IQR]) | Consistent % | Inconsistent % |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name, r in summary.splits.items():
        null_str = f"{r.null_counts.get('title', 0)}/{r.null_counts.get('lp_text', 0)}/{r.null_counts.get('label', 0)}"
        t_stat = f"{r.title_length_stats.median:.0f} [{r.title_length_stats.iqr:.0f}]"
        lp_stat = f"{r.lp_length_stats.median:.0f} [{r.lp_length_stats.iqr:.0f}]"
        c_pct = f"{r.label_proportions.get('consistent', 0.0)*100:.1f}%"
        i_pct = f"{r.label_proportions.get('inconsistent', 0.0)*100:.1f}%"
        lines.append(
            f"| {name} | {r.total_rows} | {null_str} | {r.exact_duplicate_rows} | {t_stat} | {lp_stat} | {c_pct} | {i_pct} |"
        )

    lines.extend([
        "",
        "## 2. Cross-Split Overlap and Leakage Check",
        "",
        "| Comparison | Exact Pair Overlap | Shared LP Texts | Shared Titles |",
        "|---|---|---|---|",
    ])

    for ov in summary.cross_split_overlaps:
        lines.append(
            f"| {ov.pair_name} | {ov.exact_pair_overlap_count} ({ov.exact_pair_overlap_pct_of_split2}%) | "
            f"{ov.lp_overlap_count} ({ov.lp_overlap_pct_of_split2}%) | "
            f"{ov.title_overlap_count} ({ov.title_overlap_pct_of_split2}%) |"
        )

    lines.extend(["", "## 3. Findings and Notes", ""])
    if summary.notes:
        for note in summary.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No data leakage or critical duplicate issues detected.")

    return "\n".join(lines)
