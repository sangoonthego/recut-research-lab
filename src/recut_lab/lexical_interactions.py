"""Reusable pairwise lexical interaction features and fold-safe overlap subgroup binning.

Implements explicit directional lexical overlap features for Japanese advertising
titles and landing-page texts (NFKC-normalized), a scikit-learn transformer for
fold-safe feature extraction, and training-fold quantile subgroup binning with
explicit tie/zero-heavy handling.

Important scientific note:
    All features in this module are surface-level descriptive proxies for lexical
    overlap, NOT semantic consistency or factual claim verification.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from recut_lab.features import compute_char_ngram_set
from recut_lab.preprocessing import normalize_japanese_text

LEXICAL_INTERACTION_FEATURE_NAMES: Tuple[str, ...] = (
    "char_bigram_jaccard",
    "char_trigram_jaccard",
    "title_to_lp_bigram_containment",
    "lp_to_title_bigram_containment",
    "title_to_lp_trigram_containment",
    "lp_to_title_trigram_containment",
    "title_to_lp_char_containment",
    "lp_to_title_char_containment",
    "char_length_ratio",
    "log_char_length_ratio",
    "shared_char_count",
    "shared_bigram_count",
    "shared_trigram_count",
    "title_char_length",
    "lp_char_length",
)

OVERLAP_BIN_ORDER: Tuple[str, ...] = ("Low", "Medium", "High")

# Objective surface-pattern detector for digits/numeric quantity expressions (NFKC normalized)
_NUMERIC_PATTERN = re.compile(
    r"[0-9]+(?:[.,][0-9]+)?(?:\s*(?:%|円|万円|億|倍|日|週間|ヶ月|か月|年|回|歳|時間|分|秒|mg|ml|g|kg|cm|mm|km|個|本|名|人|件|社|種|割|点))?"
)


def has_numeric_or_quantity_token(text: Optional[str]) -> bool:
    """Return True if NFKC-normalized text contains any digit/numeric expression.

    Used strictly as an objective descriptive flag for error-slice stratification;
    never interpreted as a numerical contradiction without manual text review.
    """
    clean = normalize_japanese_text(text)
    return bool(_NUMERIC_PATTERN.search(clean))


def _precompute_text_sets(clean_text: str) -> Tuple[int, set[str], set[str], set[str]]:
    """Compute character length and unigram/bigram/trigram sets for an NFKC string."""
    length = len(clean_text)
    chars = set(clean_text)
    if length == 0:
        return 0, chars, set(), set()
    bigrams = {clean_text[i : i + 2] for i in range(length - 1)} if length >= 2 else {clean_text}
    trigrams = {clean_text[i : i + 3] for i in range(length - 2)} if length >= 3 else {clean_text}
    return length, chars, bigrams, trigrams


def compute_pairwise_lexical_features(title: Optional[str], lp_text: Optional[str]) -> Dict[str, float]:
    """Compute explicit symmetric and directional pairwise lexical overlap features.

    Containment direction is explicit:
        - ``title_to_lp_*_containment``: fraction of unique title n-grams present in LP.
        - ``lp_to_title_*_containment``: fraction of unique LP n-grams present in title.
    """
    t_clean = normalize_japanese_text(title)
    lp_clean = normalize_japanese_text(lp_text)

    t_len, t_chars, t_bi, t_tri = _precompute_text_sets(t_clean)
    lp_len, lp_chars, lp_bi, lp_tri = _precompute_text_sets(lp_clean)

    shared_chars = len(t_chars.intersection(lp_chars))
    shared_bi = len(t_bi.intersection(lp_bi))
    shared_tri = len(t_tri.intersection(lp_tri))

    union_bi = len(t_bi.union(lp_bi))
    union_tri = len(t_tri.union(lp_tri))

    bigram_jaccard = shared_bi / max(union_bi, 1)
    trigram_jaccard = shared_tri / max(union_tri, 1)

    t2lp_bi = shared_bi / max(len(t_bi), 1)
    lp2t_bi = shared_bi / max(len(lp_bi), 1)

    t2lp_tri = shared_tri / max(len(t_tri), 1)
    lp2t_tri = shared_tri / max(len(lp_tri), 1)

    t2lp_ch = shared_chars / max(len(t_chars), 1)
    lp2t_ch = shared_chars / max(len(lp_chars), 1)

    len_ratio = t_len / max(lp_len, 1)
    log_len_ratio = math.log1p(t_len) - math.log1p(lp_len)

    return {
        "char_bigram_jaccard": float(bigram_jaccard),
        "char_trigram_jaccard": float(trigram_jaccard),
        "title_to_lp_bigram_containment": float(t2lp_bi),
        "lp_to_title_bigram_containment": float(lp2t_bi),
        "title_to_lp_trigram_containment": float(t2lp_tri),
        "lp_to_title_trigram_containment": float(lp2t_tri),
        "title_to_lp_char_containment": float(t2lp_ch),
        "lp_to_title_char_containment": float(lp2t_ch),
        "char_length_ratio": float(len_ratio),
        "log_char_length_ratio": float(log_len_ratio),
        "shared_char_count": float(shared_chars),
        "shared_bigram_count": float(shared_bi),
        "shared_trigram_count": float(shared_tri),
        "title_char_length": float(t_len),
        "lp_char_length": float(lp_len),
    }


def extract_lexical_interaction_frame(
    df: pd.DataFrame,
    title_col: str = "title",
    lp_col: str = "lp_text",
) -> pd.DataFrame:
    """Compute pairwise lexical interaction features efficiently across a DataFrame.

    Precomputes normalized n-gram sets once per unique normalized landing page to
    avoid redundant work on repeated landing pages.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    titles_norm = [normalize_japanese_text(t) for t in df[title_col].tolist()]
    lps_norm = [normalize_japanese_text(lp) for lp in df[lp_col].tolist()]

    # Cache n-gram sets per unique normalized LP (label-independent, stateless)
    lp_cache: Dict[str, Tuple[int, set[str], set[str], set[str]]] = {}
    for lp_clean in set(lps_norm):
        lp_cache[lp_clean] = _precompute_text_sets(lp_clean)

    rows = np.empty((len(df), len(LEXICAL_INTERACTION_FEATURE_NAMES)), dtype=np.float32)

    for idx, (t_clean, lp_clean) in enumerate(zip(titles_norm, lps_norm)):
        t_len, t_chars, t_bi, t_tri = _precompute_text_sets(t_clean)
        lp_len, lp_chars, lp_bi, lp_tri = lp_cache[lp_clean]

        shared_chars = len(t_chars.intersection(lp_chars))
        shared_bi = len(t_bi.intersection(lp_bi))
        shared_tri = len(t_tri.intersection(lp_tri))

        union_bi = len(t_bi) + len(lp_bi) - shared_bi
        union_tri = len(t_tri) + len(lp_tri) - shared_tri

        rows[idx, 0] = shared_bi / max(union_bi, 1)
        rows[idx, 1] = shared_tri / max(union_tri, 1)
        rows[idx, 2] = shared_bi / max(len(t_bi), 1)
        rows[idx, 3] = shared_bi / max(len(lp_bi), 1)
        rows[idx, 4] = shared_tri / max(len(t_tri), 1)
        rows[idx, 5] = shared_tri / max(len(lp_tri), 1)
        rows[idx, 6] = shared_chars / max(len(t_chars), 1)
        rows[idx, 7] = shared_chars / max(len(lp_chars), 1)
        rows[idx, 8] = t_len / max(lp_len, 1)
        rows[idx, 9] = math.log1p(t_len) - math.log1p(lp_len)
        rows[idx, 10] = float(shared_chars)
        rows[idx, 11] = float(shared_bi)
        rows[idx, 12] = float(shared_tri)
        rows[idx, 13] = float(t_len)
        rows[idx, 14] = float(lp_len)

    return pd.DataFrame(rows, columns=list(LEXICAL_INTERACTION_FEATURE_NAMES), index=df.index)


class LexicalInteractionTransformer(BaseEstimator, TransformerMixin):
    """Stateless scikit-learn transformer extracting pairwise lexical interaction features."""

    def __init__(
        self,
        title_col: str = "title",
        lp_col: str = "lp_text",
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.title_col = title_col
        self.lp_col = lp_col
        self.feature_names = (
            tuple(feature_names) if feature_names is not None else LEXICAL_INTERACTION_FEATURE_NAMES
        )

    def fit(self, X: pd.DataFrame, y=None) -> LexicalInteractionTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = extract_lexical_interaction_frame(X, title_col=self.title_col, lp_col=self.lp_col)
        cols = list(getattr(self, "feature_names", LEXICAL_INTERACTION_FEATURE_NAMES))
        arr = frame[cols].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite values detected in lexical interaction features.")
        return arr

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cols = list(getattr(self, "feature_names", LEXICAL_INTERACTION_FEATURE_NAMES))
        return np.asarray(cols, dtype=object)


# =============================================================================
# FOLD-SAFE QUANTILE OVERLAP SUBGROUP BINNING (WITH EXPLICIT TIE HANDLING)
# =============================================================================


def fit_overlap_bin_edges(
    train_values: Sequence[float] | np.ndarray,
    metric_name: str,
) -> Dict[str, Any]:
    """Derive Low/Medium/High lexical overlap boundaries strictly from training-fold quantiles.

    Never inspects held-out data or model performance. Explicitly handles zero-heavy,
    one-heavy, and tied quantile distributions deterministically.
    """
    arr = np.asarray(train_values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot fit overlap bin edges on an empty array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Overlap values contain NaN or Inf.")

    q1 = float(np.quantile(arr, 1.0 / 3.0))
    q2 = float(np.quantile(arr, 2.0 / 3.0))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    zero_frac = float(np.mean(arr == 0.0))
    one_frac = float(np.mean(arr >= 1.0 - 1e-12))

    if min_val == max_val:
        rule = "degenerate_constant_all_low"
        low_cut = min_val
        high_cut = max_val
    elif q1 < q2:
        rule = "standard_tertiles"
        low_cut = q1
        high_cut = q2
    elif q1 == min_val:
        # Zero-heavy or lower-bound heavy distribution (e.g., >= 66.7% at min_val)
        above = arr[arr > min_val]
        if above.size > 0:
            high_cut = float(np.quantile(above, 0.5))
            rule = "lower_tie_split_positive_median"
        else:
            high_cut = min_val
            rule = "degenerate_constant_all_low"
        low_cut = min_val
    elif q2 == max_val:
        # Upper-bound heavy distribution (e.g., >= 66.7% at 1.0 containment)
        below = arr[arr < max_val]
        if below.size > 0:
            low_cut = float(np.quantile(below, 0.5))
            rule = "upper_tie_split_below_max_median"
        else:
            low_cut = max_val
            rule = "degenerate_constant_all_low"
        high_cut = max_val
    else:
        # Interior tie: q1 == q2 strictly between min and max
        rule = "interior_tie_three_way"
        low_cut = q1
        high_cut = q2

    return {
        "metric_name": metric_name,
        "rule": rule,
        "raw_tertile_q1": round(q1, 6),
        "raw_tertile_q2": round(q2, 6),
        "low_cut": round(low_cut, 6),
        "high_cut": round(high_cut, 6),
        "train_min": round(min_val, 6),
        "train_max": round(max_val, 6),
        "train_median": round(float(np.median(arr)), 6),
        "train_zero_fraction": round(zero_frac, 4),
        "train_one_fraction": round(one_frac, 4),
        "train_sample_count": int(arr.size),
    }


def assign_overlap_bins(
    values: Sequence[float] | np.ndarray,
    bin_spec: Dict[str, Any],
) -> np.ndarray:
    """Assign each sample to 'Low', 'Medium', or 'High' using fixed training-fold edges."""
    arr = np.asarray(values, dtype=np.float64)
    rule = bin_spec["rule"]
    low_cut = float(bin_spec["low_cut"])
    high_cut = float(bin_spec["high_cut"])

    bins = np.empty(arr.shape[0], dtype=object)

    if rule == "degenerate_constant_all_low":
        bins[:] = "Low"
    elif rule in ("standard_tertiles", "lower_tie_split_positive_median"):
        # Low: x <= low_cut; Medium: low_cut < x <= high_cut; High: x > high_cut
        bins[arr <= low_cut] = "Low"
        bins[(arr > low_cut) & (arr <= high_cut)] = "Medium"
        bins[arr > high_cut] = "High"
    elif rule == "upper_tie_split_below_max_median":
        # Low: x <= low_cut; Medium: low_cut < x < high_cut; High: x >= high_cut
        bins[arr <= low_cut] = "Low"
        bins[(arr > low_cut) & (arr < high_cut)] = "Medium"
        bins[arr >= high_cut] = "High"
    elif rule == "interior_tie_three_way":
        # Low: x < low_cut; Medium: x == low_cut; High: x > low_cut
        bins[arr < low_cut] = "Low"
        bins[arr == low_cut] = "Medium"
        bins[arr > low_cut] = "High"
    else:
        raise ValueError(f"Unknown overlap bin rule: '{rule}'")

    # Verify complete deterministic assignment
    if np.any(pd.isna(bins)):
        raise AssertionError("Unassigned overlap bin detected.")
    return bins
