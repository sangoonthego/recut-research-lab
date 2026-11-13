"""Feature extraction utilities for ad consistency classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from recut_lab.preprocessing import normalize_japanese_text


def build_char_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_features: Optional[int] = 25000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a character n-gram TF-IDF vectorizer optimized for Japanese text.

    Args:
        ngram_range: Tuple of (min_n, max_n) character n-grams.
        min_df: Minimum document frequency.
        max_features: Maximum vocabulary size.
        sublinear_tf: Apply sublinear scaling (1 + log(tf)).

    Returns:
        Configured scikit-learn TfidfVectorizer.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def compute_char_ngram_set(text: str, n: int = 2) -> set[str]:
    """Extract set of character n-grams from text."""
    clean = normalize_japanese_text(text)
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def calculate_lexical_overlap(title: str, lp_text: str) -> Dict[str, float]:
    """Compute basic surface-level lexical overlap features between title and LP.

    Returns:
        Dictionary of overlap metrics:
            - char_jaccard: Jaccard similarity of character unigrams.
            - bigram_jaccard: Jaccard similarity of character bigrams.
            - title_char_coverage: Proportion of title chars present in LP.
            - length_ratio: Ratio of title length to LP length.
    """
    t_clean = normalize_japanese_text(title)
    lp_clean = normalize_japanese_text(lp_text)

    t_chars = set(t_clean)
    lp_chars = set(lp_clean)

    union_chars = t_chars.union(lp_chars)
    char_jaccard = len(t_chars.intersection(lp_chars)) / max(len(union_chars), 1)

    t_bi = compute_char_ngram_set(t_clean, 2)
    lp_bi = compute_char_ngram_set(lp_clean, 2)
    union_bi = t_bi.union(lp_bi)
    bigram_jaccard = len(t_bi.intersection(lp_bi)) / max(len(union_bi), 1)

    # Coverage: what fraction of title characters appear in LP
    coverage = len(t_chars.intersection(lp_chars)) / max(len(t_chars), 1)

    len_ratio = len(t_clean) / max(len(lp_clean), 1)

    return {
        "char_jaccard": round(char_jaccard, 4),
        "bigram_jaccard": round(bigram_jaccard, 4),
        "title_char_coverage": round(coverage, 4),
        "length_ratio": round(len_ratio, 6),
    }


class LexicalOverlapExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer computing numerical overlap features from title & lp_text."""

    def __init__(self, title_col: str = "title", lp_col: str = "lp_text"):
        self.title_col = title_col
        self.lp_col = lp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract lexical overlap matrix."""
        rows = []
        for _, row in X.iterrows():
            feats = calculate_lexical_overlap(row[self.title_col], row[self.lp_col])
            rows.append(list(feats.values()))
        return np.array(rows, dtype=np.float32)
