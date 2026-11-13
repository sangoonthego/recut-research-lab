"""Text normalization and preprocessing utilities for Japanese ad and landing page text."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def normalize_japanese_text(text: Optional[str]) -> str:
    """Normalize Japanese text using Unicode NFKC and whitespace cleanup.

    - Applies NFKC normalization (converts full-width numbers/alphanumerics to half-width,
      half-width kana to full-width kana).
    - Collapses consecutive whitespace characters into a single space.
    - Strips leading and trailing whitespace.

    Args:
        text: Raw Japanese text string or None.

    Returns:
        Normalized string.
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Replace multiple whitespace/newlines with single space
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def combine_ad_inputs(
    title: str,
    lp_text: str,
    separator: str = " [SEP] ",
    normalize: bool = True,
) -> str:
    """Combine ad title and landing page text with a boundary separator.

    Args:
        title: Advertising title.
        lp_text: Landing page text.
        separator: Boundary separator string.
        normalize: Whether to apply Japanese NFKC normalization.

    Returns:
        Combined single string.
    """
    clean_title = normalize_japanese_text(title) if normalize else str(title)
    clean_lp = normalize_japanese_text(lp_text) if normalize else str(lp_text)
    return f"{clean_title}{separator}{clean_lp}"


class TextPairCombiner(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that combines title and lp_text columns."""

    def __init__(
        self,
        title_col: str = "title",
        lp_col: str = "lp_text",
        separator: str = " [SEP] ",
        normalize: bool = True,
    ):
        self.title_col = title_col
        self.lp_col = lp_col
        self.separator = separator
        self.normalize = normalize

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> List[str]:
        """Combine columns for each row into a list of strings."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        titles = X[self.title_col].tolist()
        lps = X[self.lp_col].tolist()

        combined = [
            combine_ad_inputs(t, lp, separator=self.separator, normalize=self.normalize)
            for t, lp in zip(titles, lps)
        ]
        return combined

