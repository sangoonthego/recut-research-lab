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


VALID_INPUT_MODES = ("title", "lp_text", "pair")


class TextInputSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer selecting title, lp_text, or a deterministic pair string.

    Supports:
        - ``"title"``: Extracts and optionally normalizes advertising titles only (B1).
        - ``"lp_text"``: Extracts and optionally normalizes landing-page texts only (B2).
        - ``"pair"``: Concatenates title and landing-page text with an explicit separator (B3).
    """

    def __init__(
        self,
        mode: str = "pair",
        title_col: str = "title",
        lp_col: str = "lp_text",
        separator: str = " [SEP] ",
        normalize: bool = True,
    ):
        self.mode = mode
        self.title_col = title_col
        self.lp_col = lp_col
        self.separator = separator
        self.normalize = normalize

    def _validate_mode(self) -> None:
        if self.mode not in VALID_INPUT_MODES:
            raise ValueError(
                f"Invalid input mode '{self.mode}'. Expected one of {VALID_INPUT_MODES}."
            )

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_mode()
        return self

    def transform(self, X: pd.DataFrame) -> List[str]:
        """Extract configured text input from DataFrame rows."""
        self._validate_mode()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        if self.mode == "title":
            if self.title_col not in X.columns:
                raise ValueError(f"Missing column '{self.title_col}' for mode='title'.")
            return [
                normalize_japanese_text(t) if self.normalize else str(t)
                for t in X[self.title_col].tolist()
            ]

        if self.mode == "lp_text":
            if self.lp_col not in X.columns:
                raise ValueError(f"Missing column '{self.lp_col}' for mode='lp_text'.")
            return [
                normalize_japanese_text(lp) if self.normalize else str(lp)
                for lp in X[self.lp_col].tolist()
            ]

        # mode == "pair"
        for col in (self.title_col, self.lp_col):
            if col not in X.columns:
                raise ValueError(f"Missing column '{col}' for mode='pair'.")
        return [
            combine_ad_inputs(t, lp, separator=self.separator, normalize=self.normalize)
            for t, lp in zip(X[self.title_col].tolist(), X[self.lp_col].tolist())
        ]

