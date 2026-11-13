"""Baseline models for advertising consistency classification.

Implements:
1. Majority class baseline (predicts most frequent training class).
2. Character n-gram TF-IDF + Logistic Regression baseline.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from recut_lab.features import build_char_tfidf_vectorizer
from recut_lab.preprocessing import TextPairCombiner


class BaselineModel(ABC):
    """Abstract interface for recut_lab baseline classifiers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "label") -> BaselineModel:
        """Fit model on training DataFrame."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> List[str]:
        """Generate predicted string labels."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate class probability estimates if supported."""
        pass

    def save(self, file_path: Path | str) -> None:
        """Serialize model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path | str) -> BaselineModel:
        """Load serialized model from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path.resolve()}")
        with path.open("rb") as f:
            model = pickle.load(f)
        return model


class MajorityBaseline(BaselineModel):
    """Baseline predicting the majority class observed in the training split."""

    def __init__(self):
        self.dummy_clf = DummyClassifier(strategy="most_frequent")
        self.classes_: Optional[np.ndarray] = None
        self.majority_label_: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> MajorityBaseline:
        y = df[target_col].to_numpy()
        self.dummy_clf.fit(np.zeros((len(df), 1)), y)
        self.classes_ = self.dummy_clf.classes_
        self.majority_label_ = str(self.dummy_clf.predict([[0]])[0])
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        if self.majority_label_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return [self.majority_label_] * len(df)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.dummy_clf is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.dummy_clf.predict_proba(np.zeros((len(df), 1)))


class TfidfLogRegBaseline(BaselineModel):
    """Character n-gram TF-IDF Vectorizer + Logistic Regression classifier."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_features: Optional[int] = 25000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        title_col: str = "title",
        lp_col: str = "lp_text",
        separator: str = " [SEP] ",
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.title_col = title_col
        self.lp_col = lp_col
        self.separator = separator

        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        combiner = TextPairCombiner(
            title_col=self.title_col,
            lp_col=self.lp_col,
            separator=self.separator,
            normalize=True,
        )
        vectorizer = build_char_tfidf_vectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        classifier = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        return Pipeline([
            ("combiner", combiner),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> TfidfLogRegBaseline:
        self.pipeline = self._build_pipeline()
        y = df[target_col].to_numpy()
        self.pipeline.fit(df, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.pipeline.predict(df)
        return [str(p) for p in preds]

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.pipeline.predict_proba(df)


def get_baseline_model(model_type: str, **kwargs) -> BaselineModel:
    """Factory function for instantiating baseline models."""
    normalized = model_type.lower().strip()
    if normalized in ("majority", "majority_class", "dummy"):
        return MajorityBaseline()
    if normalized in ("tfidf_logreg", "tfidf_lr", "logistic_regression", "tfidf"):
        return TfidfLogRegBaseline(**kwargs)
    raise ValueError(f"Unknown baseline model type: '{model_type}'. Expected 'majority' or 'tfidf_logreg'.")

