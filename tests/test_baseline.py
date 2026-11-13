"""Unit tests for baseline classification models and serialization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from recut_lab.baseline import (
    MajorityBaseline,
    TfidfLogRegBaseline,
    get_baseline_model,
)
from recut_lab.data import create_synthetic_dataset


def test_majority_baseline():
    train_df = pd.DataFrame({
        "label": ["consistent", "consistent", "inconsistent"],
        "title": ["T1", "T2", "T3"],
        "lp_text": ["L1", "L2", "L3"],
    })
    test_df = pd.DataFrame({
        "label": ["inconsistent", "consistent"],
        "title": ["T4", "T5"],
        "lp_text": ["L4", "L5"],
    })

    model = MajorityBaseline()
    model.fit(train_df)
    preds = model.predict(test_df)
    assert preds == ["consistent", "consistent"]

    probs = model.predict_proba(test_df)
    assert probs is not None
    assert probs.shape == (2, 2)


def test_tfidf_logreg_baseline(tmp_path: Path):
    ds = create_synthetic_dataset(num_samples_per_split=20)

    model = TfidfLogRegBaseline(
        ngram_range=(1, 2),
        min_df=1,
        max_features=500,
        random_state=42,
    )
    model.fit(ds.train)
    preds = model.predict(ds.valid)
    assert len(preds) == len(ds.valid)
    assert set(preds).issubset({"consistent", "inconsistent"})

    probs = model.predict_proba(ds.valid)
    assert probs is not None
    assert probs.shape == (len(ds.valid), 2)

    # Test serialization
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    loaded_model = TfidfLogRegBaseline.load(save_path)
    loaded_preds = loaded_model.predict(ds.valid)
    assert loaded_preds == preds


def test_get_baseline_model_factory():
    maj = get_baseline_model("majority")
    assert isinstance(maj, MajorityBaseline)

    tfidf = get_baseline_model("tfidf_logreg", min_df=1)
    assert isinstance(tfidf, TfidfLogRegBaseline)

    with pytest.raises(ValueError, match="Unknown baseline model"):
        get_baseline_model("unsupported_model_type")

