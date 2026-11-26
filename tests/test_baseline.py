"""Unit tests for baseline classification models, input selectors, and M2.0 benchmark reproducibility."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from recut_lab.analysis import build_error_inspection_table
from recut_lab.baseline import (
    M2_DEFAULT_CONFIG,
    MajorityBaseline,
    TfidfLogRegBaseline,
    get_baseline_model,
    run_m2_baseline_benchmark,
)
from recut_lab.data import create_synthetic_dataset
from recut_lab.preprocessing import TextInputSelector


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


def test_text_input_selector_modes():
    df = pd.DataFrame({
        "title": ["Title One", "Title Two"],
        "lp_text": ["Landing Page A", "Landing Page B"],
    })

    sel_title = TextInputSelector(mode="title", normalize=False)
    assert sel_title.fit_transform(df) == ["Title One", "Title Two"]

    sel_lp = TextInputSelector(mode="lp_text", normalize=False)
    assert sel_lp.fit_transform(df) == ["Landing Page A", "Landing Page B"]

    sel_pair = TextInputSelector(mode="pair", separator=" [SEP] ", normalize=False)
    assert sel_pair.fit_transform(df) == [
        "Title One [SEP] Landing Page A",
        "Title Two [SEP] Landing Page B",
    ]

    with pytest.raises(ValueError, match="Invalid input mode"):
        TextInputSelector(mode="invalid_mode").fit(df)


def test_b1_b2_b3_input_isolation():
    """Verify B1 ignores lp_text changes and B2 ignores title changes at inference."""
    train_df = pd.DataFrame({
        "label": ["consistent", "consistent", "inconsistent", "inconsistent"] * 4,
        "title": ["Good Ad A", "Good Ad B", "Bad Title X", "Bad Title Y"] * 4,
        "lp_text": ["Relevant LP A", "Relevant LP B", "Unrelated Page Z", "Unrelated Page W"] * 4,
    })

    b1 = TfidfLogRegBaseline(input_mode="title", ngram_range=(2, 3), min_df=1, max_features=200)
    b2 = TfidfLogRegBaseline(input_mode="lp_text", ngram_range=(2, 3), min_df=1, max_features=200)
    b3 = TfidfLogRegBaseline(input_mode="pair", ngram_range=(2, 3), min_df=1, max_features=200)

    b1.fit(train_df)
    b2.fit(train_df)
    b3.fit(train_df)

    eval_base = pd.DataFrame({"title": ["Good Ad A"], "lp_text": ["Relevant LP A"]})
    eval_lp_mutated = pd.DataFrame({"title": ["Good Ad A"], "lp_text": ["COMPLETELY DIFFERENT 999"]})
    eval_title_mutated = pd.DataFrame({"title": ["COMPLETELY DIFFERENT 999"], "lp_text": ["Relevant LP A"]})

    # B1 must produce identical scores when only lp_text changes
    np.testing.assert_allclose(b1.predict_proba(eval_base), b1.predict_proba(eval_lp_mutated))
    # B2 must produce identical scores when only title changes
    np.testing.assert_allclose(b2.predict_proba(eval_base), b2.predict_proba(eval_title_mutated))
    # B3 should react to both title and lp_text
    assert b3.predict(eval_base)[0] in {"consistent", "inconsistent"}


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

    b1 = get_baseline_model("B1_title_only", min_df=1)
    assert isinstance(b1, TfidfLogRegBaseline)
    assert b1.input_mode == "title"

    b2 = get_baseline_model("B2_lp_only", min_df=1)
    assert isinstance(b2, TfidfLogRegBaseline)
    assert b2.input_mode == "lp_text"

    tfidf = get_baseline_model("tfidf_logreg", min_df=1)
    assert isinstance(tfidf, TfidfLogRegBaseline)
    assert tfidf.input_mode == "pair"

    with pytest.raises(ValueError, match="Unknown baseline model"):
        get_baseline_model("unsupported_model_type")


def test_m2_benchmark_reproducibility_and_serialization(tmp_path: Path):
    ds = create_synthetic_dataset(num_samples_per_split=24)
    cfg = {
        **M2_DEFAULT_CONFIG,
        "experiment_id": "test_m2_run",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [2, 3],
            "max_features": 500,
            "min_df": 1,
            "sublinear_tf": True,
        },
        "latency_benchmark": {
            "warmup_runs": 1,
            "timed_runs": 2,
            "method": "full_validation_batch_amortized_median",
        },
    }

    res1 = run_m2_baseline_benchmark(
        train_df=ds.train,
        valid_df=ds.valid,
        dataset_metadata={"upstream_revision": "test_rev", "splits": {}},
        config=cfg,
        runs_base_dir=tmp_path / "runs1",
        models_base_dir=tmp_path / "models1",
        reports_dir=tmp_path / "reports1",
    )
    res2 = run_m2_baseline_benchmark(
        train_df=ds.train,
        valid_df=ds.valid,
        dataset_metadata={"upstream_revision": "test_rev", "splits": {}},
        config=cfg,
        runs_base_dir=tmp_path / "runs2",
        models_base_dir=tmp_path / "models2",
        reports_dir=tmp_path / "reports2",
    )

    # Predictions and classification metrics must be bit-for-bit identical across runs
    pd.testing.assert_frame_equal(res1["predictions_df"], res2["predictions_df"])
    for col in [
        "model_id",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_inconsistent",
        "recall_inconsistent",
        "f1_inconsistent",
    ]:
        assert res1["benchmark_df"][col].tolist() == res2["benchmark_df"][col].tolist()

    # Persisted predictions_valid.csv must NOT contain raw text columns
    pred_csv = pd.read_csv(tmp_path / "runs1" / "test_m2_run" / "predictions_valid.csv")
    assert "title" not in pred_csv.columns
    assert "lp_text" not in pred_csv.columns

    # Metrics JSON and CSV must agree
    metrics_json = json.loads((tmp_path / "runs1" / "test_m2_run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_json["test_split_accessed"] is False
    for _, row in res1["benchmark_df"].iterrows():
        m_id = row["model_id"]
        assert metrics_json["models"][m_id]["evaluation"]["macro_f1"] == row["macro_f1"]

    # Error inspection table builder preserves required Table 2 columns
    err_tbl = build_error_inspection_table(ds.valid, res1["predictions_df"], model_id="B0_majority", limit=5)
    assert list(err_tbl.columns) == [
        "sample_id",
        "true_label",
        "predicted_label",
        "model_score",
        "lp_text",
        "title",
    ]
