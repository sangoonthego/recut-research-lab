"""Unit tests for M2.0.1 failure diagnostics: D1 score extraction/PR/ROC, D2 thresholds/CV, D3 representations, D4 LP grouping."""

import numpy as np
import pandas as pd
import pytest

from recut_lab.baseline import TfidfLogRegBaseline
from recut_lab.data import create_synthetic_dataset
from recut_lab.diagnostics import (
    SeparateBlocksTfidfLogReg,
    TruncatedLpPairBaseline,
    build_threshold_grid,
    compute_d1_score_diagnostics,
    compute_histogram_overlap,
    compute_oof_scores_for_pipeline,
    compute_seen_unseen_confounding,
    evaluate_threshold_grid,
    evaluate_threshold_point,
    extract_class_scores,
    label_seen_unseen_lps,
    select_threshold_from_oof,
)


class _SwappedOrderModel:
    """Mock classifier exposing reversed classes_ order ['inconsistent', 'consistent']."""

    def __init__(self, probs: np.ndarray):
        self.classes_ = np.array(["inconsistent", "consistent"])
        self._probs = probs

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._probs


def test_extract_class_scores_dynamic_order():
    probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    mdl = _SwappedOrderModel(probs)
    df = pd.DataFrame({"title": ["A", "B"], "lp_text": ["X", "Y"]})

    inc_scores = extract_class_scores(mdl, df, target_class="inconsistent")
    cons_scores = extract_class_scores(mdl, df, target_class="consistent")
    np.testing.assert_allclose(inc_scores, [0.8, 0.1])
    np.testing.assert_allclose(cons_scores, [0.2, 0.9])

    with pytest.raises(ValueError, match="not found"):
        extract_class_scores(mdl, df, target_class="unknown")


def test_d1_roc_pr_and_constant_score_handling():
    y_true = ["consistent", "consistent", "inconsistent", "inconsistent"]
    perfect_scores = [0.1, 0.2, 0.8, 0.9]
    rep = compute_d1_score_diagnostics(y_true, perfect_scores, model_id="perfect")
    assert rep["roc_auc"] == 1.0
    assert rep["average_precision"] == 1.0
    assert rep["prevalence_inconsistent"] == 0.5
    assert rep["is_constant_score"] is False
    assert rep["predicted_inconsistent_at_default"] == 2
    assert rep["histogram_overlap_20bins"] == 0.0

    # Constant-score reference (B0_majority)
    const_rep = compute_d1_score_diagnostics(y_true, [0.0, 0.0, 0.0, 0.0], model_id="B0_majority")
    assert const_rep["is_constant_score"] is True
    assert const_rep["roc_auc"] == 0.5
    assert const_rep["average_precision"] == 0.5
    assert const_rep["predicted_inconsistent_at_default"] == 0
    assert compute_histogram_overlap(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 1.0


def test_d2_threshold_calculations():
    grid = build_threshold_grid(0.1, 0.9, 0.2)
    assert grid == [0.1, 0.3, 0.5, 0.7, 0.9]

    y_true = ["consistent", "consistent", "inconsistent", "inconsistent"]
    scores = [0.15, 0.40, 0.35, 0.85]

    pt_default = evaluate_threshold_point(y_true, scores, 0.50, "M", "default_0.50")
    assert pt_default["predicted_inconsistent_count"] == 1
    assert pt_default["precision_inconsistent"] == 1.0
    assert pt_default["recall_inconsistent"] == 0.5
    assert pt_default["balanced_accuracy"] == 0.75

    pt_low = evaluate_threshold_point(y_true, scores, 0.30, "M", "valid_exploratory_grid")
    # scores >= 0.30 -> indices 1 (cons), 2 (inc), 3 (inc): 2 TP, 1 FP, 0 FN
    assert pt_low["predicted_inconsistent_count"] == 3
    assert pt_low["recall_inconsistent"] == 1.0
    assert pt_low["precision_inconsistent"] == pytest.approx(2 / 3, abs=1e-4)

    grid_df = evaluate_threshold_grid(y_true, scores, "M", grid)
    assert len(grid_df) == 5
    assert set(grid_df["selection_source"]) == {"valid_exploratory_grid"}


def test_d2_deterministic_oof_cv_behavior():
    ds = create_synthetic_dataset(num_samples_per_split=20)
    builder = lambda: TfidfLogRegBaseline(
        input_mode="title", ngram_range=(2, 3), min_df=1, max_features=200, random_state=42
    )._build_pipeline()

    oof1 = compute_oof_scores_for_pipeline(ds.train, builder, n_splits=4, random_state=42)
    oof2 = compute_oof_scores_for_pipeline(ds.train, builder, n_splits=4, random_state=42)
    np.testing.assert_allclose(oof1, oof2)

    thr1, _, _ = select_threshold_from_oof(ds.train["label"], oof1, [0.2, 0.5, 0.8], "B1")
    thr2, _, _ = select_threshold_from_oof(ds.train["label"], oof2, [0.2, 0.5, 0.8], "B1")
    assert thr1 == thr2


def test_d3_representation_feature_isolation_and_truncation():
    train_df = pd.DataFrame({
        "label": ["consistent", "inconsistent"] * 10,
        "title": ["Good Product A", "Mismatch Claim B"] * 10,
        "lp_text": [
            "PREFIX_A " + ("detail_good " * 15),
            "PREFIX_B " + ("detail_bad " * 15),
        ] * 10,
    })

    # R1/R2 separate blocks preserve sparse structure and report block norms
    r2 = SeparateBlocksTfidfLogReg(
        title_weight=2.0, lp_weight=1.0, ngram_range=(2, 3), min_df=1, max_features_per_block=200
    )
    r2.fit(train_df)
    diag = r2.get_feature_diagnostics(train_df.iloc[:4])
    assert diag["total_features"] == diag["title_features"] + diag["lp_features"]
    assert diag["title_features"] > 0 and diag["lp_features"] > 0
    # Weighted title block should have L2 norm ~= 2.0 (since TF-IDF rows are unit L2 normalized)
    assert diag["mean_title_block_l2_norm"] == pytest.approx(2.0, abs=0.05)
    assert diag["mean_lp_block_l2_norm"] == pytest.approx(1.0, abs=0.05)
    assert pytest.approx(diag["title_logit_share"] + diag["lp_logit_share"], abs=1e-3) == 1.0

    # R3 truncation strictly ignores characters beyond lp_max_chars
    r3 = TruncatedLpPairBaseline(lp_max_chars=12, ngram_range=(2, 3), min_df=1, max_features=200)
    r3.fit(train_df)
    df_a = pd.DataFrame({"title": ["Good Product A"], "lp_text": ["PREFIX_A 123_TAIL_ONE"]})
    df_b = pd.DataFrame({"title": ["Good Product A"], "lp_text": ["PREFIX_A 123_COMPLETELY_DIFFERENT"]})
    np.testing.assert_allclose(r3.predict_proba(df_a), r3.predict_proba(df_b))


def test_d4_seen_unseen_lp_grouping_and_confounding():
    # Test NFKC normalization (full-width '１２３' matches half-width '123')
    train_df = pd.DataFrame({
        "label": ["consistent", "inconsistent"],
        "title": ["T1", "T2"],
        "lp_text": ["LP text 123", "LP text ABC"],
    })
    valid_df = pd.DataFrame({
        "label": ["consistent", "inconsistent", "consistent", "inconsistent"],
        "title": ["V1", "V2", "V3", "V4"],
        "lp_text": ["LP text １２３", "LP text 123", "Unseen LP 1", "Unseen LP 2"],
    })
    groups = label_seen_unseen_lps(train_df, valid_df)
    assert groups.tolist() == ["seen_lp", "seen_lp", "unseen_lp", "unseen_lp"]

    preds_df = pd.DataFrame({
        "valid_row_index": [0, 1, 2, 3],
        "model_id": ["B0_majority"] * 4,
        "true_label": ["consistent", "inconsistent", "consistent", "inconsistent"],
        "predicted_label": ["consistent", "consistent", "consistent", "consistent"],
        "is_error": [False, True, False, True],
    })
    conf_df = compute_seen_unseen_confounding(train_df, valid_df, preds_df)
    seen_row = conf_df[conf_df["lp_group"] == "seen_lp"].iloc[0]
    unseen_row = conf_df[conf_df["lp_group"] == "unseen_lp"].iloc[0]
    assert seen_row["n_samples"] == 2
    assert seen_row["model_error_rate"] == 0.5
    assert seen_row["balanced_accuracy"] == 0.5
    assert unseen_row["n_samples"] == 2
    assert unseen_row["inconsistent_pct"] == 50.0
    assert unseen_row["train_majority_reference_error"] == 0.5
    assert unseen_row["within_group_majority_error"] == 0.5
