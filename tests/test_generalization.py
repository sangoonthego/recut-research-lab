"""Unit tests for Milestone M2.1 (RQ1) generalization module on synthetic fixtures."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from recut_lab.generalization import (
    M2_1_DEFAULT_CONFIG,
    PROTOCOL_A,
    PROTOCOL_B,
    aggregate_protocol_comparison,
    assign_lp_group_ids,
    audit_lp_groups,
    compute_fold_integrity_record,
    compute_fold_metrics,
    compute_group_bootstrap_ci,
    generate_protocol_splits,
    load_adtec_train_only,
    run_rq1_generalization_study,
)


def _make_grouped_synthetic_df(n_groups: int = 12, rows_per_group: int = 4) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        for r in range(rows_per_group):
            lbl = "consistent" if (g + r) % 2 == 0 else "inconsistent"
            rows.append({
                "label": lbl,
                "title": f"Ad Headline G{g} R{r} {'Good' if lbl == 'consistent' else 'Bad'}",
                "lp_text": f"Landing page text for group {g} describing offerings.",
            })
    return pd.DataFrame(rows)


def test_group_key_normalization_and_label_independence():
    df = pd.DataFrame({
        "label": ["consistent", "inconsistent", "consistent"],
        "title": ["T1", "T2", "T3"],
        "lp_text": ["LP text １２３", "LP   text 123 ", "Different LP"],
    })
    gids = assign_lp_group_ids(df)
    # Row 0 and Row 1 normalize to identical string despite different labels
    assert gids.iloc[0] == gids.iloc[1]
    assert gids.iloc[0] != gids.iloc[2]
    assert gids.iloc[0].startswith("lp_grp_")


def test_train_only_loader_never_requires_valid_or_test(tmp_path: Path):
    df = _make_grouped_synthetic_df(n_groups=6, rows_per_group=2)
    df.to_csv(tmp_path / "train.tsv", sep="\t", index=False, encoding="utf-8")
    assert not (tmp_path / "valid.tsv").exists()
    assert not (tmp_path / "test.tsv").exists()

    loaded_train, prov = load_adtec_train_only(tmp_path)
    assert len(loaded_train) == len(df)
    assert prov["valid_split_loaded"] is False
    assert prov["test_split_loaded"] is False


def test_group_disjoint_splitting_and_fold_integrity():
    df = _make_grouped_synthetic_df(n_groups=15, rows_per_group=4)
    gids = assign_lp_group_ids(df)
    summary, grp_df = audit_lp_groups(df, gids)
    assert summary["group_count"] == 15
    assert summary["sample_count"] == 60
    assert summary["mixed_label_group_count"] == 15

    splits_a = generate_protocol_splits(df, gids, PROTOCOL_A, n_splits=3, seed=42)
    splits_b = generate_protocol_splits(df, gids, PROTOCOL_B, n_splits=3, seed=42)

    # Protocol B must have strictly 0 shared LPs and 0% shared eval rows on every fold
    for f_idx, (tr, ev) in enumerate(splits_b):
        rec = compute_fold_integrity_record(df, gids, tr, ev, PROTOCOL_B, 42, f_idx, set())
        assert rec["shared_lp_count"] == 0
        assert rec["shared_eval_row_percentage"] == 0.0
        assert rec["train_size"] + rec["eval_size"] == len(df)

    # Protocol A will have non-zero shared LPs when groups span multiple rows
    shared_counts_a = [
        compute_fold_integrity_record(df, gids, tr, ev, PROTOCOL_A, 42, f_idx, set())["shared_lp_count"]
        for f_idx, (tr, ev) in enumerate(splits_a)
    ]
    assert sum(shared_counts_a) > 0


def test_undefined_single_class_metrics_are_explicit_none():
    y_true = ["consistent", "consistent", "consistent"]
    y_pred = ["consistent", "consistent", "inconsistent"]
    scores = [0.1, 0.2, 0.7]

    m = compute_fold_metrics(y_true, y_pred, scores, train_time_seconds=0.1, inference_time_seconds=0.01)
    assert m["both_classes_present"] is False
    assert m["roc_auc"] is None
    assert m["average_precision"] is None
    assert m["ap_minus_prevalence"] is None
    assert m["balanced_accuracy"] is None
    assert m["eval_inconsistent_prevalence"] == 0.0


def test_rq1_end_to_end_reproducibility_and_oof_alignment(tmp_path: Path):
    df = _make_grouped_synthetic_df(n_groups=15, rows_per_group=4)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "train.tsv", sep="\t", index=False, encoding="utf-8")

    cfg = {
        **M2_1_DEFAULT_CONFIG,
        "experiment_id": "test_rq1",
        "seeds": [42],
        "candidate_n_splits": [3],
        "selected_n_splits": 3,
        "group_bootstrap_resamples": 25,
        "frozen_m2_baseline_config": {
            **M2_1_DEFAULT_CONFIG["frozen_m2_baseline_config"],
            "vectorizer": {
                "analyzer": "char",
                "ngram_range": [2, 3],
                "max_features": 300,
                "min_df": 1,
                "sublinear_tf": True,
            },
        },
    }

    res1 = run_rq1_generalization_study(
        config=cfg,
        data_dir=data_dir,
        runs_base_dir=tmp_path / "runs1",
        reports_dir=tmp_path / "rep1",
    )
    res2 = run_rq1_generalization_study(
        config=cfg,
        data_dir=data_dir,
        runs_base_dir=tmp_path / "runs2",
        reports_dir=tmp_path / "rep2",
    )

    # Identical OOF predictions and protocol comparison across repeated runs
    pd.testing.assert_frame_equal(res1["oof_predictions_df"], res2["oof_predictions_df"])
    pd.testing.assert_frame_equal(res1["protocol_comp_df"], res2["protocol_comp_df"])

    # Every record receives exactly one held-out prediction per (protocol, seed, model_id)
    oof = res1["oof_predictions_df"]
    for (_, _, _), grp in oof.groupby(["protocol", "seed", "model_id"]):
        assert len(grp) == len(df)
        assert grp["row_id"].nunique() == len(df)

    # No raw text persisted in CSV artifacts
    assert "lp_text" not in oof.columns and "title" not in oof.columns
    assert "lp_text" not in res1["group_dist_df"].columns

    # Group bootstrap CI structure
    ci = compute_group_bootstrap_ci(
        oof[(oof["model_id"] == "B3_pair") & (oof["protocol"] == PROTOCOL_B)],
        n_resamples=20,
        seed=42,
    )
    assert ci["method"] == "lp_group_cluster_bootstrap_on_fixed_oof"
    assert ci["metrics"]["macro_f1"]["ci_low"] <= ci["metrics"]["macro_f1"]["ci_high"]
