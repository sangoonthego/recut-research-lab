"""Targeted synthetic unit tests for Milestone M2.2 (RQ2: Lexical vs Semantic Modeling).

Uses strictly synthetic Japanese/ASCII fixtures and a deterministic mock encoder
(zero network calls, zero Hugging Face downloads, zero access to official splits).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from recut_lab.generalization import PROTOCOL_B, assign_lp_group_ids, generate_protocol_splits
from recut_lab.lexical_interactions import (
    LEXICAL_INTERACTION_FEATURE_NAMES,
    OVERLAP_BIN_ORDER,
    LexicalInteractionTransformer,
    assign_overlap_bins,
    compute_pairwise_lexical_features,
    extract_lexical_interaction_frame,
    fit_overlap_bin_edges,
    has_numeric_or_quantity_token,
)
from recut_lab.lexical_semantic import (
    M2_2_DEFAULT_CONFIG,
    MODEL_ORDER_RQ2,
    _compute_subgroup_metrics_core,
    compute_paired_group_bootstrap_interaction,
    fit_clustered_interaction_regression,
    load_m2_1_protocol_b_folds,
    run_rq2_lexical_semantic_study,
)
from recut_lab.semantic import (
    build_semantic_pair_features,
    compute_cosine_similarity,
    extract_or_load_cached_embeddings,
    l2_normalize_embeddings,
)


class _DeterministicMockEncoder:
    """Tiny deterministic encoder for offline synthetic unit testing."""

    def __init__(self, dim: int = 8, max_seq_length: int = 32, revision: str = "mockrev0001"):
        self.model_id = "mock/tiny-e5"
        self.resolved_revision = revision
        self.license_name = "mit"
        self.device = "cpu"
        self.max_seq_length = max_seq_length
        self.embedding_dim = dim
        self.encode_call_count = 0

    def weight_sha256(self) -> str:
        return hashlib.sha256(f"{self.model_id}:{self.resolved_revision}:{self.embedding_dim}".encode()).hexdigest()

    def measure_token_lengths(self, prefixed_texts: Sequence[str]) -> Dict[str, Any]:
        lengths = [max(2, len(t) // 2) for t in prefixed_texts]
        arr = np.asarray(lengths, dtype=np.int64)
        trunc = int((arr > self.max_seq_length).sum())
        return {
            "count": int(arr.size),
            "max_seq_length": int(self.max_seq_length),
            "truncated_count": trunc,
            "truncated_pct": round(trunc / max(arr.size, 1) * 100.0, 2),
            "min_tokens": int(arr.min()) if arr.size else 0,
            "median_tokens": float(np.median(arr)) if arr.size else 0.0,
            "mean_tokens": float(arr.mean()) if arr.size else 0.0,
            "p95_tokens": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "p99_tokens": float(np.percentile(arr, 99)) if arr.size else 0.0,
            "max_tokens": int(arr.max()) if arr.size else 0,
            "per_item_token_lengths": lengths,
        }

    def encode(
        self,
        prefixed_texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.encode_call_count += 1
        vecs = np.zeros((len(prefixed_texts), self.embedding_dim), dtype=np.float32)
        for i, txt in enumerate(prefixed_texts):
            digest = hashlib.sha256(txt.encode("utf-8")).digest()
            raw = np.frombuffer(digest[: self.embedding_dim * 4], dtype=np.uint32).astype(np.float32)
            # Map deterministically into [-1, 1] with slight character-signal preservation
            v = (raw / np.float32(2**31)) - np.float32(1.0)
            if "一致" in txt or "promo" in txt:
                v[0] += 1.5
            if "不一致" in txt or "mismatch" in txt:
                v[1] -= 1.5
            vecs[i] = v
        if normalize_embeddings:
            return l2_normalize_embeddings(vecs)
        return vecs


def _build_synthetic_adtec_df(n_groups: int = 12, rows_per_group: int = 6) -> pd.DataFrame:
    """Create a synthetic grouped Japanese-style DataFrame with both classes across LP groups."""
    rows = []
    for g in range(n_groups):
        lp_base = f"グループ{g:02d}の公式ランディングページ説明文です。美容液キャンペーン実施中。" + ("詳細情報" * (g % 4 + 2))
        for r in range(rows_per_group):
            is_inc = (g + r) % 2 == 1
            if is_inc:
                title = f"別商品{g}_{r} 最大90%オフ 不一致広告" if r % 2 == 0 else f"グループ{g:02d}の公式 限定500円 不一致"
                label = "inconsistent"
            else:
                title = f"グループ{g:02d}の公式 美容液キャンペーン 一致{r}"
                label = "consistent"
            rows.append({"label": label, "lp_text": lp_base, "title": title})
    return pd.DataFrame(rows)


def test_lexical_overlap_and_containment_direction():
    """Verify exact lexical overlap math and strict title->LP vs LP->title containment direction."""
    title = "美容液"  # 3 chars -> bigrams: {"美容", "容液"} (2)
    lp = "美容液と化粧水の公式サイト"  # 13 chars -> 12 unique bigrams, contains both title bigrams
    feats = compute_pairwise_lexical_features(title, lp)

    assert feats["title_to_lp_bigram_containment"] == pytest.approx(1.0, abs=1e-6)
    assert feats["lp_to_title_bigram_containment"] == pytest.approx(2.0 / 12.0, abs=1e-6)
    assert feats["title_to_lp_bigram_containment"] > feats["lp_to_title_bigram_containment"]
    assert feats["char_bigram_jaccard"] == pytest.approx(2.0 / 12.0, abs=1e-6)
    assert feats["char_length_ratio"] == pytest.approx(3.0 / 13.0, abs=1e-6)
    assert feats["shared_bigram_count"] == 2.0

    # Zero overlap case
    feats_zero = compute_pairwise_lexical_features("ABC", "あいうえお")
    assert feats_zero["char_bigram_jaccard"] == 0.0
    assert feats_zero["title_to_lp_bigram_containment"] == 0.0
    assert feats_zero["lp_to_title_bigram_containment"] == 0.0

    # Numeric pattern helper (NFKC full-width digits normalized to ASCII)
    assert has_numeric_or_quantity_token("今なら５０％オフ") is True
    assert has_numeric_or_quantity_token("公式キャンペーン実施中") is False


def test_lexical_interaction_transformer_shape_and_consistency():
    """Verify LexicalInteractionTransformer produces finite float32 matrices matching frame extraction."""
    df = _build_synthetic_adtec_df(n_groups=4, rows_per_group=3)
    frame = extract_lexical_interaction_frame(df)
    assert list(frame.columns) == list(LEXICAL_INTERACTION_FEATURE_NAMES)
    assert frame.shape == (len(df), len(LEXICAL_INTERACTION_FEATURE_NAMES))

    tr = LexicalInteractionTransformer()
    mat = tr.fit_transform(df)
    assert mat.dtype == np.float32
    assert mat.shape == (len(df), len(LEXICAL_INTERACTION_FEATURE_NAMES))
    assert np.allclose(mat, frame.to_numpy(dtype=np.float32), atol=1e-6)


def test_semantic_normalization_and_pair_feature_construction():
    """Verify L2 normalization and exact [u, v, |u-v|, u*v, cos(u,v)] pair feature formula."""
    u_raw = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    v_raw = np.array([[0.0, 4.0, 3.0], [0.0, 0.0, -5.0]], dtype=np.float32)

    u = l2_normalize_embeddings(u_raw)
    v = l2_normalize_embeddings(v_raw)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)

    cos = compute_cosine_similarity(u, v)
    assert cos[0] == pytest.approx(16.0 / 25.0, abs=1e-5)
    assert cos[1] == pytest.approx(-1.0, abs=1e-5)

    pair = build_semantic_pair_features(u, v)
    d = u.shape[1]
    assert pair.shape == (2, 4 * d + 1)
    assert pair.dtype == np.float32
    assert np.allclose(pair[:, :d], u)
    assert np.allclose(pair[:, d : 2 * d], v)
    assert np.allclose(pair[:, 2 * d : 3 * d], np.abs(u - v))
    assert np.allclose(pair[:, 3 * d : 4 * d], u * v)
    assert np.allclose(pair[:, -1], cos)


def test_embedding_cache_alignment_and_reuse(tmp_path: Path):
    """Verify content-hashed cache alignment, revision awareness, and zero re-encoding on reload."""
    df = _build_synthetic_adtec_df(n_groups=5, rows_per_group=4)
    enc1 = _DeterministicMockEncoder(dim=8, revision="rev_alpha")

    bundle1 = extract_or_load_cached_embeddings(
        df=df,
        cache_base_dir=tmp_path / "cache",
        encoder=enc1,
    )
    assert bundle1.manifest["loaded_from_cache"] is False
    assert bundle1.manifest["weights_verified_unchanged"] is True
    assert enc1.encode_call_count == 2  # once for unique LPs, once for unique titles

    # Reload from cache: encoder.encode should NOT be called again
    enc2 = _DeterministicMockEncoder(dim=8, revision="rev_alpha")
    bundle2 = extract_or_load_cached_embeddings(
        df=df,
        cache_base_dir=tmp_path / "cache",
        encoder=enc2,
    )
    assert bundle2.manifest["loaded_from_cache"] is True
    assert enc2.encode_call_count == 0

    # Verify alignment invariant under arbitrary row shuffling
    shuffled = df.sample(frac=1.0, random_state=7)
    u_orig, v_orig = bundle1.align_for_dataframe(df)
    u_shuf, v_shuf = bundle2.align_for_dataframe(shuffled)
    for shuf_pos, orig_idx in enumerate(shuffled.index.tolist()):
        assert np.allclose(u_shuf[shuf_pos], u_orig[orig_idx], atol=1e-6)
        assert np.allclose(v_shuf[shuf_pos], v_orig[orig_idx], atol=1e-6)

    # Different revision creates separate cache entry
    enc3 = _DeterministicMockEncoder(dim=8, revision="rev_beta")
    bundle3 = extract_or_load_cached_embeddings(
        df=df,
        cache_base_dir=tmp_path / "cache",
        encoder=enc3,
    )
    assert bundle3.cache_dir != bundle1.cache_dir
    assert bundle3.manifest["loaded_from_cache"] is False


def test_overlap_subgroup_thresholds_and_undefined_single_class_metrics():
    """Verify training-fold quantile binning (standard, zero-heavy, one-heavy) and explicit None metrics."""
    # Standard tertiles
    vals = np.linspace(0.0, 1.0, 90)
    spec = fit_overlap_bin_edges(vals, metric_name="char_bigram_jaccard")
    assert spec["rule"] == "standard_tertiles"
    bins = assign_overlap_bins(vals, spec)
    assert set(bins.tolist()) == set(OVERLAP_BIN_ORDER)
    assert len(bins) == len(vals)

    # Zero-heavy distribution (75% zeros)
    zero_heavy = np.array([0.0] * 75 + list(np.linspace(0.1, 0.9, 25)))
    spec_z = fit_overlap_bin_edges(zero_heavy, metric_name="char_bigram_jaccard")
    assert spec_z["rule"] == "lower_tie_split_positive_median"
    bins_z = assign_overlap_bins(zero_heavy, spec_z)
    assert np.sum(bins_z == "Low") == 75
    assert np.sum(bins_z == "Medium") + np.sum(bins_z == "High") == 25

    # One-heavy distribution (75% ones)
    one_heavy = np.array(list(np.linspace(0.0, 0.8, 25)) + [1.0] * 75)
    spec_o = fit_overlap_bin_edges(one_heavy, metric_name="title_to_lp_bigram_containment")
    assert spec_o["rule"] == "upper_tie_split_below_max_median"
    bins_o = assign_overlap_bins(one_heavy, spec_o)
    assert np.sum(bins_o == "High") == 75
    assert np.sum(bins_o == "Low") + np.sum(bins_o == "Medium") == 25

    # Single-class subgroup must return explicit None (never fabricated 0.0)
    single_cls = _compute_subgroup_metrics_core(
        y_true=["consistent", "consistent", "consistent"],
        y_pred=["consistent", "consistent", "inconsistent"],
        scores_inconsistent=[0.1, 0.2, 0.8],
        group_ids=["g1", "g1", "g2"],
        title_lengths=[10.0, 12.0, 11.0],
        lp_lengths=[100.0, 100.0, 120.0],
    )
    assert single_cls["both_classes_present"] is False
    assert single_cls["roc_auc"] is None
    assert single_cls["average_precision"] is None
    assert single_cls["macro_f1"] is None
    assert single_cls["balanced_accuracy"] is None
    assert single_cls["recall_inconsistent"] is None


def test_group_fold_reuse_and_manifest_verification(tmp_path: Path):
    """Verify M2.1 Protocol B fold manifest loading enforces strict LP disjointness and group alignment."""
    df = _build_synthetic_adtec_df(n_groups=10, rows_per_group=4)
    gids = assign_lp_group_ids(df)
    manifest_rows = []
    for seed in (42, 123):
        splits = generate_protocol_splits(df, gids, protocol=PROTOCOL_B, n_splits=3, seed=seed)
        for f_idx, (_, ev_idx) in enumerate(splits):
            for pos in ev_idx:
                manifest_rows.append({
                    "row_id": int(df.index[pos]),
                    "group_id": str(gids.iloc[pos]),
                    "protocol": PROTOCOL_B,
                    "seed": seed,
                    "fold": f_idx,
                })
    manifest_path = tmp_path / "fold_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    splits_by_seed, ref = load_m2_1_protocol_b_folds(
        manifest_path=manifest_path,
        train_df=df,
        group_ids=gids,
        expected_seeds=(42, 123),
        expected_n_splits=3,
    )
    assert ref["all_folds_strict_disjoint"] is True
    assert ref["all_group_ids_verified_exact_match"] is True
    assert len(splits_by_seed[42]) == 3
    for tr_idx, ev_idx in splits_by_seed[42]:
        assert set(gids.iloc[tr_idx]).isdisjoint(set(gids.iloc[ev_idx]))


def test_interaction_bootstrap_and_clustered_regression():
    """Verify paired LP-group bootstrap deltas and CR1 clustered interaction regression."""
    rng = np.random.default_rng(42)
    rows = []
    n_groups = 18
    for g in range(n_groups):
        gid = f"lp_grp_{g:03d}"
        bin_label = OVERLAP_BIN_ORDER[g % 3]
        for k in range(4):
            rid = g * 4 + k
            y_true = "inconsistent" if k % 2 == 0 else "consistent"
            # Construct clear synthetic interaction: Model A much better on Low, equal on High
            if bin_label == "Low":
                p_a = y_true
                s_a = 0.85 if y_true == "inconsistent" else 0.15
                p_b = "consistent"
                s_b = 0.45
            else:
                p_a = y_true
                s_a = 0.85 if y_true == "inconsistent" else 0.15
                p_b = y_true
                s_b = 0.85 if y_true == "inconsistent" else 0.15

            for m_id, pred, score in [("M_A", p_a, s_a), ("M_B", p_b, s_b)]:
                rows.append({
                    "row_id": rid,
                    "group_id": gid,
                    "model_id": m_id,
                    "true_label": y_true,
                    "predicted_label": pred,
                    "score_inconsistent": float(score + rng.normal(0, 0.01)),
                    "bin_ov": bin_label,
                })
    oof = pd.DataFrame(rows)

    boot = compute_paired_group_bootstrap_interaction(
        oof_primary_df=oof,
        model_a="M_A",
        model_b="M_B",
        overlap_col="bin_ov",
        n_resamples=40,
        seed=42,
    )
    assert boot["Low"]["delta_macro_f1"] > 0.3
    assert boot["High"]["delta_macro_f1"] == pytest.approx(0.0, abs=1e-4)
    assert boot["Interaction_Low_minus_High"]["delta_macro_f1"] > 0.3

    reg = fit_clustered_interaction_regression(
        oof_primary_df=oof,
        model_a="M_A",
        model_b="M_B",
        overlap_col="bin_ov",
    )
    assert reg["terms"]["low_minus_high_interaction"]["estimate"] > 0.3
    assert reg["n_lp_groups"] == n_groups


def test_end_to_end_rq2_determinism_on_synthetic_data(tmp_path: Path):
    """Verify deterministic end-to-end execution of L0, L1, S1, S2 and all isolated artifacts."""
    df = _build_synthetic_adtec_df(n_groups=12, rows_per_group=6)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    df.to_csv(raw_dir / "train.tsv", sep="\t", index=False)

    gids = assign_lp_group_ids(df)
    manifest_rows = []
    for seed in (42, 123):
        splits = generate_protocol_splits(df, gids, protocol=PROTOCOL_B, n_splits=3, seed=seed)
        for f_idx, (_, ev_idx) in enumerate(splits):
            for pos in ev_idx:
                manifest_rows.append({
                    "row_id": int(df.index[pos]),
                    "group_id": str(gids.iloc[pos]),
                    "protocol": PROTOCOL_B,
                    "seed": seed,
                    "fold": f_idx,
                })
    manifest_path = tmp_path / "m2_1_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    cfg = dict(M2_2_DEFAULT_CONFIG)
    cfg = {
        **M2_2_DEFAULT_CONFIG,
        "experiment_id": "test_rq2_run_1",
        "seeds": [42, 123],
        "n_splits": 3,
        "group_bootstrap_resamples": 25,
        "data": {
            **M2_2_DEFAULT_CONFIG["data"],
            "raw_dir": str(raw_dir),
            "m2_1_fold_manifest_path": str(manifest_path),
            "m2_1_protocol_comparison_path": str(tmp_path / "nonexistent.csv"),
        },
        "semantic": {
            **M2_2_DEFAULT_CONFIG["semantic"],
            "model_id": "mock/tiny-e5",
            "revision": "mockrev0001",
            "cache_base_dir": str(tmp_path / "emb_cache"),
        },
    }

    enc = _DeterministicMockEncoder(dim=8, revision="mockrev0001")
    res1 = run_rq2_lexical_semantic_study(
        config=cfg,
        runs_base_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports_1",
        encoder=enc,
    )

    cfg2 = {**cfg, "experiment_id": "test_rq2_run_2"}
    res2 = run_rq2_lexical_semantic_study(
        config=cfg2,
        runs_base_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports_2",
        encoder=enc,
    )

    # Exact metric determinism across runs
    assert list(res1["benchmark_df"]["model_id"]) == list(MODEL_ORDER_RQ2)
    for col in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent"):
        assert np.allclose(
            res1["benchmark_df"][col].to_numpy(),
            res2["benchmark_df"][col].to_numpy(),
            atol=1e-6,
        )

    # Verify all 13 required artifacts exist in isolated run directory
    required_files = [
        "config.json",
        "environment.json",
        "model_manifest.json",
        "embedding_manifest.json",
        "fold_manifest_reference.json",
        "per_fold_metrics.csv",
        "benchmark_comparison.csv",
        "lexical_overlap_analysis.csv",
        "interaction_analysis.csv",
        "error_disagreement.csv",
        "oof_predictions.csv",
        "model_costs.csv",
        "rq2_metrics.json",
    ]
    for fname in required_files:
        assert (res1["run_dir"] / fname).is_file(), f"Missing artifact: {fname}"
    assert (res1["reports_dir"] / "rq2_lexical_semantic_report.md").is_file()
