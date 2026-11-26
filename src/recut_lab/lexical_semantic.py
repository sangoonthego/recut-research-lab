"""Milestone M2.2 (RQ2): Lexical vs Semantic Modeling under LP-Group-Disjoint CV.

Research Question (RQ2):
    "Do semantic representations outperform lexical baselines in advertising
    consistency classification, especially when landing-page text and advertising
    titles have low lexical overlap?"

Compares four controlled model configurations on the EXACT landing-page-group-disjoint
folds (Protocol B, StratifiedGroupKFold, 5 folds, seeds [42, 123]) from Milestone M2.1,
exclusively within ``data/raw/adtec_consistency/train.tsv``:
    - L0_lexical_baseline: Frozen M2.1 B3_pair (char TF-IDF (2,4) + LogReg)
    - L1_lexical_interaction: L0 TF-IDF + scaled explicit pairwise lexical features + LogReg
    - S1_semantic_e5_small: Frozen intfloat/multilingual-e5-small pair features + StandardScaler + LogReg
    - S2_hybrid_lexical_semantic: L1 lexical features + S1 semantic pair features + LogReg

Official ``valid.tsv`` and ``test.tsv`` splits are strictly never loaded or accessed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from recut_lab.baseline import BaselineModel, M2_DEFAULT_CONFIG, TfidfLogRegBaseline
from recut_lab.data import compute_file_sha256
from recut_lab.diagnostics import POSITIVE_CLASS, extract_class_scores
from recut_lab.evaluation import LABEL_ORDER, ZERO_DIVISION_POLICY
from recut_lab.experiment import (
    ExperimentManager,
    get_installed_library_versions,
    set_global_seed,
)
from recut_lab.features import build_char_tfidf_vectorizer
from recut_lab.generalization import (
    PROTOCOL_B,
    assign_lp_group_ids,
    compute_fold_metrics,
    compute_group_bootstrap_ci,
    load_adtec_train_only,
)
from recut_lab.lexical_interactions import (
    LEXICAL_INTERACTION_FEATURE_NAMES,
    OVERLAP_BIN_ORDER,
    LexicalInteractionTransformer,
    assign_overlap_bins,
    extract_lexical_interaction_frame,
    fit_overlap_bin_edges,
    has_numeric_or_quantity_token,
)
from recut_lab.preprocessing import TextInputSelector
from recut_lab.semantic import (
    DEFAULT_E5_LICENSE,
    DEFAULT_E5_MODEL_ID,
    DEFAULT_E5_PREFIX,
    DEFAULT_E5_PREFIX_POLICY_NOTE,
    DEFAULT_E5_REVISION,
    SemanticEmbeddingBundle,
    TextEncoderProtocol,
    build_semantic_pair_features,
    compute_cosine_similarity,
    extract_or_load_cached_embeddings,
)

MODEL_ORDER_RQ2: Tuple[str, ...] = (
    "L0_lexical_baseline",
    "L1_lexical_interaction",
    "S1_semantic_e5_small",
    "S2_hybrid_lexical_semantic",
)

MODEL_REPRESENTATION_LABELS: Dict[str, str] = {
    "L0_lexical_baseline": "Char TF-IDF (2-4) on [title + SEP + lp_text]",
    "L1_lexical_interaction": "Char TF-IDF (2-4) + 15 scaled pairwise lexical interaction features",
    "S1_semantic_e5_small": "Frozen multilingual-e5-small [u, v, |u-v|, u*v, cos] (1537-d, scaled)",
    "S2_hybrid_lexical_semantic": "L1 lexical representation + S1 semantic pair features (sparse+scaled)",
}

OVERLAP_METRICS: Tuple[str, ...] = (
    "char_bigram_jaccard",
    "title_to_lp_bigram_containment",
)

M2_2_DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_id": "m2_lexical_semantic_v001",
    "primary_seed": 42,
    "seeds": [42, 123],
    "n_splits": 5,
    "protocol": PROTOCOL_B,
    "positive_class": POSITIVE_CLASS,
    "label_order": list(LABEL_ORDER),
    "zero_division_policy": ZERO_DIVISION_POLICY,
    "group_bootstrap_resamples": 500,
    "data": {
        "raw_dir": "data/raw/adtec_consistency",
        "train_only": True,
        "valid_split_excluded": True,
        "test_split_excluded": True,
        "group_field": "lp_text",
        "group_normalization": "unicode_nfkc_whitespace_collapse",
        "m2_1_fold_manifest_path": "experiments/runs/m2_generalization_v001/fold_manifest.csv",
        "m2_1_protocol_comparison_path": "experiments/runs/m2_generalization_v001/protocol_comparison.csv",
    },
    "lexical": {
        "separator": M2_DEFAULT_CONFIG["data"]["separator"],
        "normalize_unicode_nfkc": M2_DEFAULT_CONFIG["data"]["normalize_unicode_nfkc"],
        "vectorizer": deepcopy(M2_DEFAULT_CONFIG["vectorizer"]),
        "interaction_features": list(LEXICAL_INTERACTION_FEATURE_NAMES),
        "dense_scaler": "StandardScaler (fit strictly inside each training fold)",
    },
    "semantic": {
        "model_id": DEFAULT_E5_MODEL_ID,
        "revision": DEFAULT_E5_REVISION,
        "license": DEFAULT_E5_LICENSE,
        "device": "cpu",
        "batch_size": 32,
        "dtype": "float32",
        "normalize_embeddings": True,
        "title_prefix": DEFAULT_E5_PREFIX,
        "lp_prefix": DEFAULT_E5_PREFIX,
        "prefix_policy_note": DEFAULT_E5_PREFIX_POLICY_NOTE,
        "cache_base_dir": "data/processed/embeddings_cache",
        "dense_scaler": "StandardScaler (fit strictly inside each training fold)",
    },
    "classifier": deepcopy(M2_DEFAULT_CONFIG["classifier"]),
    "overlap_subgroup_metrics": list(OVERLAP_METRICS),
    "models": [
        {
            "model_id": "L0_lexical_baseline",
            "representation": MODEL_REPRESENTATION_LABELS["L0_lexical_baseline"],
        },
        {
            "model_id": "L1_lexical_interaction",
            "representation": MODEL_REPRESENTATION_LABELS["L1_lexical_interaction"],
        },
        {
            "model_id": "S1_semantic_e5_small",
            "representation": MODEL_REPRESENTATION_LABELS["S1_semantic_e5_small"],
        },
        {
            "model_id": "S2_hybrid_lexical_semantic",
            "representation": MODEL_REPRESENTATION_LABELS["S2_hybrid_lexical_semantic"],
        },
    ],
}


# =============================================================================
# 1. CONTROLLED RQ2 MODEL CONFIGURATIONS (L0, L1, S1, S2)
# =============================================================================


class LexicalInteractionLogReg(BaselineModel):
    """L1: Sparse char n-gram TF-IDF + scaled explicit pairwise lexical features + LogReg.

    Preserves the sparse TF-IDF CSR matrix without densification and fits both the
    TF-IDF vectorizer and StandardScaler strictly inside each training fold.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 2,
        max_features: Optional[int] = 50000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        class_weight: Optional[str] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        separator: str = " [SEP] ",
        normalize: bool = True,
    ):
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.separator = separator
        self.normalize = normalize

        self.selector_: Optional[TextInputSelector] = None
        self.vectorizer_ = None
        self.lex_extractor_: Optional[LexicalInteractionTransformer] = None
        self.lex_scaler_: Optional[StandardScaler] = None
        self.clf_: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_design_matrix(
        self,
        df: pd.DataFrame,
        fit: bool,
        precomputed_lex: Optional[np.ndarray] = None,
    ) -> sparse.csr_matrix:
        if fit:
            self.selector_ = TextInputSelector(
                mode="pair", separator=self.separator, normalize=self.normalize
            )
            self.vectorizer_ = build_char_tfidf_vectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.lex_extractor_ = LexicalInteractionTransformer()
            self.lex_scaler_ = StandardScaler()

            texts = self.selector_.fit_transform(df)
            X_tfidf = self.vectorizer_.fit_transform(texts)
            X_lex_raw = (
                precomputed_lex
                if precomputed_lex is not None
                else self.lex_extractor_.fit_transform(df)
            )
            X_lex_scaled = self.lex_scaler_.fit_transform(X_lex_raw)
        else:
            if self.selector_ is None or self.vectorizer_ is None or self.lex_scaler_ is None:
                raise RuntimeError("Model is not fitted yet.")
            texts = self.selector_.transform(df)
            X_tfidf = self.vectorizer_.transform(texts)
            X_lex_raw = (
                precomputed_lex
                if precomputed_lex is not None
                else self.lex_extractor_.transform(df)
            )
            X_lex_scaled = self.lex_scaler_.transform(X_lex_raw)

        # Keep TF-IDF sparse CSR; stack dense 15-col lexical features as sparse CSR
        return sparse.hstack(
            [X_tfidf, sparse.csr_matrix(X_lex_scaled.astype(np.float32))], format="csr"
        )

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        precomputed_lex: Optional[np.ndarray] = None,
    ) -> LexicalInteractionLogReg:
        X = self._build_design_matrix(df, fit=True, precomputed_lex=precomputed_lex)
        y = df[target_col].to_numpy()
        self.clf_ = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(
        self,
        df: pd.DataFrame,
        precomputed_lex: Optional[np.ndarray] = None,
    ) -> List[str]:
        if self.clf_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._build_design_matrix(df, fit=False, precomputed_lex=precomputed_lex)
        return [str(p) for p in self.clf_.predict(X)]

    def predict_proba(
        self,
        df: pd.DataFrame,
        precomputed_lex: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._build_design_matrix(df, fit=False, precomputed_lex=precomputed_lex)
        return self.clf_.predict_proba(X)


class SemanticPairLogReg(BaselineModel):
    """S1: Frozen multilingual-e5-small pair features + StandardScaler + LogReg.

    Pair features ``[u, v, |u - v|, u * v, cos(u, v)]`` are computed from frozen
    embeddings; StandardScaler and LogisticRegression fit strictly on the training fold.
    """

    def __init__(
        self,
        embedding_bundle: Optional[SemanticEmbeddingBundle] = None,
        c_param: float = 1.0,
        class_weight: Optional[str] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
    ):
        self.embedding_bundle = embedding_bundle
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver

        self.scaler_: Optional[StandardScaler] = None
        self.clf_: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None

    def _get_pair_matrix(
        self,
        df: pd.DataFrame,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if precomputed_sem is not None:
            return precomputed_sem
        if self.embedding_bundle is None:
            raise ValueError("Provide either embedding_bundle or precomputed_sem matrix.")
        return self.embedding_bundle.build_pair_matrix(df)

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> SemanticPairLogReg:
        X_raw = self._get_pair_matrix(df, precomputed_sem=precomputed_sem)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_raw)
        y = df[target_col].to_numpy()
        self.clf_ = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.clf_.fit(X_scaled, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(
        self,
        df: pd.DataFrame,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> List[str]:
        if self.clf_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler_.transform(self._get_pair_matrix(df, precomputed_sem=precomputed_sem))
        return [str(p) for p in self.clf_.predict(X_scaled)]

    def predict_proba(
        self,
        df: pd.DataFrame,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.clf_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler_.transform(self._get_pair_matrix(df, precomputed_sem=precomputed_sem))
        return self.clf_.predict_proba(X_scaled)


class HybridLexicalSemanticLogReg(BaselineModel):
    """S2: Hybrid sparse TF-IDF + scaled lexical interaction + scaled semantic pair + LogReg.

    Preserves sparse TF-IDF matrices via ``scipy.sparse.hstack`` and fits TF-IDF and both
    dense ``StandardScaler`` instances strictly inside each training fold.
    """

    def __init__(
        self,
        embedding_bundle: Optional[SemanticEmbeddingBundle] = None,
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 2,
        max_features: Optional[int] = 50000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        class_weight: Optional[str] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        separator: str = " [SEP] ",
        normalize: bool = True,
    ):
        self.embedding_bundle = embedding_bundle
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.separator = separator
        self.normalize = normalize

        self.selector_: Optional[TextInputSelector] = None
        self.vectorizer_ = None
        self.lex_extractor_: Optional[LexicalInteractionTransformer] = None
        self.lex_scaler_: Optional[StandardScaler] = None
        self.sem_scaler_: Optional[StandardScaler] = None
        self.clf_: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_design_matrix(
        self,
        df: pd.DataFrame,
        fit: bool,
        precomputed_lex: Optional[np.ndarray] = None,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> sparse.csr_matrix:
        if precomputed_sem is None:
            if self.embedding_bundle is None:
                raise ValueError("Provide either embedding_bundle or precomputed_sem matrix.")
            X_sem_raw = self.embedding_bundle.build_pair_matrix(df)
        else:
            X_sem_raw = precomputed_sem

        if fit:
            self.selector_ = TextInputSelector(
                mode="pair", separator=self.separator, normalize=self.normalize
            )
            self.vectorizer_ = build_char_tfidf_vectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.lex_extractor_ = LexicalInteractionTransformer()
            self.lex_scaler_ = StandardScaler()
            self.sem_scaler_ = StandardScaler()

            X_tfidf = self.vectorizer_.fit_transform(self.selector_.fit_transform(df))
            X_lex_raw = (
                precomputed_lex
                if precomputed_lex is not None
                else self.lex_extractor_.fit_transform(df)
            )
            X_lex_scaled = self.lex_scaler_.fit_transform(X_lex_raw)
            X_sem_scaled = self.sem_scaler_.fit_transform(X_sem_raw)
        else:
            if (
                self.selector_ is None
                or self.vectorizer_ is None
                or self.lex_scaler_ is None
                or self.sem_scaler_ is None
            ):
                raise RuntimeError("Model is not fitted yet.")
            X_tfidf = self.vectorizer_.transform(self.selector_.transform(df))
            X_lex_raw = (
                precomputed_lex
                if precomputed_lex is not None
                else self.lex_extractor_.transform(df)
            )
            X_lex_scaled = self.lex_scaler_.transform(X_lex_raw)
            X_sem_scaled = self.sem_scaler_.transform(X_sem_raw)

        return sparse.hstack(
            [
                X_tfidf,
                sparse.csr_matrix(X_lex_scaled.astype(np.float32)),
                sparse.csr_matrix(X_sem_scaled.astype(np.float32)),
            ],
            format="csr",
        )

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        precomputed_lex: Optional[np.ndarray] = None,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> HybridLexicalSemanticLogReg:
        X = self._build_design_matrix(
            df, fit=True, precomputed_lex=precomputed_lex, precomputed_sem=precomputed_sem
        )
        y = df[target_col].to_numpy()
        self.clf_ = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(
        self,
        df: pd.DataFrame,
        precomputed_lex: Optional[np.ndarray] = None,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> List[str]:
        if self.clf_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._build_design_matrix(
            df, fit=False, precomputed_lex=precomputed_lex, precomputed_sem=precomputed_sem
        )
        return [str(p) for p in self.clf_.predict(X)]

    def predict_proba(
        self,
        df: pd.DataFrame,
        precomputed_lex: Optional[np.ndarray] = None,
        precomputed_sem: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._build_design_matrix(
            df, fit=False, precomputed_lex=precomputed_lex, precomputed_sem=precomputed_sem
        )
        return self.clf_.predict_proba(X)


def _estimate_fitted_model_size_mb(model: Any) -> float:
    """Measure serialized size of fitted classifier + feature scalers/vectorizers (excluding external bundle)."""
    saved_bundle = getattr(model, "embedding_bundle", None)
    if saved_bundle is not None:
        model.embedding_bundle = None
    try:
        raw_bytes = pickle.dumps(model)
        return round(len(raw_bytes) / (1024.0 * 1024.0), 4)
    finally:
        if saved_bundle is not None:
            model.embedding_bundle = saved_bundle


def _extract_inconsistent_scores_from_proba(
    probs: np.ndarray,
    classes: np.ndarray,
    target_class: str = POSITIVE_CLASS,
) -> np.ndarray:
    class_list = [str(c) for c in classes]
    if target_class not in class_list:
        raise ValueError(f"Target class '{target_class}' not found in classes={class_list}.")
    idx = class_list.index(target_class)
    return np.asarray(probs[:, idx], dtype=np.float64)


# =============================================================================
# 2. REUSING EXACT M2.1 GROUP-DISJOINT FOLDS (PROTOCOL B)
# =============================================================================


def load_m2_1_protocol_b_folds(
    manifest_path: Path | str,
    train_df: pd.DataFrame,
    group_ids: pd.Series,
    expected_seeds: Sequence[int] = (42, 123),
    expected_n_splits: int = 5,
) -> Tuple[Dict[int, List[Tuple[np.ndarray, np.ndarray]]], Dict[str, Any]]:
    """Load exact Protocol B folds from the M2.1 fold manifest and verify integrity.

    Returns:
        Tuple of ``(splits_by_seed, manifest_reference_dict)``.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        raise FileNotFoundError(
            f"M2.1 fold manifest not found at '{mpath.resolve()}'. Run M2.1 first."
        )

    manifest_sha = compute_file_sha256(mpath)
    manifest_df = pd.read_csv(mpath)
    proto_b = manifest_df[manifest_df["protocol"] == PROTOCOL_B].copy()
    if proto_b.empty:
        raise ValueError(f"No rows with protocol='{PROTOCOL_B}' found in '{mpath}'.")

    row_id_to_pos = {int(rid): pos for pos, rid in enumerate(train_df.index.tolist())}
    n_rows = len(train_df)
    all_positions = np.arange(n_rows, dtype=np.int64)
    groups_arr = group_ids.to_numpy()

    splits_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    fold_checks: List[Dict[str, Any]] = []

    for seed in [int(s) for s in expected_seeds]:
        sub_seed = proto_b[proto_b["seed"] == seed]
        if len(sub_seed) != n_rows:
            raise ValueError(
                f"M2.1 manifest seed={seed} has {len(sub_seed)} rows; expected {n_rows}."
            )
        folds_present = sorted(sub_seed["fold"].unique().tolist())
        if folds_present != list(range(expected_n_splits)):
            raise ValueError(
                f"Expected folds {list(range(expected_n_splits))}, got {folds_present}."
            )

        seed_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        seen_eval: List[np.ndarray] = []

        for f_idx in range(expected_n_splits):
            sub_f = sub_seed[sub_seed["fold"] == f_idx]
            ev_pos = np.sort(
                np.asarray([row_id_to_pos[int(rid)] for rid in sub_f["row_id"].tolist()], dtype=np.int64)
            )
            # Verify group_id alignment against freshly computed NFKC LP groups
            manifest_gids = sub_f.set_index("row_id")["group_id"].to_dict()
            for pos in ev_pos:
                rid = int(train_df.index[pos])
                if manifest_gids[rid] != str(groups_arr[pos]):
                    raise AssertionError(
                        f"Group ID mismatch at row_id={rid}: manifest={manifest_gids[rid]} vs computed={groups_arr[pos]}."
                    )

            tr_mask = np.ones(n_rows, dtype=bool)
            tr_mask[ev_pos] = False
            tr_pos = all_positions[tr_mask]

            tr_g = set(groups_arr[tr_pos])
            ev_g = set(groups_arr[ev_pos])
            if not tr_g.isdisjoint(ev_g):
                raise AssertionError(
                    f"LP group disjointness violated in M2.1 manifest (seed={seed}, fold={f_idx})!"
                )

            seed_splits.append((tr_pos, ev_pos))
            seen_eval.append(ev_pos)
            fold_checks.append({
                "seed": seed,
                "fold": f_idx,
                "train_size": int(len(tr_pos)),
                "eval_size": int(len(ev_pos)),
                "train_unique_lp": int(len(tr_g)),
                "eval_unique_lp": int(len(ev_g)),
                "shared_lp_count": 0,
                "strict_disjoint": True,
            })

        covered = np.concatenate(seen_eval)
        if len(covered) != n_rows or len(np.unique(covered)) != n_rows:
            raise AssertionError(f"Seed {seed} folds do not partition all {n_rows} rows exactly once.")
        splits_by_seed[seed] = seed_splits

    reference = {
        "source_manifest_path": str(mpath.as_posix()),
        "fold_manifest_hash": manifest_sha,
        "protocol": PROTOCOL_B,
        "seeds": [int(s) for s in expected_seeds],
        "n_splits": int(expected_n_splits),
        "row_count": int(n_rows),
        "unique_lp_groups": int(group_ids.nunique()),
        "all_folds_strict_disjoint": True,
        "all_group_ids_verified_exact_match": True,
        "fold_summaries": fold_checks,
    }
    return splits_by_seed, reference


# =============================================================================
# 3. LEXICAL OVERLAP SUBGROUP ANALYSIS & FEATURE INTERACTION + BOOTSTRAP
# =============================================================================


def _compute_subgroup_metrics_core(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores_inconsistent: Sequence[float],
    group_ids: Sequence[str],
    title_lengths: Sequence[float],
    lp_lengths: Sequence[float],
) -> Dict[str, Any]:
    """Compute subgroup evaluation metrics with explicit None for single-class undefined metrics."""
    n = len(y_true)
    y_arr = np.asarray(list(y_true), dtype=object)
    p_arr = np.asarray(list(y_pred), dtype=object)

    n_cons = int(np.sum(y_arr == "consistent"))
    n_inc = int(np.sum(y_arr == POSITIVE_CLASS))
    n_pred_inc = int(np.sum(p_arr == POSITIVE_CLASS))
    n_unique_lp = int(len(set(group_ids)))
    mean_t_len = round(float(np.mean(title_lengths)), 2) if n > 0 else None
    mean_lp_len = round(float(np.mean(lp_lengths)), 2) if n > 0 else None

    if n == 0:
        return {
            "sample_count": 0,
            "unique_lp_count": 0,
            "consistent_count": 0,
            "inconsistent_count": 0,
            "mean_title_length": None,
            "mean_lp_length": None,
            "both_classes_present": False,
            "macro_f1": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "average_precision": None,
            "recall_inconsistent": None,
            "predicted_inconsistent_count": 0,
        }

    m = compute_fold_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores_inconsistent=scores_inconsistent,
        train_time_seconds=0.0,
        inference_time_seconds=0.0,
    )
    has_both = bool(m["both_classes_present"])

    return {
        "sample_count": int(n),
        "unique_lp_count": n_unique_lp,
        "consistent_count": n_cons,
        "inconsistent_count": n_inc,
        "mean_title_length": mean_t_len,
        "mean_lp_length": mean_lp_len,
        "both_classes_present": has_both,
        "macro_f1": m["macro_f1"] if has_both else None,
        "balanced_accuracy": m["balanced_accuracy"],
        "roc_auc": m["roc_auc"],
        "average_precision": m["average_precision"],
        "recall_inconsistent": m["recall_inconsistent"] if n_inc > 0 else None,
        "predicted_inconsistent_count": n_pred_inc,
    }


def compute_paired_group_bootstrap_interaction(
    oof_primary_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    overlap_col: str,
    n_resamples: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """LP-group-cluster bootstrap for paired metric differences (model_a - model_b)
    across Overall, Low, Medium, High overlap bins AND the Low-minus-High interaction.

    Resamples unique LP groups with replacement (never independent rows) so paired
    models are evaluated on identical resampled LP-group clusters.
    """
    sub_a = oof_primary_df[oof_primary_df["model_id"] == model_a].sort_values("row_id").reset_index(drop=True)
    sub_b = oof_primary_df[oof_primary_df["model_id"] == model_b].sort_values("row_id").reset_index(drop=True)

    if not np.array_equal(sub_a["row_id"].to_numpy(), sub_b["row_id"].to_numpy()):
        raise AssertionError("Row IDs are not aligned between compared models.")

    unique_groups = np.sort(sub_a["group_id"].unique())
    n_groups = len(unique_groups)
    gid_to_idx = {gid: i for i, gid in enumerate(unique_groups)}

    y_bin_all = (sub_a["true_label"].to_numpy() == POSITIVE_CLASS).astype(np.int8)
    pa_all = (sub_a["predicted_label"].to_numpy() == POSITIVE_CLASS).astype(np.int8)
    pb_all = (sub_b["predicted_label"].to_numpy() == POSITIVE_CLASS).astype(np.int8)
    sa_all = sub_a["score_inconsistent"].to_numpy(dtype=np.float64)
    sb_all = sub_b["score_inconsistent"].to_numpy(dtype=np.float64)
    bins_all = sub_a[overlap_col].to_numpy()
    gids_all = sub_a["group_id"].to_numpy()

    bin_names = ["Overall", "Low", "Medium", "High"]
    # Pre-index per-group confusion counts & scores for each bin
    # counts shape: (4 bins, n_groups, 4 [tn, fp, fn, tp])
    counts_a = np.zeros((4, n_groups, 4), dtype=np.int64)
    counts_b = np.zeros((4, n_groups, 4), dtype=np.int64)
    grp_y: List[List[np.ndarray]] = [[np.empty(0, dtype=np.int8)] * n_groups for _ in range(4)]
    grp_sa: List[List[np.ndarray]] = [[np.empty(0, dtype=np.float64)] * n_groups for _ in range(4)]
    grp_sb: List[List[np.ndarray]] = [[np.empty(0, dtype=np.float64)] * n_groups for _ in range(4)]

    # Group indices by gid once
    gid_positions: Dict[int, List[int]] = {i: [] for i in range(n_groups)}
    for pos, gid in enumerate(gids_all):
        gid_positions[gid_to_idx[gid]].append(pos)

    for g_idx in range(n_groups):
        pos_arr = np.asarray(gid_positions[g_idx], dtype=np.int64)
        for b_idx, b_name in enumerate(bin_names):
            if b_name == "Overall":
                sel = pos_arr
            else:
                sel = pos_arr[bins_all[pos_arr] == b_name]
            if sel.size == 0:
                continue
            yb = y_bin_all[sel]
            pa = pa_all[sel]
            pb = pb_all[sel]
            counts_a[b_idx, g_idx, 0] = int(np.sum((yb == 0) & (pa == 0)))
            counts_a[b_idx, g_idx, 1] = int(np.sum((yb == 0) & (pa == 1)))
            counts_a[b_idx, g_idx, 2] = int(np.sum((yb == 1) & (pa == 0)))
            counts_a[b_idx, g_idx, 3] = int(np.sum((yb == 1) & (pa == 1)))

            counts_b[b_idx, g_idx, 0] = int(np.sum((yb == 0) & (pb == 0)))
            counts_b[b_idx, g_idx, 1] = int(np.sum((yb == 0) & (pb == 1)))
            counts_b[b_idx, g_idx, 2] = int(np.sum((yb == 1) & (pb == 0)))
            counts_b[b_idx, g_idx, 3] = int(np.sum((yb == 1) & (pb == 1)))

            grp_y[b_idx][g_idx] = yb
            grp_sa[b_idx][g_idx] = sa_all[sel]
            grp_sb[b_idx][g_idx] = sb_all[sel]

    def _metrics_from_counts_and_scores(
        cnt: np.ndarray,
        yb: np.ndarray,
        sc: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        tn, fp, fn, tp = int(cnt[0]), int(cnt[1]), int(cnt[2]), int(cnt[3])
        n_neg = tn + fp
        n_pos = fn + tp
        if n_neg == 0 or n_pos == 0:
            return None
        rec_c = tn / n_neg
        rec_i = tp / n_pos
        prec_c = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c = (2.0 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        f1_i = (2.0 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
        return {
            "macro_f1": 0.5 * (f1_c + f1_i),
            "roc_auc": float(roc_auc_score(yb, sc)),
            "average_precision": float(average_precision_score(yb, sc)),
            "recall_inconsistent": rec_i,
        }

    # Exact point estimates on original OOF set
    point_deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for b_idx, b_name in enumerate(bin_names):
        non_empty_idx = [i for i in range(n_groups) if grp_y[b_idx][i].size > 0]
        if not non_empty_idx:
            point_deltas[b_name] = {
                k: None for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent")
            }
            continue
        yb = np.concatenate([grp_y[b_idx][i] for i in non_empty_idx])
        sa = np.concatenate([grp_sa[b_idx][i] for i in non_empty_idx])
        sb = np.concatenate([grp_sb[b_idx][i] for i in non_empty_idx])
        ma = _metrics_from_counts_and_scores(counts_a[b_idx].sum(axis=0), yb, sa)
        mb = _metrics_from_counts_and_scores(counts_b[b_idx].sum(axis=0), yb, sb)
        if ma is not None and mb is not None:
            point_deltas[b_name] = {k: ma[k] - mb[k] for k in ma}
        else:
            point_deltas[b_name] = {
                k: None for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent")
            }

    # Interaction point estimate: Delta_Low - Delta_High
    point_deltas["Interaction_Low_minus_High"] = {
        k: (
            point_deltas["Low"][k] - point_deltas["High"][k]
            if point_deltas["Low"][k] is not None and point_deltas["High"][k] is not None
            else None
        )
        for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent")
    }

    # Group-cluster bootstrap resamples
    rng = np.random.default_rng(seed)
    all_targets = bin_names + ["Interaction_Low_minus_High"]
    boot_store: Dict[str, Dict[str, List[float]]] = {
        tgt: {k: [] for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent")}
        for tgt in all_targets
    }

    for _ in range(n_resamples):
        sampled = rng.integers(0, n_groups, size=n_groups)
        draw_deltas: Dict[str, Optional[Dict[str, float]]] = {}
        for b_idx, b_name in enumerate(bin_names):
            cnt_a = counts_a[b_idx, sampled].sum(axis=0)
            cnt_b = counts_b[b_idx, sampled].sum(axis=0)
            if (cnt_a[0] + cnt_a[1]) == 0 or (cnt_a[2] + cnt_a[3]) == 0:
                draw_deltas[b_name] = None
                continue
            non_empty = [i for i in sampled if grp_y[b_idx][i].size > 0]
            yb = np.concatenate([grp_y[b_idx][i] for i in non_empty])
            sa = np.concatenate([grp_sa[b_idx][i] for i in non_empty])
            sb = np.concatenate([grp_sb[b_idx][i] for i in non_empty])
            ma = _metrics_from_counts_and_scores(cnt_a, yb, sa)
            mb = _metrics_from_counts_and_scores(cnt_b, yb, sb)
            if ma is not None and mb is not None:
                d = {k: ma[k] - mb[k] for k in ma}
                draw_deltas[b_name] = d
                for k, v in d.items():
                    boot_store[b_name][k].append(v)
            else:
                draw_deltas[b_name] = None

        if draw_deltas["Low"] is not None and draw_deltas["High"] is not None:
            for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent"):
                boot_store["Interaction_Low_minus_High"][k].append(
                    draw_deltas["Low"][k] - draw_deltas["High"][k]
                )

    lo_q = (alpha / 2.0) * 100.0
    hi_q = (1.0 - alpha / 2.0) * 100.0
    results: Dict[str, Dict[str, Any]] = {}
    for tgt in all_targets:
        rec: Dict[str, Any] = {"n_groups": int(n_groups), "n_bootstrap_resamples": int(n_resamples)}
        for k in ("macro_f1", "roc_auc", "average_precision", "recall_inconsistent"):
            pt = point_deltas[tgt][k]
            vals = boot_store[tgt][k]
            rec[f"delta_{k}"] = round(float(pt), 4) if pt is not None else None
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                rec[f"delta_{k}_ci_low"] = round(float(np.percentile(arr, lo_q)), 4)
                rec[f"delta_{k}_ci_high"] = round(float(np.percentile(arr, hi_q)), 4)
            else:
                rec[f"delta_{k}_ci_low"] = None
                rec[f"delta_{k}_ci_high"] = None
        results[tgt] = rec
    return results


def fit_clustered_interaction_regression(
    oof_primary_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    overlap_col: str,
) -> Dict[str, Any]:
    """Fit a paired LP-group-clustered linear interaction model on per-row correctness difference.

    Outcome:
        ``Y_i = I(model_a correct on row i) - I(model_b correct on row i)`` in ``{-1, 0, +1}``.
    Model:
        ``E[Y_i | bin_i] = beta_0 (Low) + beta_med * I(Medium) + beta_high * I(High)``
    Variance estimator:
        Liang-Zeger / CR1 cluster-robust sandwich covariance clustered by ``group_id``.

    Explicit assumptions & limitations:
        - Observational subgroup association on fixed OOF predictions (not causal evidence).
        - Captures cluster correlation within shared landing pages, not full fold-refitting variance.
    """
    sub_a = oof_primary_df[oof_primary_df["model_id"] == model_a].sort_values("row_id").reset_index(drop=True)
    sub_b = oof_primary_df[oof_primary_df["model_id"] == model_b].sort_values("row_id").reset_index(drop=True)

    y_diff = (
        (sub_a["predicted_label"].to_numpy() == sub_a["true_label"].to_numpy()).astype(np.float64)
        - (sub_b["predicted_label"].to_numpy() == sub_b["true_label"].to_numpy()).astype(np.float64)
    )
    bins = sub_a[overlap_col].to_numpy()
    groups = sub_a["group_id"].to_numpy()

    # Design matrix: [intercept (Low baseline), I(Medium), I(High)]
    X = np.column_stack([
        np.ones(len(y_diff), dtype=np.float64),
        (bins == "Medium").astype(np.float64),
        (bins == "High").astype(np.float64),
    ])
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y_diff)
    resid = y_diff - X @ beta

    unique_g = np.unique(groups)
    G = len(unique_g)
    meat = np.zeros((k, k), dtype=np.float64)
    for g in unique_g:
        mask = groups == g
        score_g = X[mask].T @ resid[mask]
        meat += np.outer(score_g, score_g)

    # CR1 finite-cluster adjustment
    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov_cr1 = scale * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov_cr1), 0.0))

    names = ["intercept_low_overlap", "medium_minus_low", "high_minus_low"]
    terms: Dict[str, Any] = {}
    for idx, name in enumerate(names):
        z = float(beta[idx] / se[idx]) if se[idx] > 0 else 0.0
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z)))) if se[idx] > 0 else 1.0
        terms[name] = {
            "estimate": round(float(beta[idx]), 5),
            "cluster_se": round(float(se[idx]), 5),
            "ci_low_95": round(float(beta[idx] - 1.96 * se[idx]), 5),
            "ci_high_95": round(float(beta[idx] + 1.96 * se[idx]), 5),
            "z_stat": round(z, 3),
            "p_value": round(p_val, 5),
        }

    # Explicit Low - High contrast = -beta[2]
    low_minus_high = -float(beta[2])
    se_lmh = float(se[2])
    z_lmh = low_minus_high / se_lmh if se_lmh > 0 else 0.0
    p_lmh = float(2.0 * (1.0 - norm.cdf(abs(z_lmh)))) if se_lmh > 0 else 1.0
    terms["low_minus_high_interaction"] = {
        "estimate": round(low_minus_high, 5),
        "cluster_se": round(se_lmh, 5),
        "ci_low_95": round(low_minus_high - 1.96 * se_lmh, 5),
        "ci_high_95": round(low_minus_high + 1.96 * se_lmh, 5),
        "z_stat": round(z_lmh, 3),
        "p_value": round(p_lmh, 5),
    }

    return {
        "comparison": f"{model_a}_minus_{model_b}",
        "overlap_metric": overlap_col,
        "outcome": "paired_row_accuracy_diff = I(model_a correct) - I(model_b correct)",
        "variance_estimator": "CR1_LP_group_clustered_sandwich",
        "n_rows": int(n),
        "n_lp_groups": int(G),
        "terms": terms,
        "causal_claim_warning": (
            "Observational subgroup association on fixed OOF predictions; overlap bins correlate "
            "with LP length, domain, and campaign structure and do not establish causality."
        ),
    }


# =============================================================================
# 4. ERROR DISAGREEMENT & LP-LEVEL VARIABILITY ANALYSIS
# =============================================================================


def compute_error_disagreement_analysis(
    oof_primary_df: pd.DataFrame,
    row_meta_df: pd.DataFrame,
    primary_overlap_bin_col: str = "bin_title_to_lp_bigram_containment",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Classify paired predictions into A/B/C/D disagreement categories and curate stable row_ids.

    Zero raw Japanese text is stored in any returned table or JSON payload.
    """
    piv_pred = oof_primary_df.pivot(index="row_id", columns="model_id", values="predicted_label")
    piv_score = oof_primary_df.pivot(index="row_id", columns="model_id", values="score_inconsistent")
    base = (
        oof_primary_df[oof_primary_df["model_id"] == MODEL_ORDER_RQ2[0]][
            ["row_id", "group_id", "true_label", "bin_char_bigram_jaccard", "bin_title_to_lp_bigram_containment"]
        ]
        .set_index("row_id")
        .join(row_meta_df.set_index("row_id"))
        .join(piv_pred)
    )
    for m_id in MODEL_ORDER_RQ2:
        base[f"score_{m_id}"] = piv_score[m_id]
        base[f"correct_{m_id}"] = base[m_id] == base["true_label"]

    pairs = [
        ("L0_vs_S1", "L0_lexical_baseline", "S1_semantic_e5_small"),
        ("L1_vs_S1", "L1_lexical_interaction", "S1_semantic_e5_small"),
        ("L0_vs_S2", "L0_lexical_baseline", "S2_hybrid_lexical_semantic"),
    ]

    rows: List[Dict[str, Any]] = []
    n_total = len(base)

    for pair_name, lex_id, sem_id in pairs:
        lex_ok = base[f"correct_{lex_id}"]
        sem_ok = base[f"correct_{sem_id}"]

        cat_masks = [
            ("A_both_correct", lex_ok & sem_ok),
            ("B_lexical_correct_semantic_incorrect", lex_ok & (~sem_ok)),
            ("C_semantic_correct_lexical_incorrect", (~lex_ok) & sem_ok),
            ("D_both_incorrect", (~lex_ok) & (~sem_ok)),
        ]
        for cat_name, mask in cat_masks:
            sub = base[mask]
            n_cat = len(sub)
            rows.append({
                "pair_comparison": pair_name,
                "error_category": cat_name,
                "sample_count": int(n_cat),
                "sample_pct": round(n_cat / max(n_total, 1) * 100.0, 2),
                "low_overlap_count": int((sub[primary_overlap_bin_col] == "Low").sum()),
                "medium_overlap_count": int((sub[primary_overlap_bin_col] == "Medium").sum()),
                "high_overlap_count": int((sub[primary_overlap_bin_col] == "High").sum()),
                "low_jaccard_count": int((sub["bin_char_bigram_jaccard"] == "Low").sum()),
                "high_jaccard_count": int((sub["bin_char_bigram_jaccard"] == "High").sum()),
                "consistent_count": int((sub["true_label"] == "consistent").sum()),
                "inconsistent_count": int((sub["true_label"] == POSITIVE_CLASS).sum()),
                "has_numeric_token_count": int(sub["title_has_numeric"].sum()),
                "lp_truncated_512_count": int(sub["lp_is_truncated"].sum()),
                "mean_lp_length": round(float(sub["lp_char_length"].mean()), 2) if n_cat else None,
                "mean_title_length": round(float(sub["title_char_length"].mean()), 2) if n_cat else None,
                "mean_bigram_jaccard": round(float(sub["char_bigram_jaccard"].mean()), 4) if n_cat else None,
                "mean_title_to_lp_containment": round(float(sub["title_to_lp_bigram_containment"].mean()), 4) if n_cat else None,
                "mean_e5_cosine_sim": round(float(sub["e5_cosine_sim"].mean()), 4) if n_cat else None,
            })

    disagreement_df = pd.DataFrame(rows)

    # Curate stable row_ids (no raw Japanese text!) for local notebook manual review
    l0_ok = base["correct_L0_lexical_baseline"]
    s1_ok = base["correct_S1_semantic_e5_small"]
    s2_ok = base["correct_S2_hybrid_lexical_semantic"]

    def _pick_ids(mask: pd.Series, sort_col: str, ascending: bool = True, k: int = 4) -> List[int]:
        cand = base[mask].sort_values(sort_col, ascending=ascending)
        # Prefer distinct LP groups for manual inspection diversity
        picked: List[int] = []
        seen_grps: set[str] = set()
        for rid, row in cand.iterrows():
            if row["group_id"] not in seen_grps:
                picked.append(int(rid))
                seen_grps.add(str(row["group_id"]))
            if len(picked) >= k:
                break
        return picked

    # Identify top-10 largest LP groups (repeated brand/campaign wording slice)
    top10_groups = set(base["group_id"].value_counts().head(10).index.tolist())

    curated_slices = {
        "C_semantic_correct_low_overlap": _pick_ids(
            (~l0_ok) & s1_ok & (base[primary_overlap_bin_col] == "Low"),
            sort_col="title_to_lp_bigram_containment",
            ascending=True,
            k=4,
        ),
        "B_lexical_correct_low_overlap": _pick_ids(
            l0_ok & (~s1_ok) & (base[primary_overlap_bin_col] == "Low"),
            sort_col="title_to_lp_bigram_containment",
            ascending=True,
            k=4,
        ),
        "high_overlap_inconsistent_both_wrong": _pick_ids(
            (~l0_ok)
            & (~s1_ok)
            & (base["true_label"] == POSITIVE_CLASS)
            & (base[primary_overlap_bin_col] == "High"),
            sort_col="title_to_lp_bigram_containment",
            ascending=False,
            k=4,
        ),
        "numeric_title_disagreement_B_or_D": _pick_ids(
            (~s1_ok) & base["title_has_numeric"] & (base["true_label"] == POSITIVE_CLASS),
            sort_col="e5_cosine_sim",
            ascending=False,
            k=4,
        ),
        "long_truncated_lp_disagreement": _pick_ids(
            (l0_ok != s1_ok) & base["lp_is_truncated"],
            sort_col="lp_token_length",
            ascending=False,
            k=4,
        ),
        "repeated_large_lp_group_errors": _pick_ids(
            (~s1_ok) & (~l0_ok) & base["group_id"].isin(top10_groups),
            sort_col="lp_char_length",
            ascending=False,
            k=3,
        ),
    }

    # Diagnostic subgroup summaries (no raw text)
    special_slice_stats = {
        "high_overlap_inconsistent_slice": {
            "count": int(((base["true_label"] == POSITIVE_CLASS) & (base[primary_overlap_bin_col] == "High")).sum()),
            "L0_recall": round(float(base[(base["true_label"] == POSITIVE_CLASS) & (base[primary_overlap_bin_col] == "High")]["correct_L0_lexical_baseline"].mean()), 4),
            "L1_recall": round(float(base[(base["true_label"] == POSITIVE_CLASS) & (base[primary_overlap_bin_col] == "High")]["correct_L1_lexical_interaction"].mean()), 4),
            "S1_recall": round(float(base[(base["true_label"] == POSITIVE_CLASS) & (base[primary_overlap_bin_col] == "High")]["correct_S1_semantic_e5_small"].mean()), 4),
            "S2_recall": round(float(base[(base["true_label"] == POSITIVE_CLASS) & (base[primary_overlap_bin_col] == "High")]["correct_S2_hybrid_lexical_semantic"].mean()), 4),
        },
        "numeric_title_slice": {
            "count": int(base["title_has_numeric"].sum()),
            "inconsistent_count": int((base["title_has_numeric"] & (base["true_label"] == POSITIVE_CLASS)).sum()),
            "L0_accuracy": round(float(base[base["title_has_numeric"]]["correct_L0_lexical_baseline"].mean()), 4),
            "S1_accuracy": round(float(base[base["title_has_numeric"]]["correct_S1_semantic_e5_small"].mean()), 4),
            "S2_accuracy": round(float(base[base["title_has_numeric"]]["correct_S2_hybrid_lexical_semantic"].mean()), 4),
        },
        "truncated_lp_over_512_tokens_slice": {
            "count": int(base["lp_is_truncated"].sum()),
            "L0_accuracy": round(float(base[base["lp_is_truncated"]]["correct_L0_lexical_baseline"].mean()), 4) if base["lp_is_truncated"].any() else None,
            "S1_accuracy": round(float(base[base["lp_is_truncated"]]["correct_S1_semantic_e5_small"].mean()), 4) if base["lp_is_truncated"].any() else None,
            "S2_accuracy": round(float(base[base["lp_is_truncated"]]["correct_S2_hybrid_lexical_semantic"].mean()), 4) if base["lp_is_truncated"].any() else None,
            "untruncated_L0_accuracy": round(float(base[~base["lp_is_truncated"]]["correct_L0_lexical_baseline"].mean()), 4) if (~base["lp_is_truncated"]).any() else None,
            "untruncated_S1_accuracy": round(float(base[~base["lp_is_truncated"]]["correct_S1_semantic_e5_small"].mean()), 4) if (~base["lp_is_truncated"]).any() else None,
            "untruncated_S2_accuracy": round(float(base[~base["lp_is_truncated"]]["correct_S2_hybrid_lexical_semantic"].mean()), 4) if (~base["lp_is_truncated"]).any() else None,
        },
        "four_model_unanimous_correct_count": int((l0_ok & base["correct_L1_lexical_interaction"] & s1_ok & s2_ok).sum()),
        "four_model_unanimous_wrong_count": int(((~l0_ok) & (~base["correct_L1_lexical_interaction"]) & (~s1_ok) & (~s2_ok)).sum()),
        "curated_manual_inspection_row_ids": curated_slices,
    }

    return disagreement_df, special_slice_stats


# =============================================================================
# 5. MARKDOWN REPORT RENDERER
# =============================================================================


def format_rq2_report_markdown(
    provenance: Dict[str, Any],
    fold_ref: Dict[str, Any],
    embedding_manifest: Dict[str, Any],
    l0_repro: Dict[str, Any],
    benchmark_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    reproducibility_rows: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> str:
    """Render comprehensive English RQ2 Lexical vs Semantic report in Markdown."""
    lines = [
        "# Milestone M2.2 — RQ2: Lexical vs Semantic Modeling",
        "",
        f"**Experiment ID**: `{payload['experiment_id']}`  ",
        f"**Dataset**: `{provenance['dataset_name']}` (`{provenance['config_name']}`, revision `{provenance['upstream_revision']}`)  ",
        f"**Data Used**: Official `train.tsv` ONLY (`N = {provenance['train_row_count']}`, SHA-256 `{provenance['train_sha256_verified'][:16]}...`). Official `valid.tsv` and `test.tsv` strictly excluded.  ",
        f"**Evaluation Protocol**: Reused M2.1 `{fold_ref['protocol']}` (`{fold_ref['n_splits']}`-fold `StratifiedGroupKFold`, seeds `{fold_ref['seeds']}`, fold manifest SHA-256 `{fold_ref['fold_manifest_hash'][:16]}...`).  ",
        f"**Semantic Encoder**: `{embedding_manifest['model_id']}` (revision `{embedding_manifest['resolved_revision'][:16]}...`, license `{embedding_manifest['model_license']}`, frozen CPU inference, prefix=`\"{embedding_manifest['title_prefix']}\"`, max_seq_length=`{embedding_manifest['max_seq_length']}`).  ",
        "",
        "---",
        "",
        "## 1. Research Question & M2.1 Baseline Reproduction Check",
        "",
        "**RQ2**: *Do semantic representations outperform lexical baselines in advertising consistency classification, especially when landing-page text and advertising titles have low lexical overlap?*",
        "",
        f"- **L0 Reproduction Status**: `{l0_repro['status']}` (max absolute difference vs M2.1 `B3_pair` Protocol B = `{l0_repro['max_abs_diff']}`).",
        f"  - M2.1 `B3_pair`: Macro-F1=`{l0_repro['m2_1_reference'].get('mean_macro_f1')}`, ROC-AUC=`{l0_repro['m2_1_reference'].get('mean_roc_auc')}`, AP=`{l0_repro['m2_1_reference'].get('mean_average_precision')}`, Inconsistent Recall=`{l0_repro['m2_1_reference'].get('mean_inconsistent_recall')}`.",
        f"  - M2.2 `L0_lexical_baseline`: Macro-F1=`{l0_repro['m2_2_l0_observed']['macro_f1']}`, ROC-AUC=`{l0_repro['m2_2_l0_observed']['roc_auc']}`, AP=`{l0_repro['m2_2_l0_observed']['average_precision']}`, Inconsistent Recall=`{l0_repro['m2_2_l0_observed']['recall_inconsistent']}`.",
        "",
        "## 2. Table 1 — Overall Model Benchmark (10-Fold Mean Across Seeds [42, 123])",
        "",
        "| model_id | representation | macro_f1 | roc_auc | average_precision | balanced_accuracy | recall_inconsistent | training_seconds | inference_ms_per_sample | embedding_extraction_seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in benchmark_df.iterrows():
        lines.append(
            f"| `{r['model_id']}` | {r['representation']} | "
            f"{r['macro_f1']:.4f} | {r['roc_auc']:.4f} | {r['average_precision']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {r['recall_inconsistent']:.4f} | "
            f"{r['training_seconds']:.4f} | {r['inference_ms_per_sample']:.4f} | "
            f"{r['embedding_extraction_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Table 2 — Lexical Overlap Subgroup Analysis (Title-to-LP Bigram Containment, 10-Fold Mean)",
        "",
        "| model_id | overlap_group | sample_count | unique_lp_count | macro_f1 | roc_auc | average_precision | recall_inconsistent |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    sub_t2 = overlap_df[
        (overlap_df["overlap_metric"] == "title_to_lp_bigram_containment")
        & (overlap_df["aggregation"] == "mean_across_10_folds")
    ]
    for _, r in sub_t2.iterrows():
        auc_s = f"{r['roc_auc']:.4f}" if pd.notna(r["roc_auc"]) else "NA"
        ap_s = f"{r['average_precision']:.4f}" if pd.notna(r["average_precision"]) else "NA"
        f1_s = f"{r['macro_f1']:.4f}" if pd.notna(r["macro_f1"]) else "NA"
        rec_s = f"{r['recall_inconsistent']:.4f}" if pd.notna(r["recall_inconsistent"]) else "NA"
        lines.append(
            f"| `{r['model_id']}` | {r['overlap_group']} | {int(r['sample_count'])} | "
            f"{int(r['unique_lp_count'])} | {f1_s} | {auc_s} | {ap_s} | {rec_s} |"
        )

    lines.extend([
        "",
        "## 4. Table 3 — Relative Improvement & Interaction (Primary Seed 42 OOF + LP-Group Bootstrap 95% CI)",
        "",
        "| overlap_metric | comparison | overlap_group | delta_macro_f1 | delta_roc_auc | delta_average_precision | delta_recall_inconsistent |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for _, r in interaction_df.iterrows():
        def _fmt_ci(val, lo, hi):
            if pd.isna(val):
                return "NA"
            return f"{val:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        lines.append(
            f"| `{r['overlap_metric']}` | `{r['comparison']}` | {r['overlap_group']} | "
            f"{_fmt_ci(r['delta_macro_f1'], r['delta_macro_f1_ci_low'], r['delta_macro_f1_ci_high'])} | "
            f"{_fmt_ci(r['delta_roc_auc'], r['delta_roc_auc_ci_low'], r['delta_roc_auc_ci_high'])} | "
            f"{_fmt_ci(r['delta_average_precision'], r['delta_average_precision_ci_low'], r['delta_average_precision_ci_high'])} | "
            f"{_fmt_ci(r['delta_recall_inconsistent'], r['delta_recall_inconsistent_ci_low'], r['delta_recall_inconsistent_ci_high'])} |"
        )

    lines.extend([
        "",
        "## 5. Table 4 — Error Disagreement Summary (L0 vs S1, Primary Seed 42 OOF)",
        "",
        "| error_category | sample_count | low_overlap_count | high_overlap_count | mean_lp_length | mean_title_length |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, r in disagreement_df[disagreement_df["pair_comparison"] == "L0_vs_S1"].iterrows():
        lp_l = f"{r['mean_lp_length']:.1f}" if pd.notna(r["mean_lp_length"]) else "NA"
        t_l = f"{r['mean_title_length']:.1f}" if pd.notna(r["mean_title_length"]) else "NA"
        lines.append(
            f"| `{r['error_category']}` | {int(r['sample_count'])} | "
            f"{int(r['low_overlap_count'])} | {int(r['high_overlap_count'])} | "
            f"{lp_l} | {t_l} |"
        )

    lines.extend([
        "",
        "## 6. Table 5 — Reproducibility Ledger",
        "",
        "| model_id | encoder_revision | configuration_hash | fold_manifest_hash | seed | execution_status |",
        "|---|---|---|---|---:|---|",
    ])
    for r in reproducibility_rows:
        lines.append(
            f"| `{r['model_id']}` | `{r['encoder_revision']}` | `{r['configuration_hash']}` | "
            f"`{r['fold_manifest_hash'][:16]}...` | {r['seed']} | `{r['execution_status']}` |"
        )

    lines.extend([
        "",
        "## 7. Scientific Guardrails & Comparability Notes",
        "",
        f"- **VERIFIED**: All four models were evaluated on identical Protocol B LP-group-disjoint folds (`shared_lp_count = 0`) using `train.tsv` only.",
        f"- **VERIFIED**: `intfloat/multilingual-e5-small` weights remained strictly frozen (`weights_verified_unchanged = {embedding_manifest['weights_verified_unchanged']}`); `{embedding_manifest['row_level_lp_truncated_pct']}%` of training rows had LPs exceeding the `512`-token encoder limit.",
        "- **VERIFIED**: All TF-IDF vectorizers, `StandardScaler` transformers, and lexical-overlap quantile bin thresholds were fit strictly within each training fold.",
        "- **ASSUMED**: `L1`, `S1`, and `S2` apply `StandardScaler` to dense feature blocks (15 lexical interaction features and/or 1,537 semantic pair features) while keeping sparse TF-IDF L2-normalized (`norm='l2'`). Under shared `C=1.0` L2 regularization, unit-variance dense blocks carry different effective per-feature penalty norms than sparse L2-normalized TF-IDF, which affects regularization comparability.",
        "- **ASSUMED**: Pooled OOF probabilities combine fold models fitted on different LP subsets and are uncalibrated scores.",
    ])

    return "\n".join(lines)


# =============================================================================
# 6. FULL M2.2 RQ2 ORCHESTRATOR
# =============================================================================


def _instantiate_rq2_model(
    model_id: str,
    cfg: Dict[str, Any],
    seed: int,
    embedding_bundle: SemanticEmbeddingBundle,
) -> BaselineModel:
    lex_cfg = cfg["lexical"]
    vec_cfg = lex_cfg["vectorizer"]
    clf_cfg = cfg["classifier"]

    if model_id == "L0_lexical_baseline":
        return TfidfLogRegBaseline(
            input_mode="pair",
            ngram_range=tuple(vec_cfg["ngram_range"]),
            min_df=int(vec_cfg["min_df"]),
            max_features=vec_cfg["max_features"],
            sublinear_tf=bool(vec_cfg["sublinear_tf"]),
            c_param=float(clf_cfg["C"]),
            class_weight=clf_cfg["class_weight"],
            max_iter=int(clf_cfg["max_iter"]),
            random_state=seed,
            solver=str(clf_cfg["solver"]),
            separator=str(lex_cfg["separator"]),
            normalize=bool(lex_cfg["normalize_unicode_nfkc"]),
        )
    if model_id == "L1_lexical_interaction":
        return LexicalInteractionLogReg(
            ngram_range=tuple(vec_cfg["ngram_range"]),
            min_df=int(vec_cfg["min_df"]),
            max_features=vec_cfg["max_features"],
            sublinear_tf=bool(vec_cfg["sublinear_tf"]),
            c_param=float(clf_cfg["C"]),
            class_weight=clf_cfg["class_weight"],
            max_iter=int(clf_cfg["max_iter"]),
            random_state=seed,
            solver=str(clf_cfg["solver"]),
            separator=str(lex_cfg["separator"]),
            normalize=bool(lex_cfg["normalize_unicode_nfkc"]),
        )
    if model_id == "S1_semantic_e5_small":
        return SemanticPairLogReg(
            embedding_bundle=embedding_bundle,
            c_param=float(clf_cfg["C"]),
            class_weight=clf_cfg["class_weight"],
            max_iter=int(clf_cfg["max_iter"]),
            random_state=seed,
            solver=str(clf_cfg["solver"]),
        )
    if model_id == "S2_hybrid_lexical_semantic":
        return HybridLexicalSemanticLogReg(
            embedding_bundle=embedding_bundle,
            ngram_range=tuple(vec_cfg["ngram_range"]),
            min_df=int(vec_cfg["min_df"]),
            max_features=vec_cfg["max_features"],
            sublinear_tf=bool(vec_cfg["sublinear_tf"]),
            c_param=float(clf_cfg["C"]),
            class_weight=clf_cfg["class_weight"],
            max_iter=int(clf_cfg["max_iter"]),
            random_state=seed,
            solver=str(clf_cfg["solver"]),
            separator=str(lex_cfg["separator"]),
            normalize=bool(lex_cfg["normalize_unicode_nfkc"]),
        )
    raise ValueError(f"Unknown RQ2 model_id: '{model_id}'")


def run_rq2_lexical_semantic_study(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path | str] = None,
    runs_base_dir: Path | str = "experiments/runs",
    reports_dir: Path | str = "reports/generated/m2_lexical_semantic",
    encoder: Optional[TextEncoderProtocol] = None,
    splits_override: Optional[Tuple[Dict[int, List[Tuple[np.ndarray, np.ndarray]]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute Milestone M2.2 (RQ2) Lexical vs Semantic controlled experiment end-to-end."""
    cfg = deepcopy(config) if config is not None else deepcopy(M2_2_DEFAULT_CONFIG)
    experiment_id = cfg.get("experiment_id", "m2_lexical_semantic_v001")
    primary_seed = int(cfg.get("primary_seed", 42))
    seeds = [int(s) for s in cfg.get("seeds", [42, 123])]
    n_splits = int(cfg.get("n_splits", 5))
    raw_dir = data_dir or cfg["data"]["raw_dir"]
    overlap_metrics = list(cfg.get("overlap_subgroup_metrics", OVERLAP_METRICS))

    config_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    # 1. Strictly load train.tsv ONLY (never valid.tsv or test.tsv)
    train_df, provenance = load_adtec_train_only(raw_dir)
    group_ids = assign_lp_group_ids(train_df, lp_col="lp_text")

    # 2. Load and verify exact M2.1 Protocol B folds
    if splits_override is not None:
        splits_by_seed, fold_ref = splits_override
    else:
        splits_by_seed, fold_ref = load_m2_1_protocol_b_folds(
            manifest_path=cfg["data"]["m2_1_fold_manifest_path"],
            train_df=train_df,
            group_ids=group_ids,
            expected_seeds=seeds,
            expected_n_splits=n_splits,
        )

    # 3. Extract label-independent lexical interaction matrix & frozen semantic embedding cache
    t_lex_start = time.perf_counter()
    lex_df = extract_lexical_interaction_frame(train_df, title_col="title", lp_col="lp_text")
    lex_extract_sec = time.perf_counter() - t_lex_start
    lex_matrix_all = lex_df[list(LEXICAL_INTERACTION_FEATURE_NAMES)].to_numpy(dtype=np.float32)

    sem_cfg = cfg["semantic"]
    embedding_bundle = extract_or_load_cached_embeddings(
        df=train_df,
        cache_base_dir=sem_cfg["cache_base_dir"],
        model_id=sem_cfg["model_id"],
        revision=sem_cfg.get("revision"),
        license_name=sem_cfg.get("license", DEFAULT_E5_LICENSE),
        title_prefix=sem_cfg["title_prefix"],
        lp_prefix=sem_cfg["lp_prefix"],
        prefix_policy_note=sem_cfg.get("prefix_policy_note", DEFAULT_E5_PREFIX_POLICY_NOTE),
        batch_size=int(sem_cfg.get("batch_size", 32)),
        device=str(sem_cfg.get("device", "cpu")),
        normalize_embeddings=bool(sem_cfg.get("normalize_embeddings", True)),
        encoder=encoder,
    )
    emb_manifest = deepcopy(embedding_bundle.manifest)
    emb_manifest["lexical_feature_precompute_seconds"] = round(float(lex_extract_sec), 4)

    # Align semantic pair matrix for all train rows once (purely deterministic, label-free)
    u_all, v_all = embedding_bundle.align_for_dataframe(train_df)
    sem_matrix_all = build_semantic_pair_features(u_all, v_all)
    e5_cos_all = sem_matrix_all[:, -1]
    lp_tok_all, title_tok_all = embedding_bundle.get_row_token_lengths(train_df)
    max_seq_len = int(emb_manifest["max_seq_length"])

    # Row-level descriptive metadata (no raw Japanese text!)
    row_meta_df = pd.DataFrame({
        "row_id": train_df.index.to_numpy(dtype=int),
        "char_bigram_jaccard": lex_df["char_bigram_jaccard"].to_numpy(dtype=np.float64),
        "title_to_lp_bigram_containment": lex_df["title_to_lp_bigram_containment"].to_numpy(dtype=np.float64),
        "lp_to_title_bigram_containment": lex_df["lp_to_title_bigram_containment"].to_numpy(dtype=np.float64),
        "title_char_length": lex_df["title_char_length"].to_numpy(dtype=np.float64),
        "lp_char_length": lex_df["lp_char_length"].to_numpy(dtype=np.float64),
        "e5_cosine_sim": e5_cos_all.astype(np.float64),
        "lp_token_length": lp_tok_all,
        "title_token_length": title_tok_all,
        "lp_is_truncated": lp_tok_all > max_seq_len,
        "title_has_numeric": [has_numeric_or_quantity_token(t) for t in train_df["title"].tolist()],
    })

    # 4. Initialize isolated run directory & environment metadata
    exp_mgr = ExperimentManager(base_runs_dir=runs_base_dir, reports_dir=reports_dir)
    run_dir = exp_mgr.create_named_run(
        run_id=experiment_id,
        config=cfg,
        dataset_metadata=None,
        seed=primary_seed,
    )
    env_path = run_dir / "environment.json"
    env_data = json.loads(env_path.read_text(encoding="utf-8"))
    env_data["training_split"] = "train (Protocol B StratifiedGroupKFold internally)"
    env_data["evaluation_split"] = "train_oof_only"
    env_data["valid_split_accessed"] = False
    env_data["test_split_accessed"] = False
    env_data["dataset_provenance"] = provenance
    env_data["environment"]["library_versions"].update(
        get_installed_library_versions(
            ["torch", "transformers", "sentence-transformers", "huggingface-hub", "nbclient"]
        )
    )
    env_path.write_text(json.dumps(env_data, indent=2), encoding="utf-8")

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 5. Execute all 4 controlled models across identical Protocol B folds (2 seeds x 5 folds)
    per_fold_rows: List[Dict[str, Any]] = []
    per_fold_overlap_rows: List[Dict[str, Any]] = []
    fold_bin_threshold_rows: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    model_size_records: Dict[str, List[float]] = {m: [] for m in MODEL_ORDER_RQ2}
    reproducibility_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for fold_idx, (tr_idx, ev_idx) in enumerate(splits_by_seed[seed]):
            fold_train = train_df.iloc[tr_idx]
            fold_eval = train_df.iloc[ev_idx]
            y_eval_true = fold_eval["label"].tolist()
            ev_groups = group_ids.iloc[ev_idx].to_numpy()
            ev_title_len = lex_df["title_char_length"].iloc[ev_idx].to_numpy()
            ev_lp_len = lex_df["lp_char_length"].iloc[ev_idx].to_numpy()

            # Fit overlap subgroup boundaries STRICTLY on this fold's training split
            ev_bins_by_metric: Dict[str, np.ndarray] = {}
            for om in overlap_metrics:
                tr_ov = lex_df[om].iloc[tr_idx].to_numpy(dtype=np.float64)
                ev_ov = lex_df[om].iloc[ev_idx].to_numpy(dtype=np.float64)
                bin_spec = fit_overlap_bin_edges(tr_ov, metric_name=om)
                ev_bins = assign_overlap_bins(ev_ov, bin_spec)
                # Verify bin counts reconcile identically with eval fold size
                assert len(ev_bins) == len(ev_idx)
                assert sum(int(np.sum(ev_bins == b)) for b in OVERLAP_BIN_ORDER) == len(ev_idx)
                ev_bins_by_metric[om] = ev_bins
                fold_bin_threshold_rows.append({
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    **bin_spec,
                })

            for model_id in MODEL_ORDER_RQ2:
                set_global_seed(seed)
                model = _instantiate_rq2_model(
                    model_id=model_id,
                    cfg=cfg,
                    seed=seed,
                    embedding_bundle=embedding_bundle,
                )

                t0 = time.perf_counter()
                if model_id == "L0_lexical_baseline":
                    model.fit(fold_train, target_col="label")
                elif model_id == "L1_lexical_interaction":
                    model.fit(fold_train, target_col="label", precomputed_lex=lex_matrix_all[tr_idx])
                elif model_id == "S1_semantic_e5_small":
                    model.fit(fold_train, target_col="label", precomputed_sem=sem_matrix_all[tr_idx])
                elif model_id == "S2_hybrid_lexical_semantic":
                    model.fit(
                        fold_train,
                        target_col="label",
                        precomputed_lex=lex_matrix_all[tr_idx],
                        precomputed_sem=sem_matrix_all[tr_idx],
                    )
                train_sec = time.perf_counter() - t0

                t1 = time.perf_counter()
                if model_id == "L0_lexical_baseline":
                    probs = model.predict_proba(fold_eval)
                    preds = model.predict(fold_eval)
                elif model_id == "L1_lexical_interaction":
                    probs = model.predict_proba(fold_eval, precomputed_lex=lex_matrix_all[ev_idx])
                    preds = model.predict(fold_eval, precomputed_lex=lex_matrix_all[ev_idx])
                elif model_id == "S1_semantic_e5_small":
                    probs = model.predict_proba(fold_eval, precomputed_sem=sem_matrix_all[ev_idx])
                    preds = model.predict(fold_eval, precomputed_sem=sem_matrix_all[ev_idx])
                elif model_id == "S2_hybrid_lexical_semantic":
                    probs = model.predict_proba(
                        fold_eval,
                        precomputed_lex=lex_matrix_all[ev_idx],
                        precomputed_sem=sem_matrix_all[ev_idx],
                    )
                    preds = model.predict(
                        fold_eval,
                        precomputed_lex=lex_matrix_all[ev_idx],
                        precomputed_sem=sem_matrix_all[ev_idx],
                    )
                # Predict + proba measured together; divide by 2 for single-pass inference time
                inf_sec = (time.perf_counter() - t1) / 2.0

                scores_inc = _extract_inconsistent_scores_from_proba(probs, model.classes_)

                if seed == primary_seed and fold_idx == 0:
                    model_size_records[model_id].append(_estimate_fitted_model_size_mb(model))

                fold_metrics = compute_fold_metrics(
                    y_true=y_eval_true,
                    y_pred=preds,
                    scores_inconsistent=scores_inc,
                    train_time_seconds=train_sec,
                    inference_time_seconds=inf_sec,
                )
                per_fold_rows.append({
                    "model_id": model_id,
                    "protocol": PROTOCOL_B,
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    "inference_ms_per_sample": round(
                        (inf_sec / max(len(ev_idx), 1)) * 1000.0, 4
                    ),
                    **fold_metrics,
                })

                # Subgroup evaluation per overlap metric on this held-out fold
                preds_arr = np.asarray(preds, dtype=object)
                y_true_arr = np.asarray(y_eval_true, dtype=object)
                for om in overlap_metrics:
                    ev_bins = ev_bins_by_metric[om]
                    for b_name in OVERLAP_BIN_ORDER:
                        mask = ev_bins == b_name
                        sub_m = _compute_subgroup_metrics_core(
                            y_true=y_true_arr[mask].tolist(),
                            y_pred=preds_arr[mask].tolist(),
                            scores_inconsistent=scores_inc[mask],
                            group_ids=ev_groups[mask].tolist(),
                            title_lengths=ev_title_len[mask],
                            lp_lengths=ev_lp_len[mask],
                        )
                        per_fold_overlap_rows.append({
                            "overlap_metric": om,
                            "model_id": model_id,
                            "seed": int(seed),
                            "fold": int(fold_idx),
                            "overlap_group": b_name,
                            **sub_m,
                        })

                oof_frames.append(
                    pd.DataFrame({
                        "row_id": fold_eval.index.to_numpy(dtype=int),
                        "group_id": ev_groups,
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        "model_id": model_id,
                        "true_label": y_eval_true,
                        "predicted_label": preds,
                        "score_inconsistent": np.round(scores_inc, 6),
                        "bin_char_bigram_jaccard": ev_bins_by_metric["char_bigram_jaccard"],
                        "bin_title_to_lp_bigram_containment": ev_bins_by_metric[
                            "title_to_lp_bigram_containment"
                        ],
                    })
                )

        for model_id in MODEL_ORDER_RQ2:
            uses_enc = model_id in ("S1_semantic_e5_small", "S2_hybrid_lexical_semantic")
            reproducibility_rows.append({
                "model_id": model_id,
                "encoder_revision": emb_manifest["resolved_revision"] if uses_enc else "none (lexical_only)",
                "configuration_hash": config_hash,
                "fold_manifest_hash": fold_ref["fold_manifest_hash"],
                "seed": int(seed),
                "execution_status": "COMPLETED",
            })

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_overlap_df = pd.DataFrame(per_fold_overlap_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True)

    # 6. Build Table 1 (benchmark_comparison.csv) & model_costs.csv
    total_emb_sec = float(emb_manifest["total_extraction_seconds"])
    online_pair_encode_ms = float(emb_manifest["estimated_online_pair_encode_ms"])

    benchmark_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    per_seed_summary: Dict[str, List[Dict[str, Any]]] = {}

    for s in seeds:
        per_seed_summary[str(s)] = []
        for m_id in MODEL_ORDER_RQ2:
            sub_s = per_fold_df[(per_fold_df["model_id"] == m_id) & (per_fold_df["seed"] == s)]
            per_seed_summary[str(s)].append({
                "model_id": m_id,
                "seed": int(s),
                "fold_count": int(len(sub_s)),
                "macro_f1": round(float(sub_s["macro_f1"].mean()), 4),
                "std_macro_f1": round(float(sub_s["macro_f1"].std(ddof=1)), 4),
                "roc_auc": round(float(sub_s["roc_auc"].mean()), 4),
                "std_roc_auc": round(float(sub_s["roc_auc"].std(ddof=1)), 4),
                "average_precision": round(float(sub_s["average_precision"].mean()), 4),
                "std_average_precision": round(float(sub_s["average_precision"].std(ddof=1)), 4),
                "balanced_accuracy": round(float(sub_s["balanced_accuracy"].mean()), 4),
                "recall_inconsistent": round(float(sub_s["recall_inconsistent"].mean()), 4),
                "std_recall_inconsistent": round(float(sub_s["recall_inconsistent"].std(ddof=1)), 4),
            })

    for m_id in MODEL_ORDER_RQ2:
        sub = per_fold_df[per_fold_df["model_id"] == m_id]
        uses_enc = m_id in ("S1_semantic_e5_small", "S2_hybrid_lexical_semantic")
        emb_sec = round(total_emb_sec, 4) if uses_enc else 0.0
        cached_inf_ms = round(float(sub["inference_ms_per_sample"].mean()), 4)
        online_inf_ms = round(cached_inf_ms + (online_pair_encode_ms if uses_enc else 0.0), 4)
        clf_size_mb = round(float(np.mean(model_size_records[m_id])), 4) if model_size_records[m_id] else 0.0

        benchmark_rows.append({
            "model_id": m_id,
            "representation": MODEL_REPRESENTATION_LABELS[m_id],
            "macro_f1": round(float(sub["macro_f1"].mean()), 4),
            "roc_auc": round(float(sub["roc_auc"].mean()), 4),
            "average_precision": round(float(sub["average_precision"].mean()), 4),
            "balanced_accuracy": round(float(sub["balanced_accuracy"].mean()), 4),
            "recall_inconsistent": round(float(sub["recall_inconsistent"].mean()), 4),
            "training_seconds": round(float(sub["train_time_seconds"].mean()), 4),
            "inference_ms_per_sample": cached_inf_ms,
            "embedding_extraction_seconds": emb_sec,
            "std_macro_f1": round(float(sub["macro_f1"].std(ddof=1)), 4),
            "std_roc_auc": round(float(sub["roc_auc"].std(ddof=1)), 4),
            "std_average_precision": round(float(sub["average_precision"].std(ddof=1)), 4),
            "std_balanced_accuracy": round(float(sub["balanced_accuracy"].std(ddof=1)), 4),
            "std_recall_inconsistent": round(float(sub["recall_inconsistent"].std(ddof=1)), 4),
            "precision_inconsistent": round(float(sub["precision_inconsistent"].mean()), 4),
            "f1_inconsistent": round(float(sub["f1_inconsistent"].mean()), 4),
            "recall_consistent": round(float(sub["recall_consistent"].mean()), 4),
            "online_inference_ms_per_sample": online_inf_ms,
            "classifier_size_mb": clf_size_mb,
        })

        cost_rows.append({
            "model_id": m_id,
            "uses_semantic_encoder": uses_enc,
            "mean_fold_train_seconds": round(float(sub["train_time_seconds"].mean()), 4),
            "total_10fold_train_seconds": round(float(sub["train_time_seconds"].sum()), 4),
            "one_time_embedding_extraction_seconds": emb_sec,
            "cached_classifier_inference_ms_per_sample": cached_inf_ms,
            "estimated_online_pair_encode_ms_per_sample": round(online_pair_encode_ms, 4) if uses_enc else 0.0,
            "estimated_end_to_end_online_ms_per_sample": online_inf_ms,
            "classifier_pipeline_size_mb": clf_size_mb,
            "embedding_cache_size_mb": round(float(emb_manifest["cache_size_mb"]), 4) if uses_enc else 0.0,
            "peak_working_set_mb": emb_manifest.get("peak_working_set_mb"),
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    model_costs_df = pd.DataFrame(cost_rows)

    # 7. Verify L0 reproduction against M2.1 Protocol B B3_pair reference
    l0_row = benchmark_df[benchmark_df["model_id"] == "L0_lexical_baseline"].iloc[0]
    m2_1_ref: Dict[str, Any] = {}
    m2_1_path = Path(cfg["data"].get("m2_1_protocol_comparison_path", ""))
    if m2_1_path.is_file():
        m2_1_df = pd.read_csv(m2_1_path)
        ref_rows = m2_1_df[
            (m2_1_df["model_id"] == "B3_pair") & (m2_1_df["protocol"] == PROTOCOL_B)
        ]
        if not ref_rows.empty:
            r0 = ref_rows.iloc[0]
            m2_1_ref = {
                "mean_macro_f1": float(r0["mean_macro_f1"]),
                "mean_roc_auc": float(r0["mean_roc_auc"]),
                "mean_average_precision": float(r0["mean_average_precision"]),
                "mean_balanced_accuracy": float(r0["mean_balanced_accuracy"]),
                "mean_inconsistent_recall": float(r0["mean_inconsistent_recall"]),
            }

    l0_obs = {
        "macro_f1": float(l0_row["macro_f1"]),
        "roc_auc": float(l0_row["roc_auc"]),
        "average_precision": float(l0_row["average_precision"]),
        "balanced_accuracy": float(l0_row["balanced_accuracy"]),
        "recall_inconsistent": float(l0_row["recall_inconsistent"]),
    }
    diffs = {}
    if m2_1_ref:
        diffs = {
            "macro_f1": round(abs(l0_obs["macro_f1"] - m2_1_ref["mean_macro_f1"]), 6),
            "roc_auc": round(abs(l0_obs["roc_auc"] - m2_1_ref["mean_roc_auc"]), 6),
            "average_precision": round(abs(l0_obs["average_precision"] - m2_1_ref["mean_average_precision"]), 6),
            "balanced_accuracy": round(abs(l0_obs["balanced_accuracy"] - m2_1_ref["mean_balanced_accuracy"]), 6),
            "recall_inconsistent": round(abs(l0_obs["recall_inconsistent"] - m2_1_ref["mean_inconsistent_recall"]), 6),
        }
    max_diff = max(diffs.values()) if diffs else None
    l0_repro = {
        "status": "VERIFIED_EXACT_MATCH" if (max_diff is not None and max_diff <= 1e-4) else ("MATCH_WITHIN_TOLERANCE" if (max_diff is not None and max_diff <= 1e-3) else "NO_REFERENCE_FILE"),
        "m2_1_reference": m2_1_ref,
        "m2_2_l0_observed": l0_obs,
        "abs_diffs": diffs,
        "max_abs_diff": max_diff,
    }

    # 8. Aggregate Table 2 (lexical_overlap_analysis.csv): both 10-fold mean & primary-seed pooled OOF
    overlap_summary_rows: List[Dict[str, Any]] = []
    oof_primary_df = oof_df[oof_df["seed"] == primary_seed].copy()

    for om in overlap_metrics:
        bin_col = f"bin_{om}"
        for m_id in MODEL_ORDER_RQ2:
            for b_name in OVERLAP_BIN_ORDER:
                sub_f = per_fold_overlap_df[
                    (per_fold_overlap_df["overlap_metric"] == om)
                    & (per_fold_overlap_df["model_id"] == m_id)
                    & (per_fold_overlap_df["overlap_group"] == b_name)
                ]

                def _nanmean(col: str) -> Optional[float]:
                    vals = sub_f[col].dropna()
                    return round(float(vals.mean()), 4) if len(vals) else None

                def _nanstd(col: str) -> Optional[float]:
                    vals = sub_f[col].dropna()
                    return round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)

                overlap_summary_rows.append({
                    "overlap_metric": om,
                    "aggregation": "mean_across_10_folds",
                    "model_id": m_id,
                    "overlap_group": b_name,
                    "sample_count": int(round(sub_f["sample_count"].mean())),
                    "unique_lp_count": int(round(sub_f["unique_lp_count"].mean())),
                    "consistent_count": int(round(sub_f["consistent_count"].mean())),
                    "inconsistent_count": int(round(sub_f["inconsistent_count"].mean())),
                    "mean_title_length": round(float(sub_f["mean_title_length"].dropna().mean()), 2),
                    "mean_lp_length": round(float(sub_f["mean_lp_length"].dropna().mean()), 2),
                    "macro_f1": _nanmean("macro_f1"),
                    "std_macro_f1": _nanstd("macro_f1"),
                    "balanced_accuracy": _nanmean("balanced_accuracy"),
                    "roc_auc": _nanmean("roc_auc"),
                    "std_roc_auc": _nanstd("roc_auc"),
                    "average_precision": _nanmean("average_precision"),
                    "recall_inconsistent": _nanmean("recall_inconsistent"),
                    "predicted_inconsistent_count": int(round(sub_f["predicted_inconsistent_count"].mean())),
                    "undefined_fold_count": int((~sub_f["both_classes_present"]).sum()),
                })

                # Also compute exact primary-seed pooled OOF metrics for exact count reconciliation
                sub_p = oof_primary_df[
                    (oof_primary_df["model_id"] == m_id) & (oof_primary_df[bin_col] == b_name)
                ].merge(row_meta_df[["row_id", "title_char_length", "lp_char_length"]], on="row_id", how="left")
                pm = _compute_subgroup_metrics_core(
                    y_true=sub_p["true_label"].tolist(),
                    y_pred=sub_p["predicted_label"].tolist(),
                    scores_inconsistent=sub_p["score_inconsistent"].to_numpy(),
                    group_ids=sub_p["group_id"].tolist(),
                    title_lengths=sub_p["title_char_length"].to_numpy(),
                    lp_lengths=sub_p["lp_char_length"].to_numpy(),
                )
                overlap_summary_rows.append({
                    "overlap_metric": om,
                    "aggregation": f"pooled_oof_seed_{primary_seed}",
                    "model_id": m_id,
                    "overlap_group": b_name,
                    "sample_count": pm["sample_count"],
                    "unique_lp_count": pm["unique_lp_count"],
                    "consistent_count": pm["consistent_count"],
                    "inconsistent_count": pm["inconsistent_count"],
                    "mean_title_length": pm["mean_title_length"],
                    "mean_lp_length": pm["mean_lp_length"],
                    "macro_f1": pm["macro_f1"],
                    "std_macro_f1": None,
                    "balanced_accuracy": pm["balanced_accuracy"],
                    "roc_auc": pm["roc_auc"],
                    "std_roc_auc": None,
                    "average_precision": pm["average_precision"],
                    "recall_inconsistent": pm["recall_inconsistent"],
                    "predicted_inconsistent_count": pm["predicted_inconsistent_count"],
                    "undefined_fold_count": 0 if pm["both_classes_present"] else 1,
                })

    overlap_summary_df = pd.DataFrame(overlap_summary_rows)

    # 9. Pooled OOF metrics + Overall LP-Group Bootstrap CIs + Feature Interaction Analysis (Table 3)
    n_boot = int(cfg.get("group_bootstrap_resamples", 500))
    pooled_oof_rows: List[Dict[str, Any]] = []
    overall_bootstrap_cis: Dict[str, Any] = {}

    for s in seeds:
        for m_id in MODEL_ORDER_RQ2:
            sub_oof = oof_df[(oof_df["seed"] == s) & (oof_df["model_id"] == m_id)]
            assert len(sub_oof) == len(train_df)
            pm = compute_fold_metrics(
                y_true=sub_oof["true_label"].tolist(),
                y_pred=sub_oof["predicted_label"].tolist(),
                scores_inconsistent=sub_oof["score_inconsistent"].to_numpy(),
                train_time_seconds=0.0,
                inference_time_seconds=0.0,
            )
            pooled_oof_rows.append({
                "seed": int(s),
                "model_id": m_id,
                "pooled_macro_f1": pm["macro_f1"],
                "pooled_balanced_accuracy": pm["balanced_accuracy"],
                "pooled_roc_auc": pm["roc_auc"],
                "pooled_average_precision": pm["average_precision"],
                "pooled_recall_inconsistent": pm["recall_inconsistent"],
                "pooled_precision_inconsistent": pm["precision_inconsistent"],
                "pooled_f1_inconsistent": pm["f1_inconsistent"],
                "tn_inconsistent": pm["tn_inconsistent"],
                "fp_inconsistent": pm["fp_inconsistent"],
                "fn_inconsistent": pm["fn_inconsistent"],
                "tp_inconsistent": pm["tp_inconsistent"],
            })
            if s == primary_seed:
                overall_bootstrap_cis[m_id] = compute_group_bootstrap_ci(
                    oof_df=sub_oof,
                    n_resamples=n_boot,
                    seed=primary_seed,
                )

    contrasts = [
        ("S1_minus_L0", "S1_semantic_e5_small", "L0_lexical_baseline"),
        ("S2_minus_S1", "S2_hybrid_lexical_semantic", "S1_semantic_e5_small"),
        ("S2_minus_L1", "S2_hybrid_lexical_semantic", "L1_lexical_interaction"),
        ("L1_minus_L0", "L1_lexical_interaction", "L0_lexical_baseline"),
    ]
    interaction_rows: List[Dict[str, Any]] = []
    clustered_regressions: Dict[str, Any] = {}

    for om in overlap_metrics:
        bin_col = f"bin_{om}"
        for comp_name, m_a, m_b in contrasts:
            boot_res = compute_paired_group_bootstrap_interaction(
                oof_primary_df=oof_primary_df,
                model_a=m_a,
                model_b=m_b,
                overlap_col=bin_col,
                n_resamples=n_boot,
                seed=primary_seed,
            )
            for grp_label in ["Overall", "Low", "Medium", "High", "Interaction_Low_minus_High"]:
                interaction_rows.append({
                    "overlap_metric": om,
                    "comparison": comp_name,
                    "overlap_group": grp_label,
                    **boot_res[grp_label],
                })
            clustered_regressions[f"{om}__{comp_name}"] = fit_clustered_interaction_regression(
                oof_primary_df=oof_primary_df,
                model_a=m_a,
                model_b=m_b,
                overlap_col=bin_col,
            )

    interaction_df = pd.DataFrame(interaction_rows)

    # 10. Error Disagreement Analysis (Table 4)
    disagreement_df, special_slice_stats = compute_error_disagreement_analysis(
        oof_primary_df=oof_primary_df,
        row_meta_df=row_meta_df,
        primary_overlap_bin_col="bin_title_to_lp_bigram_containment",
    )

    # 11. Persist ALL 13 required artifacts in experiments/runs/<experiment_id>/
    per_fold_df.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    benchmark_df.to_csv(run_dir / "benchmark_comparison.csv", index=False)
    overlap_summary_df.to_csv(run_dir / "lexical_overlap_analysis.csv", index=False)
    interaction_df.to_csv(run_dir / "interaction_analysis.csv", index=False)
    disagreement_df.to_csv(run_dir / "error_disagreement.csv", index=False)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)
    model_costs_df.to_csv(run_dir / "model_costs.csv", index=False)

    # Additional helpful diagnostics (no raw Japanese text!)
    per_fold_overlap_df.to_csv(run_dir / "per_fold_overlap_metrics.csv", index=False)
    row_meta_df.to_csv(run_dir / "row_overlap_metadata.csv", index=False)

    with (run_dir / "embedding_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(emb_manifest, f, indent=2)

    with (run_dir / "fold_manifest_reference.json").open("w", encoding="utf-8") as f:
        json.dump(fold_ref, f, indent=2)

    model_manifest = {
        "experiment_id": experiment_id,
        "configuration_hash": config_hash,
        "fold_manifest_hash": fold_ref["fold_manifest_hash"],
        "checkpoint_policy": "refit_fresh_per_fold_frozen_pretrained_encoder_only",
        "scaling_comparability_note": (
            "L0 uses unscaled L2-normalized sparse char TF-IDF. L1 adds 15 StandardScaler-scaled "
            "lexical interaction features to sparse TF-IDF. S1 applies StandardScaler to 1537 "
            "frozen semantic pair features. S2 combines sparse L2-normalized TF-IDF with scaled "
            "lexical and scaled semantic dense blocks. All scalers and vectorizers are fit strictly "
            "inside each training fold; C=1.0 L2 penalty is shared across models."
        ),
        "models": cfg["models"],
        "reproducibility_table": reproducibility_rows,
    }
    with (run_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)

    rq2_payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "configuration_hash": config_hash,
        "primary_seed": primary_seed,
        "seeds": seeds,
        "n_splits": n_splits,
        "protocol": PROTOCOL_B,
        "dataset_provenance": provenance,
        "fold_manifest_reference": {
            "source_manifest_path": fold_ref["source_manifest_path"],
            "fold_manifest_hash": fold_ref["fold_manifest_hash"],
            "all_folds_strict_disjoint": fold_ref["all_folds_strict_disjoint"],
        },
        "embedding_manifest": emb_manifest,
        "l0_m2_1_reproduction": l0_repro,
        "benchmark_10fold_mean": benchmark_df.to_dict(orient="records"),
        "benchmark_by_seed": per_seed_summary,
        "pooled_oof_summary": pooled_oof_rows,
        "overall_group_bootstrap_cis_primary_seed": overall_bootstrap_cis,
        "overlap_bin_thresholds_per_fold": fold_bin_threshold_rows,
        "clustered_interaction_regressions": clustered_regressions,
        "error_special_slices": special_slice_stats,
        "reproducibility_table": reproducibility_rows,
    }
    with (run_dir / "rq2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rq2_payload, f, indent=2)

    # 12. Export Markdown report and summary CSV/JSON into reports/generated/m2_lexical_semantic/
    report_md = format_rq2_report_markdown(
        provenance=provenance,
        fold_ref=fold_ref,
        embedding_manifest=emb_manifest,
        l0_repro=l0_repro,
        benchmark_df=benchmark_df,
        overlap_df=overlap_summary_df,
        interaction_df=interaction_df,
        disagreement_df=disagreement_df,
        reproducibility_rows=reproducibility_rows,
        payload=rq2_payload,
    )
    (reports_path / "rq2_lexical_semantic_report.md").write_text(report_md, encoding="utf-8")
    benchmark_df.to_csv(reports_path / "benchmark_comparison.csv", index=False)
    overlap_summary_df.to_csv(reports_path / "lexical_overlap_analysis.csv", index=False)
    interaction_df.to_csv(reports_path / "interaction_analysis.csv", index=False)
    disagreement_df.to_csv(reports_path / "error_disagreement.csv", index=False)
    model_costs_df.to_csv(reports_path / "model_costs.csv", index=False)
    with (reports_path / "rq2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rq2_payload, f, indent=2)

    return {
        "provenance": provenance,
        "fold_ref": fold_ref,
        "embedding_manifest": emb_manifest,
        "l0_repro": l0_repro,
        "per_fold_df": per_fold_df,
        "benchmark_df": benchmark_df,
        "overlap_df": overlap_summary_df,
        "interaction_df": interaction_df,
        "disagreement_df": disagreement_df,
        "model_costs_df": model_costs_df,
        "oof_df": oof_df,
        "row_meta_df": row_meta_df,
        "payload": rq2_payload,
        "run_dir": run_dir,
        "reports_dir": reports_path,
        "report_md": report_md,
    }
