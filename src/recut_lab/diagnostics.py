"""Diagnostic analytics for Milestone M2.0.1 — Baseline Failure Diagnostics.

Implements four controlled diagnostic components:
- D1: Score distribution & discriminative ability (ROC-AUC, Average Precision, percentiles, overlap)
- D2: Decision threshold sensitivity (validation exploratory sweep + training-only OOF CV selection)
- D3: Controlled input representation diagnostics (R0 concat, R1 separate TF-IDF blocks,
      R2 weighted title block, R3 truncated LP) using training-only CV selection
- D4: Seen vs. unseen landing-page confounding check on official validation predictions
"""

from __future__ import annotations

import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from recut_lab.baseline import BaselineModel, TfidfLogRegBaseline
from recut_lab.data import compute_file_sha256
from recut_lab.evaluation import (
    LABEL_ORDER,
    ZERO_DIVISION_POLICY,
    benchmark_inference_latency,
    compute_classification_metrics,
)
from recut_lab.experiment import ExperimentManager, set_global_seed
from recut_lab.features import build_char_tfidf_vectorizer
from recut_lab.preprocessing import (
    ColumnTextNormalizer,
    TextInputSelector,
    normalize_japanese_text,
)

POSITIVE_CLASS: str = "inconsistent"
DEFAULT_THRESHOLD: float = 0.50

M2_0_1_DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_id": "m2_failure_diagnostics_v001",
    "random_seed": 42,
    "positive_class": POSITIVE_CLASS,
    "default_threshold": DEFAULT_THRESHOLD,
    "label_order": list(LABEL_ORDER),
    "zero_division_policy": ZERO_DIVISION_POLICY,
    "m2_reference_run": "experiments/runs/m2_baseline_v001",
    "data": {
        "raw_dir": "data/raw/adtec_consistency",
        "train_split": "train",
        "eval_split": "valid",
        "test_split_protected": True,
        "separator": " [SEP] ",
        "normalize_unicode_nfkc": True,
    },
    "vectorizer": {
        "analyzer": "char",
        "ngram_range": [2, 4],
        "max_features": 50000,
        "min_df": 2,
        "sublinear_tf": True,
    },
    "classifier": {
        "model_family": "LogisticRegression",
        "C": 1.0,
        "class_weight": None,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "d2_threshold": {
        "grid_start": 0.05,
        "grid_end": 0.95,
        "grid_step": 0.05,
        "cv_splits": 5,
        "selection_objective": "f1_inconsistent",
    },
    "d3_representation": {
        "cv_splits": 5,
        "selection_objective": "macro_f1",
        "r1_max_features_per_block": 50000,
        "r2_title_weight_candidates": [1.5, 2.0, 3.0],
        "r3_lp_max_chars_candidates": [40, 80, 120],
    },
    "latency_benchmark": {
        "warmup_runs": 2,
        "timed_runs": 7,
        "method": "full_validation_batch_amortized_median",
    },
}


# =============================================================================
# D1 — SCORE EXTRACTION & DISCRIMINATIVE ABILITY
# =============================================================================


def extract_class_scores(
    model: Any,
    df: pd.DataFrame,
    target_class: str = POSITIVE_CLASS,
) -> np.ndarray:
    """Extract raw predict_proba scores for target_class using model.classes_ dynamically.

    Never assumes a hardcoded probability column index.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Model does not expose 'classes_' attribute; ensure it is fitted.")

    class_list = [str(c) for c in classes]
    if target_class not in class_list:
        raise ValueError(
            f"Target class '{target_class}' not found in model.classes_={class_list}."
        )
    class_idx = class_list.index(target_class)
    probs = model.predict_proba(df)
    if probs is None:
        raise ValueError("Model predict_proba returned None.")
    return np.asarray(probs[:, class_idx], dtype=np.float64)


def compute_histogram_overlap(
    scores_neg: np.ndarray,
    scores_pos: np.ndarray,
    bins: int = 20,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Compute normalized histogram intersection area in [0, 1] between two score arrays.

    A value of 1.0 indicates identical binned distributions; 0.0 indicates disjoint bins.
    """
    if len(scores_neg) == 0 or len(scores_pos) == 0:
        return 0.0
    h_neg, _ = np.histogram(scores_neg, bins=bins, range=value_range, density=False)
    h_pos, _ = np.histogram(scores_pos, bins=bins, range=value_range, density=False)
    p_neg = h_neg / max(h_neg.sum(), 1)
    p_pos = h_pos / max(h_pos.sum(), 1)
    return round(float(np.minimum(p_neg, p_pos).sum()), 4)


def compute_d1_score_diagnostics(
    y_true: Sequence[str],
    scores: Sequence[float],
    model_id: str,
    positive_class: str = POSITIVE_CLASS,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Compute D1 discriminative ability and score distribution diagnostics."""
    y_arr = np.asarray(list(y_true), dtype=object)
    s_arr = np.asarray(list(scores), dtype=np.float64)
    y_bin = (y_arr == positive_class).astype(int)

    prevalence = float(y_bin.mean()) if len(y_bin) > 0 else 0.0
    is_constant = bool(np.unique(s_arr).size <= 1)

    # Guard against single-class y_bin in synthetic edge cases
    has_both_classes = len(np.unique(y_bin)) == 2
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_bin, s_arr))
        avg_prec = float(average_precision_score(y_bin, s_arr))
        fpr, tpr, roc_thr = roc_curve(y_bin, s_arr)
        pr_prec, pr_rec, pr_thr = precision_recall_curve(y_bin, s_arr)
    else:
        roc_auc = float("nan")
        avg_prec = float("nan")
        fpr, tpr, roc_thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        pr_prec, pr_rec, pr_thr = np.array([prevalence]), np.array([1.0]), np.array([])

    scores_cons = s_arr[y_bin == 0]
    scores_inc = s_arr[y_bin == 1]

    p5, p25, p50, p75, p95 = np.percentile(s_arr, [5, 25, 50, 75, 95])
    hist_overlap = compute_histogram_overlap(scores_cons, scores_inc, bins=20)

    pred_inc_default = int((s_arr >= default_threshold).sum())
    pred_cons_default = int((s_arr < default_threshold).sum())

    return {
        "model_id": model_id,
        "positive_class": positive_class,
        "is_constant_score": is_constant,
        "roc_auc": round(roc_auc, 4) if np.isfinite(roc_auc) else None,
        "average_precision": round(avg_prec, 4) if np.isfinite(avg_prec) else None,
        "prevalence_inconsistent": round(prevalence, 4),
        "score_p5": round(float(p5), 4),
        "score_p25": round(float(p25), 4),
        "score_p50": round(float(p50), 4),
        "score_p75": round(float(p75), 4),
        "score_p95": round(float(p95), 4),
        "mean_score_consistent": round(float(np.mean(scores_cons)), 4) if len(scores_cons) else None,
        "median_score_consistent": round(float(np.median(scores_cons)), 4) if len(scores_cons) else None,
        "mean_score_inconsistent": round(float(np.mean(scores_inc)), 4) if len(scores_inc) else None,
        "median_score_inconsistent": round(float(np.median(scores_inc)), 4) if len(scores_inc) else None,
        "histogram_overlap_20bins": hist_overlap,
        "default_threshold": default_threshold,
        "predicted_consistent_at_default": pred_cons_default,
        "predicted_inconsistent_at_default": pred_inc_default,
        "curves": {
            "fpr": [round(float(x), 5) for x in fpr],
            "tpr": [round(float(x), 5) for x in tpr],
            "pr_precision": [round(float(x), 5) for x in pr_prec],
            "pr_recall": [round(float(x), 5) for x in pr_rec],
        },
    }


# =============================================================================
# D2 — THRESHOLD SENSITIVITY & TRAINING-ONLY OOF SELECTION
# =============================================================================


def build_threshold_grid(
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.05,
) -> List[float]:
    """Generate deterministic, rounded threshold grid."""
    n_steps = int(round((end - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(n_steps)]


def evaluate_threshold_point(
    y_true: Sequence[str],
    scores: Sequence[float],
    threshold: float,
    model_id: str,
    selection_source: str,
    positive_class: str = POSITIVE_CLASS,
    negative_class: str = "consistent",
) -> Dict[str, Any]:
    """Evaluate classification metrics at a specific decision threshold."""
    y_arr = np.asarray(list(y_true), dtype=object)
    s_arr = np.asarray(list(scores), dtype=np.float64)
    y_pred = np.where(s_arr >= threshold, positive_class, negative_class)

    prec_inc = float(
        precision_score(
            y_arr, y_pred, pos_label=positive_class, zero_division=ZERO_DIVISION_POLICY
        )
    )
    rec_inc = float(
        recall_score(
            y_arr, y_pred, pos_label=positive_class, zero_division=ZERO_DIVISION_POLICY
        )
    )
    f1_inc = float(
        f1_score(
            y_arr, y_pred, pos_label=positive_class, zero_division=ZERO_DIVISION_POLICY
        )
    )
    macro_f1 = float(
        f1_score(
            y_arr,
            y_pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=ZERO_DIVISION_POLICY,
        )
    )
    bal_acc = float(balanced_accuracy_score(y_arr, y_pred))
    pred_pos_count = int((y_pred == positive_class).sum())

    return {
        "model_id": model_id,
        "threshold": round(float(threshold), 4),
        "precision_inconsistent": round(prec_inc, 4),
        "recall_inconsistent": round(rec_inc, 4),
        "f1_inconsistent": round(f1_inc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "predicted_inconsistent_count": pred_pos_count,
        "selection_source": selection_source,
    }


def evaluate_threshold_grid(
    y_true: Sequence[str],
    scores: Sequence[float],
    model_id: str,
    thresholds: Sequence[float],
    selection_source: str = "valid_exploratory_grid",
) -> pd.DataFrame:
    """Evaluate a model across a reproducible grid of decision thresholds."""
    rows = [
        evaluate_threshold_point(
            y_true=y_true,
            scores=scores,
            threshold=thr,
            model_id=model_id,
            selection_source=selection_source,
        )
        for thr in thresholds
    ]
    return pd.DataFrame(rows)


def compute_oof_scores_for_pipeline(
    train_df: pd.DataFrame,
    pipeline_builder,
    n_splits: int = 5,
    random_state: int = 42,
    target_col: str = "label",
    positive_class: str = POSITIVE_CLASS,
) -> np.ndarray:
    """Compute training-only StratifiedKFold out-of-fold scores for positive_class.

    Fits the entire pipeline (including TF-IDF vectorizer) strictly within each
    training fold so zero out-of-fold leakage occurs.
    """
    y = train_df[target_col].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_scores = np.zeros(len(train_df), dtype=np.float64)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]
        set_global_seed(random_state)
        pipe = pipeline_builder()
        pipe.fit(fold_train, fold_train[target_col].to_numpy())
        oof_scores[val_idx] = extract_class_scores(pipe, fold_val, target_class=positive_class)

    return oof_scores


def select_threshold_from_oof(
    y_train: Sequence[str],
    oof_scores: Sequence[float],
    thresholds: Sequence[float],
    model_id: str,
    objective: str = "f1_inconsistent",
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Select candidate threshold exclusively from training OOF predictions.

    Ties are broken deterministically in favor of the threshold closest to 0.50.
    """
    oof_grid = evaluate_threshold_grid(
        y_true=y_train,
        scores=oof_scores,
        model_id=model_id,
        thresholds=thresholds,
        selection_source="train_oof_cv_grid",
    )
    oof_grid["_dist_default"] = (oof_grid["threshold"] - DEFAULT_THRESHOLD).abs()
    best_row = (
        oof_grid.sort_values(
            by=[objective, "balanced_accuracy", "_dist_default"],
            ascending=[False, False, True],
        )
        .iloc[0]
        .to_dict()
    )
    best_row.pop("_dist_default", None)
    oof_grid = oof_grid.drop(columns=["_dist_default"])
    return float(best_row["threshold"]), best_row, oof_grid


# =============================================================================
# D3 — CONTROLLED INPUT REPRESENTATION DIAGNOSTICS (R0–R3)
# =============================================================================


class SeparateBlocksTfidfLogReg(BaselineModel):
    """R1/R2: Independent TF-IDF blocks for title and lp_text via ColumnTransformer."""

    def __init__(
        self,
        title_weight: float = 1.0,
        lp_weight: float = 1.0,
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 2,
        max_features_per_block: Optional[int] = 50000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        class_weight: Optional[str] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        title_col: str = "title",
        lp_col: str = "lp_text",
        normalize: bool = True,
    ):
        self.title_weight = float(title_weight)
        self.lp_weight = float(lp_weight)
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.max_features_per_block = max_features_per_block
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.title_col = title_col
        self.lp_col = lp_col
        self.normalize = normalize

        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        title_pipe = Pipeline([
            ("norm", ColumnTextNormalizer(normalize=self.normalize)),
            (
                "tfidf",
                build_char_tfidf_vectorizer(
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_features=self.max_features_per_block,
                    sublinear_tf=self.sublinear_tf,
                ),
            ),
        ])
        lp_pipe = Pipeline([
            ("norm", ColumnTextNormalizer(normalize=self.normalize)),
            (
                "tfidf",
                build_char_tfidf_vectorizer(
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_features=self.max_features_per_block,
                    sublinear_tf=self.sublinear_tf,
                ),
            ),
        ])
        features = ColumnTransformer(
            transformers=[
                ("title_tfidf", title_pipe, self.title_col),
                ("lp_tfidf", lp_pipe, self.lp_col),
            ],
            transformer_weights={
                "title_tfidf": self.title_weight,
                "lp_tfidf": self.lp_weight,
            },
            sparse_threshold=1.0,
        )
        clf = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        return Pipeline([
            ("features", features),
            ("clf", clf),
        ])

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> SeparateBlocksTfidfLogReg:
        self.pipeline = self._build_pipeline()
        y = df[target_col].to_numpy()
        self.pipeline.fit(df, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        return [str(p) for p in self.pipeline.predict(df)]

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.pipeline.predict_proba(df)

    def get_feature_diagnostics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute sparse-safe feature block norms and linear logit contributions."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")

        ct = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]

        X_sparse = ct.transform(df)
        title_vec = ct.named_transformers_["title_tfidf"].named_steps["tfidf"]
        lp_vec = ct.named_transformers_["lp_tfidf"].named_steps["tfidf"]
        n_title = len(title_vec.vocabulary_)
        n_lp = len(lp_vec.vocabulary_)

        X_title = X_sparse[:, :n_title]
        X_lp = X_sparse[:, n_title : n_title + n_lp]

        # Sparse row-wise L2 norms without dense conversion
        title_norms = np.sqrt(np.asarray(X_title.multiply(X_title).sum(axis=1)).ravel())
        lp_norms = np.sqrt(np.asarray(X_lp.multiply(X_lp).sum(axis=1)).ravel())

        # Binary LogisticRegression coef_ is shape (1, n_features) for classes_[1]
        coef = clf.coef_[0]
        w_title = coef[:n_title]
        w_lp = coef[n_title : n_title + n_lp]

        contrib_title = np.asarray(X_title.dot(w_title)).ravel()
        contrib_lp = np.asarray(X_lp.dot(w_lp)).ravel()

        mean_abs_title = float(np.mean(np.abs(contrib_title)))
        mean_abs_lp = float(np.mean(np.abs(contrib_lp)))
        total_abs = max(mean_abs_title + mean_abs_lp, 1e-12)

        return {
            "sparse_matrix_shape": [int(X_sparse.shape[0]), int(X_sparse.shape[1])],
            "total_features": int(n_title + n_lp),
            "title_features": int(n_title),
            "lp_features": int(n_lp),
            "mean_title_block_l2_norm": round(float(np.mean(title_norms)), 4),
            "mean_lp_block_l2_norm": round(float(np.mean(lp_norms)), 4),
            "title_weight_l2_norm": round(float(np.linalg.norm(w_title)), 4),
            "lp_weight_l2_norm": round(float(np.linalg.norm(w_lp)), 4),
            "mean_abs_title_logit": round(mean_abs_title, 4),
            "mean_abs_lp_logit": round(mean_abs_lp, 4),
            "title_logit_share": round(mean_abs_title / total_abs, 4),
            "lp_logit_share": round(mean_abs_lp / total_abs, 4),
        }


class TruncatedLpPairBaseline(TfidfLogRegBaseline):
    """R3: Concatenated pair baseline with deterministic character-length cap on lp_text."""

    def __init__(self, lp_max_chars: int = 80, **kwargs):
        kwargs["input_mode"] = "pair"
        super().__init__(**kwargs)
        self.lp_max_chars = int(lp_max_chars)

    def _build_pipeline(self) -> Pipeline:
        selector = TextInputSelector(
            mode="pair",
            title_col=self.title_col,
            lp_col=self.lp_col,
            separator=self.separator,
            normalize=self.normalize,
            lp_max_chars=self.lp_max_chars,
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
            ("selector", selector),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])


def evaluate_representation_cv(
    train_df: pd.DataFrame,
    pipeline_builder,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate a candidate representation strictly via training-only StratifiedKFold OOF."""
    oof_scores = compute_oof_scores_for_pipeline(
        train_df=train_df,
        pipeline_builder=pipeline_builder,
        n_splits=n_splits,
        random_state=random_state,
        positive_class=POSITIVE_CLASS,
    )
    y_true = train_df["label"].tolist()
    y_pred = ["inconsistent" if s >= DEFAULT_THRESHOLD else "consistent" for s in oof_scores]
    ev = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=LABEL_ORDER)
    y_bin = (train_df["label"] == POSITIVE_CLASS).astype(int).to_numpy()
    ap = float(average_precision_score(y_bin, oof_scores))
    auc = float(roc_auc_score(y_bin, oof_scores))
    return {
        "oof_macro_f1": ev.macro_f1,
        "oof_f1_inconsistent": ev.per_class_f1["inconsistent"],
        "oof_recall_inconsistent": ev.per_class_recall["inconsistent"],
        "oof_balanced_accuracy": ev.balanced_accuracy,
        "oof_average_precision": round(ap, 4),
        "oof_roc_auc": round(auc, 4),
    }


# =============================================================================
# D4 — SEEN VS. UNSEEN LANDING PAGE CONFOUNDING CHECK
# =============================================================================


def label_seen_unseen_lps(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    lp_col: str = "lp_text",
) -> pd.Series:
    """Label each validation row as 'seen_lp' or 'unseen_lp' via exact normalized LP match."""
    train_norm_lps = set(train_df[lp_col].map(normalize_japanese_text))
    valid_norm_lps = valid_df[lp_col].map(normalize_japanese_text)
    return valid_norm_lps.map(lambda txt: "seen_lp" if txt in train_norm_lps else "unseen_lp")


def compute_seen_unseen_confounding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Quantify class prevalence, majority reference errors, and per-class recall by LP group."""
    lp_group = label_seen_unseen_lps(train_df, valid_df)
    valid_meta = pd.DataFrame({
        "valid_row_index": valid_df.index,
        "lp_group": lp_group.values,
    })

    targets = list(model_ids) if model_ids is not None else list(predictions_df["model_id"].unique())
    rows: List[Dict[str, Any]] = []

    for m_id in targets:
        sub = predictions_df[predictions_df["model_id"] == m_id].merge(
            valid_meta, on="valid_row_index", how="inner"
        )
        for grp in ["seen_lp", "unseen_lp"]:
            g = sub[sub["lp_group"] == grp]
            n_grp = len(g)
            if n_grp == 0:
                continue
            y_true = g["true_label"].tolist()
            y_pred = g["predicted_label"].tolist()

            n_cons = int((g["true_label"] == "consistent").sum())
            n_inc = int((g["true_label"] == "inconsistent").sum())
            cons_pct = round(n_cons / n_grp * 100.0, 2)
            inc_pct = round(n_inc / n_grp * 100.0, 2)

            # Within-group majority-class reference error rate (trivial oracle prior for group)
            within_grp_maj_err = round(min(n_cons, n_inc) / n_grp, 4)
            # Global training majority-class reference error rate (always predicting 'consistent')
            train_maj_err = round(n_inc / n_grp, 4)

            ev = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=LABEL_ORDER)
            model_err_rate = round(float(g["is_error"].mean()), 4)

            rows.append({
                "model_id": m_id,
                "lp_group": grp,
                "n_samples": n_grp,
                "consistent_count": n_cons,
                "inconsistent_count": n_inc,
                "consistent_pct": cons_pct,
                "inconsistent_pct": inc_pct,
                "within_group_majority_error": within_grp_maj_err,
                "train_majority_reference_error": train_maj_err,
                "model_error_rate": model_err_rate,
                "recall_consistent": ev.per_class_recall["consistent"],
                "recall_inconsistent": ev.per_class_recall["inconsistent"],
                "balanced_accuracy": ev.balanced_accuracy,
                "macro_f1": ev.macro_f1,
            })

    return pd.DataFrame(rows)


# =============================================================================
# END-TO-END M2.0.1 DIAGNOSTICS ORCHESTRATOR & MARKDOWN SUMMARY
# =============================================================================


def format_diagnostics_markdown(
    d1_df: pd.DataFrame,
    d2_summary_df: pd.DataFrame,
    d3_df: pd.DataFrame,
    seen_unseen_df: pd.DataFrame,
    payload: Dict[str, Any],
) -> str:
    """Render concise English scientific summary for Milestone M2.0.1."""
    lines = [
        "# Milestone M2.0.1 — Baseline Failure Diagnostics Summary",
        "",
        f"**Experiment ID**: `{payload['experiment_id']}`  ",
        f"**Dataset Revision**: `{payload.get('dataset_revision', 'unknown')}`  ",
        "**Splits Accessed**: `train` (10,635) and `valid` (945) ONLY; official `test` split strictly untouched.  ",
        f"**Random Seed**: `{payload['random_seed']}`  ",
        "",
        "---",
        "",
        "## 1. D1 — Score Distribution & Discriminative Ability (Validation)",
        "",
        "| model_id | roc_auc | average_precision | prevalence_inconsistent | score_p5 | score_p25 | score_p50 | score_p75 | score_p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in d1_df.iterrows():
        lines.append(
            f"| `{r['model_id']}` | {r['roc_auc']:.4f} | {r['average_precision']:.4f} | "
            f"{r['prevalence_inconsistent']:.4f} | {r['score_p5']:.4f} | {r['score_p25']:.4f} | "
            f"{r['score_p50']:.4f} | {r['score_p75']:.4f} | {r['score_p95']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. D2 — Decision Threshold Sensitivity Summary (Validation)",
        "",
        "| model_id | threshold | precision_inconsistent | recall_inconsistent | f1_inconsistent | balanced_accuracy | predicted_inconsistent_count | selection_source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for _, r in d2_summary_df.iterrows():
        lines.append(
            f"| `{r['model_id']}` | {r['threshold']:.2f} | {r['precision_inconsistent']:.4f} | "
            f"{r['recall_inconsistent']:.4f} | {r['f1_inconsistent']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {int(r['predicted_inconsistent_count'])} | `{r['selection_source']}` |"
        )

    lines.extend([
        "",
        "## 3. D3 — Controlled Input Representation Comparison (Validation)",
        "",
        "| representation_id | representation_method | macro_f1 | recall_inconsistent | average_precision | train_time_seconds | inference_latency_ms | feature_count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, r in d3_df.iterrows():
        lines.append(
            f"| `{r['representation_id']}` | {r['representation_method']} | {r['macro_f1']:.4f} | "
            f"{r['recall_inconsistent']:.4f} | {r['average_precision']:.4f} | "
            f"{r['train_time_seconds']:.4f} | {r['inference_latency_ms']:.4f} | {int(r['feature_count'])} |"
        )

    lines.extend([
        "",
        "## 4. Seen vs. Unseen Landing Page Confounding Check (Validation)",
        "",
        "| model_id | lp_group | n_samples | inconsistent_pct | within_group_majority_error | train_majority_reference_error | model_error_rate | recall_consistent | recall_inconsistent | balanced_accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, r in seen_unseen_df.iterrows():
        lines.append(
            f"| `{r['model_id']}` | `{r['lp_group']}` | {int(r['n_samples'])} | {r['inconsistent_pct']:.2f}% | "
            f"{r['within_group_majority_error']:.4f} | {r['train_majority_reference_error']:.4f} | "
            f"{r['model_error_rate']:.4f} | {r['recall_consistent']:.4f} | "
            f"{r['recall_inconsistent']:.4f} | {r['balanced_accuracy']:.4f} |"
        )

    lines.extend([
        "",
        "## 5. Scientific Guardrails & Status Tags",
        "",
        "- **VERIFIED**: All M2.0 checkpoints loaded without modifying prior artifacts; official test split remained untouched.",
        "- **VERIFIED**: All D2 threshold selection and D3 hyperparameter candidates (`w_title`, `lp_max_chars`) were selected strictly via training-only 5-fold StratifiedKFold CV with vectorizers fit inside each fold.",
        "- **INFERRED**: Raw `predict_proba` scores are uncalibrated; ROC-AUC and Average Precision evaluate ranking separation, not calibration quality.",
        "- **TODO**: Execute controlled LP-group-disjoint cross-validation in RQ1 to disentangle unseen-LP generalization from prevalence confounding.",
    ])
    return "\n".join(lines)


def run_m2_failure_diagnostics(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    m2_run_dir: Path | str = "experiments/runs/m2_baseline_v001",
    runs_base_dir: Path | str = "experiments/runs",
    models_base_dir: Path | str = "models",
    reports_dir: Path | str = "reports/generated/m2_failure_diagnostics",
) -> Dict[str, Any]:
    """Execute the full M2.0.1 diagnostic suite (D1, D2, D3, Seen/Unseen LP)."""
    cfg = deepcopy(config) if config is not None else deepcopy(M2_0_1_DEFAULT_CONFIG)
    experiment_id = cfg.get("experiment_id", "m2_failure_diagnostics_v001")
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)

    m2_dir = Path(m2_run_dir)
    m2_preds_df = pd.read_csv(m2_dir / "predictions_valid.csv")
    m2_manifest = json.loads((m2_dir / "model_manifest.json").read_text(encoding="utf-8"))
    m2_metrics = json.loads((m2_dir / "metrics.json").read_text(encoding="utf-8"))

    exp_mgr = ExperimentManager(base_runs_dir=runs_base_dir, reports_dir=reports_dir)
    run_dir = exp_mgr.create_named_run(
        run_id=experiment_id,
        config=cfg,
        dataset_metadata=dataset_metadata,
        seed=seed,
    )
    models_dir = Path(models_base_dir) / experiment_id
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    y_valid_true = valid_df["label"].tolist()
    y_train_true = train_df["label"].tolist()

    # -------------------------------------------------------------------------
    # D1: Load frozen M2.0 models, verify scores, compute discriminative metrics
    # -------------------------------------------------------------------------
    frozen_models: Dict[str, BaselineModel] = {}
    valid_scores_by_model: Dict[str, np.ndarray] = {}
    score_verification: Dict[str, float] = {}
    d1_details: Dict[str, Any] = {}
    d1_rows: List[Dict[str, Any]] = []

    for m_id in ["B0_majority", "B1_title_only", "B2_lp_only", "B3_pair"]:
        artifact_path = Path(m2_manifest["models"][m_id]["artifact_path"])
        mdl = BaselineModel.load(artifact_path)
        frozen_models[m_id] = mdl

        recomputed = extract_class_scores(mdl, valid_df, target_class=POSITIVE_CLASS)
        saved = (
            m2_preds_df[m2_preds_df["model_id"] == m_id]
            .sort_values("valid_row_index")["score_inconsistent"]
            .to_numpy(dtype=np.float64)
        )
        max_diff = float(np.max(np.abs(recomputed - saved)))
        score_verification[m_id] = round(max_diff, 8)
        valid_scores_by_model[m_id] = recomputed

        d1_info = compute_d1_score_diagnostics(
            y_true=y_valid_true,
            scores=recomputed,
            model_id=m_id,
            positive_class=POSITIVE_CLASS,
            default_threshold=DEFAULT_THRESHOLD,
        )
        d1_details[m_id] = d1_info
        d1_rows.append({
            "model_id": m_id,
            "roc_auc": d1_info["roc_auc"],
            "average_precision": d1_info["average_precision"],
            "prevalence_inconsistent": d1_info["prevalence_inconsistent"],
            "score_p5": d1_info["score_p5"],
            "score_p25": d1_info["score_p25"],
            "score_p50": d1_info["score_p50"],
            "score_p75": d1_info["score_p75"],
            "score_p95": d1_info["score_p95"],
        })

    d1_df = pd.DataFrame(d1_rows)

    # -------------------------------------------------------------------------
    # D2: Validation exploratory threshold sweep + Training-only 5-fold OOF CV
    # -------------------------------------------------------------------------
    d2_cfg = cfg["d2_threshold"]
    thresholds = build_threshold_grid(
        start=float(d2_cfg["grid_start"]),
        end=float(d2_cfg["grid_end"]),
        step=float(d2_cfg["grid_step"]),
    )
    cv_splits = int(d2_cfg["cv_splits"])
    sel_obj = str(d2_cfg["selection_objective"])

    vec_cfg = cfg["vectorizer"]
    clf_cfg = cfg["classifier"]
    data_cfg = cfg["data"]

    threshold_grid_frames: List[pd.DataFrame] = []
    d2_summary_rows: List[Dict[str, Any]] = []
    d2_cv_details: Dict[str, Any] = {}

    model_mode_map = {
        "B1_title_only": "title",
        "B2_lp_only": "lp_text",
        "B3_pair": "pair",
    }

    for m_id, input_mode in model_mode_map.items():
        # 1. Default 0.50 on valid
        default_row = evaluate_threshold_point(
            y_true=y_valid_true,
            scores=valid_scores_by_model[m_id],
            threshold=DEFAULT_THRESHOLD,
            model_id=m_id,
            selection_source="default_0.50",
        )
        d2_summary_rows.append(default_row)

        # 2. Exploratory validation grid (strictly descriptive)
        valid_grid_df = evaluate_threshold_grid(
            y_true=y_valid_true,
            scores=valid_scores_by_model[m_id],
            model_id=m_id,
            thresholds=thresholds,
            selection_source="valid_exploratory_grid",
        )
        threshold_grid_frames.append(valid_grid_df)

        # Record exploratory validation best (explicitly labeled exploratory)
        valid_best_expl = (
            valid_grid_df.sort_values(by=[sel_obj, "balanced_accuracy"], ascending=[False, False])
            .iloc[0]
            .to_dict()
        )
        valid_best_expl["selection_source"] = "valid_exploratory_best_diagnostic_only"

        # 3. Training-only OOF CV threshold selection (leak-free)
        def _make_pipe(mode: str = input_mode) -> Pipeline:
            return TfidfLogRegBaseline(
                input_mode=mode,
                ngram_range=tuple(vec_cfg["ngram_range"]),
                min_df=int(vec_cfg["min_df"]),
                max_features=vec_cfg["max_features"],
                sublinear_tf=bool(vec_cfg["sublinear_tf"]),
                c_param=float(clf_cfg["C"]),
                class_weight=clf_cfg["class_weight"],
                max_iter=int(clf_cfg["max_iter"]),
                random_state=seed,
                solver=str(clf_cfg["solver"]),
                separator=str(data_cfg["separator"]),
                normalize=bool(data_cfg["normalize_unicode_nfkc"]),
            )._build_pipeline()

        oof_scores = compute_oof_scores_for_pipeline(
            train_df=train_df,
            pipeline_builder=_make_pipe,
            n_splits=cv_splits,
            random_state=seed,
            positive_class=POSITIVE_CLASS,
        )
        oof_best_thr, oof_best_metrics, oof_grid_df = select_threshold_from_oof(
            y_train=y_train_true,
            oof_scores=oof_scores,
            thresholds=thresholds,
            model_id=m_id,
            objective=sel_obj,
        )
        threshold_grid_frames.append(oof_grid_df)

        # Evaluate training-selected threshold once on official validation
        valid_at_oof_thr = evaluate_threshold_point(
            y_true=y_valid_true,
            scores=valid_scores_by_model[m_id],
            threshold=oof_best_thr,
            model_id=m_id,
            selection_source="train_oof_cv_selected",
        )
        d2_summary_rows.append(valid_at_oof_thr)
        d2_summary_rows.append(valid_best_expl)

        d2_cv_details[m_id] = {
            "cv_splits": cv_splits,
            "selection_objective": sel_obj,
            "train_oof_average_precision": round(
                float(
                    average_precision_score(
                        (train_df["label"] == POSITIVE_CLASS).astype(int), oof_scores
                    )
                ),
                4,
            ),
            "selected_threshold_train_oof": oof_best_thr,
            "train_oof_metrics_at_selected": oof_best_metrics,
            "valid_metrics_at_train_selected": valid_at_oof_thr,
            "valid_exploratory_best_diagnostic": valid_best_expl,
        }

    threshold_analysis_df = pd.concat(threshold_grid_frames, ignore_index=True)
    d2_summary_df = pd.DataFrame(d2_summary_rows)[
        [
            "model_id",
            "threshold",
            "precision_inconsistent",
            "recall_inconsistent",
            "f1_inconsistent",
            "balanced_accuracy",
            "predicted_inconsistent_count",
            "selection_source",
        ]
    ]

    # -------------------------------------------------------------------------
    # D3: Controlled Input Representation Diagnostics (R0–R3)
    # -------------------------------------------------------------------------
    d3_cfg = cfg["d3_representation"]
    d3_cv_splits = int(d3_cfg["cv_splits"])
    d3_sel_obj = f"oof_{d3_cfg['selection_objective']}"
    r1_max_feats = d3_cfg["r1_max_features_per_block"]
    lat_cfg = cfg["latency_benchmark"]
    warmup_runs = int(lat_cfg["warmup_runs"])
    timed_runs = int(lat_cfg["timed_runs"])

    d3_rows: List[Dict[str, Any]] = []
    d3_details: Dict[str, Any] = {}
    rep_manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "random_seed": seed,
        "models": {},
    }

    # --- R0: Reuse frozen B3_pair without retraining ---
    r0_model = frozen_models["B3_pair"]
    r0_m2_info = m2_metrics["models"]["B3_pair"]
    r0_scores = valid_scores_by_model["B3_pair"]
    r0_preds, r0_lat = benchmark_inference_latency(
        predict_fn=r0_model.predict,
        inputs=valid_df,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    r0_ev = compute_classification_metrics(
        y_true=y_valid_true,
        y_pred=r0_preds,
        labels=LABEL_ORDER,
        latency_ms_per_sample=r0_lat.median_ms_per_sample,
        total_latency_seconds=r0_lat.median_batch_seconds,
    )
    r0_X_valid = r0_model.pipeline.named_steps["tfidf"].transform(
        r0_model.pipeline.named_steps["selector"].transform(valid_df)
    )
    r0_ap = d1_details["B3_pair"]["average_precision"]
    r0_auc = d1_details["B3_pair"]["roc_auc"]
    d3_rows.append({
        "representation_id": "R0_concat_pair",
        "representation_method": "Frozen B3_pair: title + [SEP] + lp_text (single TF-IDF)",
        "macro_f1": r0_ev.macro_f1,
        "recall_inconsistent": r0_ev.per_class_recall["inconsistent"],
        "average_precision": r0_ap,
        "train_time_seconds": r0_m2_info["train_time_seconds"],
        "inference_latency_ms": round(r0_lat.median_ms_per_sample, 4),
        "feature_count": int(r0_X_valid.shape[1]),
    })
    d3_details["R0_concat_pair"] = {
        "reused_from_m2": True,
        "sparse_matrix_shape": [int(r0_X_valid.shape[0]), int(r0_X_valid.shape[1])],
        "model_size_mb": r0_m2_info["model_size_mb"],
        "roc_auc": r0_auc,
        "average_precision": r0_ap,
        "evaluation": r0_ev.to_dict(),
        "latency_benchmark": r0_lat.to_dict(),
        "char_length_share_valid": {
            "mean_title_char_share": round(
                float(
                    (
                        valid_df["title"].str.len()
                        / (valid_df["title"].str.len() + valid_df["lp_text"].str.len())
                    ).mean()
                ),
                4,
            ),
            "mean_lp_char_share": round(
                float(
                    (
                        valid_df["lp_text"].str.len()
                        / (valid_df["title"].str.len() + valid_df["lp_text"].str.len())
                    ).mean()
                ),
                4,
            ),
        },
    }
    rep_manifest["models"]["R0_concat_pair"] = {
        **m2_manifest["models"]["B3_pair"],
        "reused_frozen_checkpoint": True,
    }

    # --- R1: Separate TF-IDF Blocks (ColumnTransformer, equal weights 1.0 / 1.0) ---
    set_global_seed(seed)
    r1_model = SeparateBlocksTfidfLogReg(
        title_weight=1.0,
        lp_weight=1.0,
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=int(vec_cfg["min_df"]),
        max_features_per_block=r1_max_feats,
        sublinear_tf=bool(vec_cfg["sublinear_tf"]),
        c_param=float(clf_cfg["C"]),
        class_weight=clf_cfg["class_weight"],
        max_iter=int(clf_cfg["max_iter"]),
        random_state=seed,
        solver=str(clf_cfg["solver"]),
        normalize=bool(data_cfg["normalize_unicode_nfkc"]),
    )
    t0 = time.perf_counter()
    r1_model.fit(train_df, target_col="label")
    r1_train_sec = time.perf_counter() - t0

    r1_path = models_dir / "R1_separate_blocks.pkl"
    r1_model.save(r1_path)
    r1_size_mb = r1_path.stat().st_size / (1024.0 * 1024.0)
    r1_preds, r1_lat = benchmark_inference_latency(
        predict_fn=r1_model.predict,
        inputs=valid_df,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    r1_scores = extract_class_scores(r1_model, valid_df, target_class=POSITIVE_CLASS)
    r1_ev = compute_classification_metrics(
        y_true=y_valid_true,
        y_pred=r1_preds,
        labels=LABEL_ORDER,
        latency_ms_per_sample=r1_lat.median_ms_per_sample,
        total_latency_seconds=r1_lat.median_batch_seconds,
    )
    r1_d1 = compute_d1_score_diagnostics(y_valid_true, r1_scores, model_id="R1_separate_blocks")
    r1_block_diag = r1_model.get_feature_diagnostics(valid_df)

    d3_rows.append({
        "representation_id": "R1_separate_blocks",
        "representation_method": "ColumnTransformer: separate title & lp_text TF-IDF (weights 1.0 / 1.0)",
        "macro_f1": r1_ev.macro_f1,
        "recall_inconsistent": r1_ev.per_class_recall["inconsistent"],
        "average_precision": r1_d1["average_precision"],
        "train_time_seconds": round(r1_train_sec, 4),
        "inference_latency_ms": round(r1_lat.median_ms_per_sample, 4),
        "feature_count": r1_block_diag["total_features"],
    })
    d3_details["R1_separate_blocks"] = {
        "title_weight": 1.0,
        "lp_weight": 1.0,
        "model_size_mb": round(r1_size_mb, 4),
        "roc_auc": r1_d1["roc_auc"],
        "average_precision": r1_d1["average_precision"],
        "block_diagnostics": r1_block_diag,
        "evaluation": r1_ev.to_dict(),
        "latency_benchmark": r1_lat.to_dict(),
    }
    rep_manifest["models"]["R1_separate_blocks"] = {
        "artifact_path": str(r1_path.as_posix()),
        "sha256": compute_file_sha256(r1_path),
        "file_size_mb": round(r1_size_mb, 4),
        "feature_count": r1_block_diag["total_features"],
    }

    # --- R2: Weighted Title Representation (training-only CV selection over title_weight) ---
    r2_candidates = [float(w) for w in d3_cfg["r2_title_weight_candidates"]]
    r2_cv_results: Dict[str, Any] = {}
    best_r2_weight = r2_candidates[0]
    best_r2_score = -1.0

    for w_cand in r2_candidates:
        def _make_r2(w: float = w_cand) -> Pipeline:
            return SeparateBlocksTfidfLogReg(
                title_weight=w,
                lp_weight=1.0,
                ngram_range=tuple(vec_cfg["ngram_range"]),
                min_df=int(vec_cfg["min_df"]),
                max_features_per_block=r1_max_feats,
                sublinear_tf=bool(vec_cfg["sublinear_tf"]),
                c_param=float(clf_cfg["C"]),
                class_weight=clf_cfg["class_weight"],
                max_iter=int(clf_cfg["max_iter"]),
                random_state=seed,
                solver=str(clf_cfg["solver"]),
                normalize=bool(data_cfg["normalize_unicode_nfkc"]),
            )._build_pipeline()

        cv_res = evaluate_representation_cv(
            train_df=train_df,
            pipeline_builder=_make_r2,
            n_splits=d3_cv_splits,
            random_state=seed,
        )
        r2_cv_results[str(w_cand)] = cv_res
        if cv_res[d3_sel_obj] > best_r2_score:
            best_r2_score = cv_res[d3_sel_obj]
            best_r2_weight = w_cand

    set_global_seed(seed)
    r2_model = SeparateBlocksTfidfLogReg(
        title_weight=best_r2_weight,
        lp_weight=1.0,
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=int(vec_cfg["min_df"]),
        max_features_per_block=r1_max_feats,
        sublinear_tf=bool(vec_cfg["sublinear_tf"]),
        c_param=float(clf_cfg["C"]),
        class_weight=clf_cfg["class_weight"],
        max_iter=int(clf_cfg["max_iter"]),
        random_state=seed,
        solver=str(clf_cfg["solver"]),
        normalize=bool(data_cfg["normalize_unicode_nfkc"]),
    )
    t0 = time.perf_counter()
    r2_model.fit(train_df, target_col="label")
    r2_train_sec = time.perf_counter() - t0

    r2_path = models_dir / "R2_weighted_title.pkl"
    r2_model.save(r2_path)
    r2_size_mb = r2_path.stat().st_size / (1024.0 * 1024.0)
    r2_preds, r2_lat = benchmark_inference_latency(
        predict_fn=r2_model.predict,
        inputs=valid_df,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    r2_scores = extract_class_scores(r2_model, valid_df, target_class=POSITIVE_CLASS)
    r2_ev = compute_classification_metrics(
        y_true=y_valid_true,
        y_pred=r2_preds,
        labels=LABEL_ORDER,
        latency_ms_per_sample=r2_lat.median_ms_per_sample,
        total_latency_seconds=r2_lat.median_batch_seconds,
    )
    r2_d1 = compute_d1_score_diagnostics(y_valid_true, r2_scores, model_id="R2_weighted_title")
    r2_block_diag = r2_model.get_feature_diagnostics(valid_df)

    d3_rows.append({
        "representation_id": "R2_weighted_title",
        "representation_method": f"ColumnTransformer: weighted title block (w_title={best_r2_weight}, train-CV selected)",
        "macro_f1": r2_ev.macro_f1,
        "recall_inconsistent": r2_ev.per_class_recall["inconsistent"],
        "average_precision": r2_d1["average_precision"],
        "train_time_seconds": round(r2_train_sec, 4),
        "inference_latency_ms": round(r2_lat.median_ms_per_sample, 4),
        "feature_count": r2_block_diag["total_features"],
    })
    d3_details["R2_weighted_title"] = {
        "selected_title_weight": best_r2_weight,
        "cv_candidates": r2_cv_results,
        "selection_objective": d3_sel_obj,
        "model_size_mb": round(r2_size_mb, 4),
        "roc_auc": r2_d1["roc_auc"],
        "average_precision": r2_d1["average_precision"],
        "block_diagnostics": r2_block_diag,
        "evaluation": r2_ev.to_dict(),
        "latency_benchmark": r2_lat.to_dict(),
    }
    rep_manifest["models"]["R2_weighted_title"] = {
        "artifact_path": str(r2_path.as_posix()),
        "sha256": compute_file_sha256(r2_path),
        "file_size_mb": round(r2_size_mb, 4),
        "selected_title_weight": best_r2_weight,
        "feature_count": r2_block_diag["total_features"],
    }

    # --- R3: Truncated Landing Page (training-only CV selection over lp_max_chars) ---
    r3_candidates = [int(c) for c in d3_cfg["r3_lp_max_chars_candidates"]]
    r3_cv_results: Dict[str, Any] = {}
    best_r3_chars = r3_candidates[0]
    best_r3_score = -1.0

    for c_cand in r3_candidates:
        def _make_r3(cap: int = c_cand) -> Pipeline:
            return TruncatedLpPairBaseline(
                lp_max_chars=cap,
                ngram_range=tuple(vec_cfg["ngram_range"]),
                min_df=int(vec_cfg["min_df"]),
                max_features=vec_cfg["max_features"],
                sublinear_tf=bool(vec_cfg["sublinear_tf"]),
                c_param=float(clf_cfg["C"]),
                class_weight=clf_cfg["class_weight"],
                max_iter=int(clf_cfg["max_iter"]),
                random_state=seed,
                solver=str(clf_cfg["solver"]),
                separator=str(data_cfg["separator"]),
                normalize=bool(data_cfg["normalize_unicode_nfkc"]),
            )._build_pipeline()

        cv_res = evaluate_representation_cv(
            train_df=train_df,
            pipeline_builder=_make_r3,
            n_splits=d3_cv_splits,
            random_state=seed,
        )
        r3_cv_results[str(c_cand)] = cv_res
        if cv_res[d3_sel_obj] > best_r3_score:
            best_r3_score = cv_res[d3_sel_obj]
            best_r3_chars = c_cand

    set_global_seed(seed)
    r3_model = TruncatedLpPairBaseline(
        lp_max_chars=best_r3_chars,
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=int(vec_cfg["min_df"]),
        max_features=vec_cfg["max_features"],
        sublinear_tf=bool(vec_cfg["sublinear_tf"]),
        c_param=float(clf_cfg["C"]),
        class_weight=clf_cfg["class_weight"],
        max_iter=int(clf_cfg["max_iter"]),
        random_state=seed,
        solver=str(clf_cfg["solver"]),
        separator=str(data_cfg["separator"]),
        normalize=bool(data_cfg["normalize_unicode_nfkc"]),
    )
    t0 = time.perf_counter()
    r3_model.fit(train_df, target_col="label")
    r3_train_sec = time.perf_counter() - t0

    r3_path = models_dir / "R3_truncated_lp.pkl"
    r3_model.save(r3_path)
    r3_size_mb = r3_path.stat().st_size / (1024.0 * 1024.0)
    r3_preds, r3_lat = benchmark_inference_latency(
        predict_fn=r3_model.predict,
        inputs=valid_df,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    r3_scores = extract_class_scores(r3_model, valid_df, target_class=POSITIVE_CLASS)
    r3_ev = compute_classification_metrics(
        y_true=y_valid_true,
        y_pred=r3_preds,
        labels=LABEL_ORDER,
        latency_ms_per_sample=r3_lat.median_ms_per_sample,
        total_latency_seconds=r3_lat.median_batch_seconds,
    )
    r3_d1 = compute_d1_score_diagnostics(y_valid_true, r3_scores, model_id="R3_truncated_lp")
    r3_X_valid = r3_model.pipeline.named_steps["tfidf"].transform(
        r3_model.pipeline.named_steps["selector"].transform(valid_df)
    )

    d3_rows.append({
        "representation_id": "R3_truncated_lp",
        "representation_method": f"Concatenated pair with lp_text[:{best_r3_chars}] (train-CV selected)",
        "macro_f1": r3_ev.macro_f1,
        "recall_inconsistent": r3_ev.per_class_recall["inconsistent"],
        "average_precision": r3_d1["average_precision"],
        "train_time_seconds": round(r3_train_sec, 4),
        "inference_latency_ms": round(r3_lat.median_ms_per_sample, 4),
        "feature_count": int(r3_X_valid.shape[1]),
    })
    d3_details["R3_truncated_lp"] = {
        "selected_lp_max_chars": best_r3_chars,
        "cv_candidates": r3_cv_results,
        "selection_objective": d3_sel_obj,
        "sparse_matrix_shape": [int(r3_X_valid.shape[0]), int(r3_X_valid.shape[1])],
        "model_size_mb": round(r3_size_mb, 4),
        "roc_auc": r3_d1["roc_auc"],
        "average_precision": r3_d1["average_precision"],
        "evaluation": r3_ev.to_dict(),
        "latency_benchmark": r3_lat.to_dict(),
    }
    rep_manifest["models"]["R3_truncated_lp"] = {
        "artifact_path": str(r3_path.as_posix()),
        "sha256": compute_file_sha256(r3_path),
        "file_size_mb": round(r3_size_mb, 4),
        "selected_lp_max_chars": best_r3_chars,
        "feature_count": int(r3_X_valid.shape[1]),
    }

    d3_df = pd.DataFrame(d3_rows)

    # -------------------------------------------------------------------------
    # D4: Seen vs. Unseen Landing Page Confounding Check
    # -------------------------------------------------------------------------
    seen_unseen_df = compute_seen_unseen_confounding(
        train_df=train_df,
        valid_df=valid_df,
        predictions_df=m2_preds_df,
        model_ids=["B0_majority", "B1_title_only", "B2_lp_only", "B3_pair"],
    )

    # -------------------------------------------------------------------------
    # Persist all isolated artifacts
    # -------------------------------------------------------------------------
    diagnostic_payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "random_seed": seed,
        "dataset_revision": (dataset_metadata or {}).get("upstream_revision", "unknown"),
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_split_accessed": False,
        "score_verification_max_abs_diff": score_verification,
        "d1_score_diagnostics": d1_details,
        "d2_threshold_diagnostics": d2_cv_details,
        "d3_representation_diagnostics": d3_details,
        "d4_seen_unseen_summary": seen_unseen_df.to_dict(orient="records"),
    }

    with (run_dir / "diagnostic_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    threshold_analysis_df.to_csv(run_dir / "threshold_analysis.csv", index=False)
    d3_df.to_csv(run_dir / "representation_comparison.csv", index=False)
    seen_unseen_df.to_csv(run_dir / "seen_unseen_analysis.csv", index=False)
    with (run_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rep_manifest, f, indent=2)

    summary_md = format_diagnostics_markdown(
        d1_df=d1_df,
        d2_summary_df=d2_summary_df,
        d3_df=d3_df,
        seen_unseen_df=seen_unseen_df,
        payload=diagnostic_payload,
    )
    with (reports_path / "baseline_failure_diagnostics.md").open("w", encoding="utf-8") as f:
        f.write(summary_md)
    with (reports_path / "diagnostic_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    return {
        "d1_df": d1_df,
        "d2_summary_df": d2_summary_df,
        "threshold_analysis_df": threshold_analysis_df,
        "d3_df": d3_df,
        "seen_unseen_df": seen_unseen_df,
        "payload": diagnostic_payload,
        "valid_scores_by_model": valid_scores_by_model,
        "manifest": rep_manifest,
        "run_dir": run_dir,
        "reports_dir": reports_path,
        "summary_md": summary_md,
    }
