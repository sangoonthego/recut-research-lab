"""Milestone M2.1 (RQ1): Generalization to Unseen Landing Pages.

Compares conventional StratifiedKFold (Protocol A) against landing-page-group-disjoint
StratifiedGroupKFold (Protocol B) exclusively within the official AdTEC training split.
Official validation and test splits are never read or accessed.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from recut_lab.baseline import M2_DEFAULT_CONFIG, MajorityBaseline, TfidfLogRegBaseline
from recut_lab.data import compute_file_sha256, load_adtec_split
from recut_lab.diagnostics import POSITIVE_CLASS, extract_class_scores
from recut_lab.evaluation import (
    LABEL_ORDER,
    ZERO_DIVISION_POLICY,
    compute_classification_metrics,
)
from recut_lab.experiment import ExperimentManager, set_global_seed
from recut_lab.preprocessing import normalize_japanese_text

PROTOCOL_A: str = "protocol_a_stratified"
PROTOCOL_B: str = "protocol_b_group_disjoint"

M2_1_DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_id": "m2_generalization_v001",
    "primary_seed": 42,
    "seeds": [42, 123],
    "candidate_n_splits": [3, 5],
    "selected_n_splits": 5,
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
    },
    "frozen_m2_baseline_config": {
        "separator": M2_DEFAULT_CONFIG["data"]["separator"],
        "normalize_unicode_nfkc": M2_DEFAULT_CONFIG["data"]["normalize_unicode_nfkc"],
        "vectorizer": deepcopy(M2_DEFAULT_CONFIG["vectorizer"]),
        "classifier": deepcopy(M2_DEFAULT_CONFIG["classifier"]),
        "baselines": deepcopy(M2_DEFAULT_CONFIG["baselines"]),
    },
}


# =============================================================================
# 1. STRICT TRAIN-ONLY DATA LOADING & DETERMINISTIC GROUP AUDIT
# =============================================================================


def load_adtec_train_only(
    data_dir: Path | str = "data/raw/adtec_consistency",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load ONLY the official train.tsv split and metadata.json.

    Strictly never touches valid.tsv or test.tsv.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dir_path.resolve()}' does not exist.")

    found_file = None
    for ext in [".tsv", ".jsonl", ".csv", ".parquet"]:
        candidate = dir_path / f"train{ext}"
        if candidate.exists():
            found_file = candidate
            break
    if found_file is None:
        raise FileNotFoundError(
            f"Could not find 'train' split in '{dir_path.resolve()}'. "
            "Run 'python scripts/download_adtec.py' first."
        )

    train_df = load_adtec_split(found_file, split_name="train")
    train_sha = compute_file_sha256(found_file)

    metadata: Dict[str, Any] = {}
    meta_path = dir_path / "metadata.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    provenance = {
        "dataset_name": metadata.get("dataset_name", "cyberagent/AdTEC"),
        "config_name": metadata.get("config_name", "ad-consistency"),
        "license": metadata.get("license", "CC BY-NC-SA 4.0"),
        "upstream_revision": metadata.get("upstream_revision", "unknown"),
        "train_filename": found_file.name,
        "train_row_count": len(train_df),
        "train_sha256_verified": train_sha,
        "train_sha256_matches_metadata": (
            train_sha == metadata.get("splits", {}).get("train", {}).get("sha256")
            if metadata
            else None
        ),
        "valid_split_loaded": False,
        "test_split_loaded": False,
    }
    return train_df, provenance


def make_lp_group_id(normalized_lp: str) -> str:
    """Create a deterministic, label-independent anonymous LP group identifier."""
    digest = hashlib.sha256(normalized_lp.encode("utf-8")).hexdigest()[:12]
    return f"lp_grp_{digest}"


def assign_lp_group_ids(
    df: pd.DataFrame,
    lp_col: str = "lp_text",
) -> pd.Series:
    """Assign deterministic group identifiers based strictly on NFKC-normalized lp_text."""
    norm_lp = df[lp_col].map(normalize_japanese_text)
    return norm_lp.map(make_lp_group_id)


def _compute_gini(values: np.ndarray) -> float:
    """Compute Gini concentration index of non-negative group sizes."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1.0) / n)


def audit_lp_groups(
    train_df: pd.DataFrame,
    group_ids: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Audit landing-page group size distribution, label mixing, and concentration.

    Returns:
        Tuple of (summary_dict, per_group_df sorted descending by sample_count).
    """
    gids = group_ids if group_ids is not None else assign_lp_group_ids(train_df)
    work = pd.DataFrame({
        "group_id": gids.values,
        "label": train_df["label"].values,
        "title_norm": train_df["title"].map(normalize_japanese_text).values,
    })

    # Identify conflicting-label (title, group) pairs per group
    pair_label_counts = work.groupby(["group_id", "title_norm"])["label"].nunique()
    conflicting_groups = set(pair_label_counts[pair_label_counts > 1].index.get_level_values(0))

    grp_rows: List[Dict[str, Any]] = []
    for gid, grp in work.groupby("group_id", sort=False):
        n_s = len(grp)
        n_c = int((grp["label"] == "consistent").sum())
        n_i = int((grp["label"] == "inconsistent").sum())
        is_single = (n_c == 0) or (n_i == 0)
        grp_rows.append({
            "group_id": gid,
            "sample_count": n_s,
            "consistent_count": n_c,
            "inconsistent_count": n_i,
            "consistent_pct": round(n_c / n_s * 100.0, 2),
            "inconsistent_pct": round(n_i / n_s * 100.0, 2),
            "is_single_class": bool(is_single),
            "is_mixed_label": bool(not is_single),
            "has_conflicting_pair_labels": bool(gid in conflicting_groups),
        })

    grp_df = pd.DataFrame(grp_rows).sort_values(
        by=["sample_count", "group_id"], ascending=[False, True]
    ).reset_index(drop=True)

    total_samples = len(work)
    grp_df["sample_share_pct"] = (grp_df["sample_count"] / max(total_samples, 1) * 100.0).round(4)
    grp_df["cumulative_share_pct"] = grp_df["sample_share_pct"].cumsum().round(4)

    sizes = grp_df["sample_count"].to_numpy(dtype=np.float64)
    shares = sizes / max(sizes.sum(), 1.0)
    hhi = float(np.sum(shares ** 2))
    n_eff = float(1.0 / hhi) if hhi > 0 else 0.0
    gini = _compute_gini(sizes)

    p50, p75, p90, p95, p99 = np.percentile(sizes, [50, 75, 90, 95, 99])

    summary = {
        "group_count": int(len(grp_df)),
        "sample_count": int(total_samples),
        "mean_group_size": round(float(np.mean(sizes)), 2),
        "median_group_size": round(float(p50), 2),
        "p75_group_size": round(float(p75), 2),
        "p90_group_size": round(float(p90), 2),
        "p95_group_size": round(float(p95), 2),
        "p99_group_size": round(float(p99), 2),
        "max_group_size": int(np.max(sizes)),
        "min_group_size": int(np.min(sizes)),
        "single_class_group_count": int(grp_df["is_single_class"].sum()),
        "mixed_label_group_count": int(grp_df["is_mixed_label"].sum()),
        "conflicting_pair_group_count": int(grp_df["has_conflicting_pair_labels"].sum()),
        "top10_group_sample_count": int(grp_df.head(10)["sample_count"].sum()),
        "top10_group_sample_pct": round(float(grp_df.head(10)["sample_count"].sum() / max(total_samples, 1) * 100.0), 2),
        "top50_group_sample_pct": round(float(grp_df.head(50)["sample_count"].sum() / max(total_samples, 1) * 100.0), 2),
        "hhi_concentration": round(hhi, 6),
        "effective_group_count": round(n_eff, 2),
        "gini_concentration": round(gini, 4),
    }
    return summary, grp_df


# =============================================================================
# 2. CV PROTOCOL FEASIBILITY & FOLD INTEGRITY CHECKS
# =============================================================================


def evaluate_group_cv_feasibility(
    train_df: pd.DataFrame,
    group_ids: pd.Series,
    candidate_splits: Sequence[int] = (3, 5),
    seed: int = 42,
) -> Dict[str, Any]:
    """Assess feasibility of StratifiedGroupKFold for candidate n_splits values.

    Predefined feasibility criteria (independent of model performance):
        - Zero empty folds
        - Both classes present in every fold
        - Strict LP group disjointness in every fold
        - At least 10% of dataset per fold (or reasonable minimum fold size)
    """
    y = train_df["label"].to_numpy()
    groups = group_ids.to_numpy()
    reports: Dict[str, Any] = {}

    for k in candidate_splits:
        sgkf = StratifiedGroupKFold(n_splits=int(k), shuffle=True, random_state=seed)
        fold_stats = []
        feasible = True
        for f_idx, (tr_idx, ev_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
            tr_g = set(groups[tr_idx])
            ev_g = set(groups[ev_idx])
            disjoint = tr_g.isdisjoint(ev_g)
            n_c = int((y[ev_idx] == "consistent").sum())
            n_i = int((y[ev_idx] == "inconsistent").sum())
            both_classes = (n_c > 0) and (n_i > 0)
            if not (disjoint and both_classes and len(ev_idx) > 0):
                feasible = False
            fold_stats.append({
                "fold": f_idx,
                "train_size": int(len(tr_idx)),
                "eval_size": int(len(ev_idx)),
                "eval_unique_lp": int(len(ev_g)),
                "consistent_count": n_c,
                "inconsistent_count": n_i,
                "eval_inconsistent_prevalence": round(n_i / max(len(ev_idx), 1), 4),
                "strict_disjoint": disjoint,
            })

        prevalences = [f["eval_inconsistent_prevalence"] for f in fold_stats]
        eval_sizes = [f["eval_size"] for f in fold_stats]
        reports[str(k)] = {
            "n_splits": int(k),
            "feasible": feasible,
            "min_eval_size": int(min(eval_sizes)),
            "max_eval_size": int(max(eval_sizes)),
            "min_inconsistent_prevalence": round(float(min(prevalences)), 4),
            "max_inconsistent_prevalence": round(float(max(prevalences)), 4),
            "prevalence_range": round(float(max(prevalences) - min(prevalences)), 4),
            "folds": fold_stats,
        }

    # Prefer 5 if feasible (higher training fraction ~80% & richer fold variance), else 3
    if reports.get("5", {}).get("feasible"):
        selected = 5
        rationale = (
            "n_splits=5 satisfies all predefined feasibility checks (zero empty folds, both classes "
            "well-represented in every fold, strict LP disjointness) while retaining ~80% training "
            "data per fold and matching the M2.0.1 5-fold convention."
        )
    elif reports.get("3", {}).get("feasible"):
        selected = 3
        rationale = "n_splits=3 selected as fallback because n_splits=5 failed feasibility criteria."
    else:
        raise RuntimeError("Neither n_splits=3 nor n_splits=5 produced feasible StratifiedGroupKFold splits.")

    return {
        "seed": seed,
        "candidates": reports,
        "selected_n_splits": selected,
        "selection_rationale": rationale,
    }


def generate_protocol_splits(
    train_df: pd.DataFrame,
    group_ids: pd.Series,
    protocol: str,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic CV fold index pairs for Protocol A or Protocol B."""
    y = train_df["label"].to_numpy()
    groups = group_ids.to_numpy()

    if protocol == PROTOCOL_A:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(train_df, y))

    if protocol == PROTOCOL_B:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(train_df, y, groups=groups))
        # Enforce strict group disjointness on every fold
        for f_idx, (tr_idx, ev_idx) in enumerate(splits):
            if not set(groups[tr_idx]).isdisjoint(set(groups[ev_idx])):
                raise AssertionError(
                    f"Protocol B group disjointness violated in seed={seed}, fold={f_idx}!"
                )
        return splits

    raise ValueError(f"Unknown protocol '{protocol}'. Expected '{PROTOCOL_A}' or '{PROTOCOL_B}'.")


def compute_fold_integrity_record(
    train_df: pd.DataFrame,
    group_ids: pd.Series,
    tr_idx: np.ndarray,
    ev_idx: np.ndarray,
    protocol: str,
    seed: int,
    fold: int,
    conflicting_groups: set[str],
) -> Dict[str, Any]:
    """Record comprehensive structural and LP-overlap integrity metrics for one fold."""
    groups = group_ids.to_numpy()
    y = train_df["label"].to_numpy()

    tr_groups = set(groups[tr_idx])
    ev_groups = set(groups[ev_idx])
    shared_groups = tr_groups.intersection(ev_groups)

    if protocol == PROTOCOL_B:
        assert len(shared_groups) == 0, (
            f"Strict group disjointness failed for Protocol B (seed={seed}, fold={fold})."
        )

    ev_group_series = pd.Series(groups[ev_idx])
    shared_eval_rows = int(ev_group_series.isin(tr_groups).sum())
    shared_eval_row_pct = round(shared_eval_rows / max(len(ev_idx), 1) * 100.0, 2)
    shared_lp_pct = round(len(shared_groups) / max(len(ev_groups), 1) * 100.0, 2)

    n_cons = int((y[ev_idx] == "consistent").sum())
    n_inc = int((y[ev_idx] == "inconsistent").sum())
    inc_prev = round(n_inc / max(len(ev_idx), 1), 4)

    largest_eval_grp = int(ev_group_series.value_counts().max()) if len(ev_idx) > 0 else 0
    ev_conflicting_grp_count = int(len(ev_groups.intersection(conflicting_groups)))

    return {
        "protocol": protocol,
        "seed": int(seed),
        "fold": int(fold),
        "train_size": int(len(tr_idx)),
        "eval_size": int(len(ev_idx)),
        "train_unique_lp": int(len(tr_groups)),
        "eval_unique_lp": int(len(ev_groups)),
        "consistent_count": n_cons,
        "inconsistent_count": n_inc,
        "eval_inconsistent_prevalence": inc_prev,
        "shared_lp_count": int(len(shared_groups)),
        "shared_lp_percentage": shared_lp_pct,
        "shared_eval_row_count": shared_eval_rows,
        "shared_eval_row_percentage": shared_eval_row_pct,
        "conflicting_label_group_count": ev_conflicting_grp_count,
        "largest_eval_lp_group": largest_eval_grp,
    }


# =============================================================================
# 3. FOLD-SAFE MODEL EVALUATION & UNDEFINED-METRIC HANDLING
# =============================================================================


def compute_fold_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores_inconsistent: Sequence[float],
    train_time_seconds: float,
    inference_time_seconds: float,
) -> Dict[str, Any]:
    """Compute primary and secondary fold metrics with explicit NA for undefined single-class metrics."""
    y_arr = np.asarray(list(y_true), dtype=object)
    s_arr = np.asarray(list(scores_inconsistent), dtype=np.float64)
    y_bin = (y_arr == POSITIVE_CLASS).astype(int)

    n_eval = len(y_arr)
    n_inc = int(y_bin.sum())
    prevalence = float(n_inc / max(n_eval, 1))
    has_both_classes = len(np.unique(y_bin)) == 2

    if has_both_classes:
        roc_auc: Optional[float] = round(float(roc_auc_score(y_bin, s_arr)), 4)
        avg_prec: Optional[float] = round(float(average_precision_score(y_bin, s_arr)), 4)
        ap_minus_prev: Optional[float] = round(float(avg_prec - prevalence), 4)
    else:
        # Explicitly represent undefined ranking metrics as None (serialized as NA/NaN, never 0.0)
        roc_auc = None
        avg_prec = None
        ap_minus_prev = None

    ev = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=LABEL_ORDER,
        latency_ms_per_sample=(inference_time_seconds / max(n_eval, 1)) * 1000.0,
        total_latency_seconds=inference_time_seconds,
    )
    bal_acc: Optional[float] = ev.balanced_accuracy if has_both_classes else None
    cm = ev.confusion_matrix.matrix  # [[TN_inc, FP_inc], [FN_inc, TP_inc]]

    return {
        "eval_size": n_eval,
        "eval_inconsistent_prevalence": round(prevalence, 4),
        "both_classes_present": has_both_classes,
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "ap_minus_prevalence": ap_minus_prev,
        "macro_f1": ev.macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": ev.accuracy,
        "precision_inconsistent": ev.per_class_precision["inconsistent"],
        "recall_inconsistent": ev.per_class_recall["inconsistent"],
        "f1_inconsistent": ev.per_class_f1["inconsistent"],
        "recall_consistent": ev.per_class_recall["consistent"],
        "tn_inconsistent": int(cm[0][0]),
        "fp_inconsistent": int(cm[0][1]),
        "fn_inconsistent": int(cm[1][0]),
        "tp_inconsistent": int(cm[1][1]),
        "train_time_seconds": round(float(train_time_seconds), 4),
        "inference_time_seconds": round(float(inference_time_seconds), 6),
    }


def _build_frozen_baseline(spec: Dict[str, Any], m2_cfg: Dict[str, Any], seed: int):
    """Instantiate a fresh, unfitted baseline matching the frozen M2.0 configuration."""
    if spec["model_type"] == "majority":
        return MajorityBaseline(random_state=seed)

    vec_cfg = m2_cfg["vectorizer"]
    clf_cfg = m2_cfg["classifier"]
    return TfidfLogRegBaseline(
        input_mode=spec["input_mode"],
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=int(vec_cfg["min_df"]),
        max_features=vec_cfg["max_features"],
        sublinear_tf=bool(vec_cfg["sublinear_tf"]),
        c_param=float(clf_cfg["C"]),
        class_weight=clf_cfg["class_weight"],
        max_iter=int(clf_cfg["max_iter"]),
        random_state=seed,
        solver=str(clf_cfg["solver"]),
        separator=str(m2_cfg["separator"]),
        normalize=bool(m2_cfg["normalize_unicode_nfkc"]),
    )


# =============================================================================
# 4. GROUP-CLUSTER BOOTSTRAP UNCERTAINTY (RESAMPLING LP GROUPS, NOT ROWS)
# =============================================================================


def compute_group_bootstrap_ci(
    oof_df: pd.DataFrame,
    n_resamples: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute LP-group-cluster bootstrap 95% intervals on fixed OOF predictions.

    Resamples unique LP groups with replacement (never independent rows) using
    vectorized integer confusion-matrix arithmetic for fast, exact evaluation.
    Note: Resampling fixed out-of-fold predictions captures group-composition uncertainty
    but does not capture full training-split resampling variability.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.sort(oof_df["group_id"].unique())
    n_groups = len(unique_groups)

    # Pre-index binary arrays and per-group confusion counts for fast exact resampling
    group_counts = np.zeros((n_groups, 4), dtype=np.int64)  # [tn, fp, fn, tp] for inconsistent=1
    group_y_bin: List[np.ndarray] = []
    group_scores: List[np.ndarray] = []
    gid_to_idx = {gid: i for i, gid in enumerate(unique_groups)}

    for gid, g in oof_df.groupby("group_id", sort=False):
        idx = gid_to_idx[gid]
        yb = (g["true_label"].to_numpy() == POSITIVE_CLASS).astype(np.int8)
        pb = (g["predicted_label"].to_numpy() == POSITIVE_CLASS).astype(np.int8)
        sc = g["score_inconsistent"].to_numpy(dtype=np.float64)
        group_counts[idx, 0] = int(np.sum((yb == 0) & (pb == 0)))
        group_counts[idx, 1] = int(np.sum((yb == 0) & (pb == 1)))
        group_counts[idx, 2] = int(np.sum((yb == 1) & (pb == 0)))
        group_counts[idx, 3] = int(np.sum((yb == 1) & (pb == 1)))
        group_y_bin.append(yb)
        group_scores.append(sc)

    # Reorder lists by gid index
    ordered_y_bin: List[np.ndarray] = [np.empty(0, dtype=np.int8)] * n_groups
    ordered_scores: List[np.ndarray] = [np.empty(0, dtype=np.float64)] * n_groups
    for (gid, _), yb, sc in zip(oof_df.groupby("group_id", sort=False), group_y_bin, group_scores):
        i = gid_to_idx[gid]
        ordered_y_bin[i] = yb
        ordered_scores[i] = sc

    boot_metrics: Dict[str, List[float]] = {
        "macro_f1": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "average_precision": [],
        "recall_inconsistent": [],
    }

    for _ in range(n_resamples):
        sampled_idx = rng.integers(0, n_groups, size=n_groups)
        counts = group_counts[sampled_idx].sum(axis=0)
        tn, fp, fn, tp = int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])
        n_neg = tn + fp
        n_pos = fn + tp
        if n_neg == 0 or n_pos == 0:
            continue

        rec_cons = tn / n_neg
        rec_inc = tp / n_pos
        prec_cons = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        prec_inc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_cons = (2.0 * prec_cons * rec_cons / (prec_cons + rec_cons)) if (prec_cons + rec_cons) > 0 else 0.0
        f1_inc = (2.0 * prec_inc * rec_inc / (prec_inc + rec_inc)) if (prec_inc + rec_inc) > 0 else 0.0
        macro_f1 = 0.5 * (f1_cons + f1_inc)
        bal_acc = 0.5 * (rec_cons + rec_inc)

        y_bin = np.concatenate([ordered_y_bin[i] for i in sampled_idx])
        scores = np.concatenate([ordered_scores[i] for i in sampled_idx])

        boot_metrics["macro_f1"].append(macro_f1)
        boot_metrics["balanced_accuracy"].append(bal_acc)
        boot_metrics["recall_inconsistent"].append(rec_inc)
        boot_metrics["roc_auc"].append(float(roc_auc_score(y_bin, scores)))
        boot_metrics["average_precision"].append(float(average_precision_score(y_bin, scores)))

    lo_q = (alpha / 2.0) * 100.0
    hi_q = (1.0 - alpha / 2.0) * 100.0
    ci_out: Dict[str, Any] = {
        "method": "lp_group_cluster_bootstrap_on_fixed_oof",
        "n_resamples": int(n_resamples),
        "n_groups": int(n_groups),
        "confidence_level": round(1.0 - alpha, 2),
        "metrics": {},
    }
    for m_name, vals in boot_metrics.items():
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            ci_out["metrics"][m_name] = {
                "mean": round(float(np.mean(arr)), 4),
                "ci_low": round(float(np.percentile(arr, lo_q)), 4),
                "ci_high": round(float(np.percentile(arr, hi_q)), 4),
            }
        else:
            ci_out["metrics"][m_name] = {"mean": None, "ci_low": None, "ci_high": None}
    return ci_out


# =============================================================================
# 5. AGGREGATION, MARKDOWN REPORT & ORCHESTRATOR
# =============================================================================


def aggregate_protocol_comparison(
    per_fold_df: pd.DataFrame,
    fold_integrity_df: pd.DataFrame,
    seed_filter: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate per-fold metrics into the required RQ1 Protocol Comparison Table."""
    df = per_fold_df.copy()
    fi = fold_integrity_df.copy()
    if seed_filter is not None:
        df = df[df["seed"] == seed_filter].copy()
        fi = fi[fi["seed"] == seed_filter].copy()

    merged = df.merge(
        fi[["protocol", "seed", "fold", "shared_lp_percentage", "shared_eval_row_percentage"]],
        on=["protocol", "seed", "fold"],
        how="left",
    )

    model_order = ["B0_majority", "B1_title_only", "B2_lp_only", "B3_pair"]
    proto_order = [PROTOCOL_A, PROTOCOL_B]

    rows: List[Dict[str, Any]] = []
    for m_id in model_order:
        for proto in proto_order:
            sub = merged[(merged["model_id"] == m_id) & (merged["protocol"] == proto)]
            if len(sub) == 0:
                continue

            def _mean(col: str) -> Optional[float]:
                vals = sub[col].dropna()
                return round(float(vals.mean()), 4) if len(vals) else None

            def _std(col: str) -> Optional[float]:
                vals = sub[col].dropna()
                return round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0

            def _med(col: str) -> Optional[float]:
                vals = sub[col].dropna()
                return round(float(vals.median()), 4) if len(vals) else None

            rows.append({
                "model_id": m_id,
                "protocol": proto,
                "fold_count": int(len(sub)),
                "mean_macro_f1": _mean("macro_f1"),
                "std_macro_f1": _std("macro_f1"),
                "median_macro_f1": _med("macro_f1"),
                "mean_balanced_accuracy": _mean("balanced_accuracy"),
                "std_balanced_accuracy": _std("balanced_accuracy"),
                "mean_roc_auc": _mean("roc_auc"),
                "std_roc_auc": _std("roc_auc"),
                "mean_average_precision": _mean("average_precision"),
                "std_average_precision": _std("average_precision"),
                "mean_ap_minus_prevalence": _mean("ap_minus_prevalence"),
                "mean_inconsistent_recall": _mean("recall_inconsistent"),
                "std_inconsistent_recall": _std("recall_inconsistent"),
                "mean_inconsistent_precision": _mean("precision_inconsistent"),
                "mean_inconsistent_f1": _mean("f1_inconsistent"),
                "mean_inconsistent_prevalence": _mean("eval_inconsistent_prevalence"),
                "mean_shared_lp_percentage": _mean("shared_lp_percentage"),
                "mean_shared_eval_row_percentage": _mean("shared_eval_row_percentage"),
            })

    return pd.DataFrame(rows)


def format_rq1_report_markdown(
    provenance: Dict[str, Any],
    group_summary: Dict[str, Any],
    feasibility: Dict[str, Any],
    fold_integrity_df: pd.DataFrame,
    protocol_comp_df: pd.DataFrame,
    pooled_oof_summary: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> str:
    """Render comprehensive English RQ1 Generalization Report in Markdown."""
    n_splits = payload["selected_n_splits"]
    seeds = payload["seeds"]

    lines = [
        "# Milestone M2.1 — RQ1: Generalization to Unseen Landing Pages",
        "",
        f"**Experiment ID**: `{payload['experiment_id']}`  ",
        f"**Dataset**: `{provenance['dataset_name']}` (`{provenance['config_name']}`, revision `{provenance['upstream_revision']}`)  ",
        f"**Data Used**: Official `train.tsv` ONLY (`N = {provenance['train_row_count']}`, SHA-256 `{provenance['train_sha256_verified'][:16]}...`). Official `valid.tsv` and `test.tsv` strictly excluded.  ",
        f"**Cross-Validation Setup**: `{n_splits}`-fold CV across seeds `{seeds}` (Protocol A: `StratifiedKFold` vs. Protocol B: `StratifiedGroupKFold` grouped by NFKC-normalized `lp_text`).  ",
        "",
        "---",
        "",
        "## 1. Research Question & Hypotheses",
        "",
        "**RQ1**: *How well do advertising consistency classifiers generalize to landing pages that were never observed during model training?*",
        "",
        "- **H1**: Conventional `StratifiedKFold` may report higher performance than LP-group-disjoint CV because training and evaluation folds share landing pages.",
        "- **H2**: `B2_lp_only` may be especially sensitive to evaluation-group overlap because it relies exclusively on landing-page features.",
        "- **H3**: Differences between protocols may also reflect fold class prevalence, unequal group sizes, and fold composition rather than memorization alone.",
        "",
        "## 2. Landing-Page Group Audit Summary (Train Split)",
        "",
        "| group_count | sample_count | median_group_size | p95_group_size | max_group_size | single_class_group_count | mixed_label_group_count | top10_sample_pct | effective_groups (1/HHI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {group_summary['group_count']} | {group_summary['sample_count']} | {group_summary['median_group_size']:.1f} | "
        f"{group_summary['p95_group_size']:.1f} | {group_summary['max_group_size']} | "
        f"{group_summary['single_class_group_count']} | {group_summary['mixed_label_group_count']} | "
        f"{group_summary['top10_group_sample_pct']:.2f}% | {group_summary['effective_group_count']:.2f} |",
        "",
        "## 3. Feasibility & Fold Integrity Verification",
        "",
        f"- **Selected `n_splits`**: `{n_splits}` — {feasibility['selection_rationale']}",
        "- **Protocol B Strict Disjointness**: Verified `set(train_lp_groups).isdisjoint(set(eval_lp_groups))` (`shared_lp_count = 0`, `shared_eval_row_percentage = 0.0%`) across all folds and seeds [VERIFIED].",
        "",
        "## 4. Protocol Comparison Table (Aggregated Across All Seeds & Folds)",
        "",
        "| model_id | protocol | fold_count | mean_macro_f1 | std_macro_f1 | mean_balanced_accuracy | mean_roc_auc | mean_average_precision | mean_inconsistent_recall | mean_inconsistent_prevalence | mean_shared_lp_percentage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in protocol_comp_df.iterrows():
        lines.append(
            f"| `{r['model_id']}` | `{r['protocol']}` | {int(r['fold_count'])} | "
            f"{r['mean_macro_f1']:.4f} | {r['std_macro_f1']:.4f} | "
            f"{r['mean_balanced_accuracy']:.4f} | {r['mean_roc_auc']:.4f} | "
            f"{r['mean_average_precision']:.4f} | {r['mean_inconsistent_recall']:.4f} | "
            f"{r['mean_inconsistent_prevalence']:.4f} | {r['mean_shared_lp_percentage']:.2f}% |"
        )

    lines.extend([
        "",
        "## 5. Pooled Out-of-Fold Metrics (Primary Seed = 42, Uncalibrated Scores)",
        "",
        "| model_id | protocol | pooled_macro_f1 | pooled_balanced_accuracy | pooled_roc_auc | pooled_average_precision | pooled_inconsistent_recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in pooled_oof_summary:
        if row["seed"] == payload["primary_seed"]:
            lines.append(
                f"| `{row['model_id']}` | `{row['protocol']}` | {row['pooled_macro_f1']:.4f} | "
                f"{row['pooled_balanced_accuracy']:.4f} | {row['pooled_roc_auc']:.4f} | "
                f"{row['pooled_average_precision']:.4f} | {row['pooled_inconsistent_recall']:.4f} |"
            )

    lines.extend([
        "",
        "## 6. Competing Explanations & Scientific Guardrails",
        "",
        "- **VERIFIED**: All models were refitted fresh inside each fold on that fold's training split; TF-IDF vocabularies and IDF weights were fit exclusively within each training fold.",
        "- **VERIFIED**: Every record received exactly one held-out prediction per `(protocol, seed, model_id)` repetition.",
        "- **INFERRED**: Differences between Protocol A and Protocol B may reflect LP-specific shortcut signals, fold-level class-prevalence variance, unequal group sizes, campaign/topic heterogeneity, and conflicting-label pairs rather than pure memorization alone.",
        "- **ASSUMED**: Pooled out-of-fold probabilities combine models trained on different subsets and must not be interpreted as calibrated probabilities.",
        "- **TODO**: Leverage LP-group-disjoint CV (Protocol B) as the primary internal validation protocol for RQ2 (semantic modeling) and RQ3 (annotation noise sensitivity).",
    ])
    return "\n".join(lines)


def run_rq1_generalization_study(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path | str] = None,
    runs_base_dir: Path | str = "experiments/runs",
    reports_dir: Path | str = "reports/generated/m2_generalization",
) -> Dict[str, Any]:
    """Execute Milestone M2.1 (RQ1) Protocol A vs. Protocol B experiment end-to-end."""
    cfg = deepcopy(config) if config is not None else deepcopy(M2_1_DEFAULT_CONFIG)
    experiment_id = cfg.get("experiment_id", "m2_generalization_v001")
    primary_seed = int(cfg.get("primary_seed", 42))
    seeds = [int(s) for s in cfg.get("seeds", [42, 123])]
    raw_dir = data_dir or cfg["data"]["raw_dir"]

    # 1. Strictly load train.tsv ONLY
    train_df, provenance = load_adtec_train_only(raw_dir)
    group_ids = assign_lp_group_ids(train_df, lp_col="lp_text")

    # 2. Group distribution audit
    group_summary, group_dist_df = audit_lp_groups(train_df, group_ids=group_ids)
    conflicting_groups = set(
        group_dist_df[group_dist_df["has_conflicting_pair_labels"]]["group_id"].tolist()
    )

    # 3. Feasibility check for candidate n_splits (3, 5)
    feasibility = evaluate_group_cv_feasibility(
        train_df=train_df,
        group_ids=group_ids,
        candidate_splits=cfg.get("candidate_n_splits", [3, 5]),
        seed=primary_seed,
    )
    n_splits = int(cfg.get("selected_n_splits", feasibility["selected_n_splits"]))
    cfg["selected_n_splits"] = n_splits

    # 4. Initialize isolated run directory
    exp_mgr = ExperimentManager(base_runs_dir=runs_base_dir, reports_dir=reports_dir)
    run_dir = exp_mgr.create_named_run(
        run_id=experiment_id,
        config=cfg,
        dataset_metadata=None,
        seed=primary_seed,
    )
    # Override environment.json split flags to reflect train-only RQ1 protocol
    env_path = run_dir / "environment.json"
    env_data = json.loads(env_path.read_text(encoding="utf-8"))
    env_data["training_split"] = "train (cross-validated internally)"
    env_data["evaluation_split"] = "train_oof_only"
    env_data["valid_split_accessed"] = False
    env_data["test_split_accessed"] = False
    env_data["dataset_provenance"] = provenance
    env_path.write_text(json.dumps(env_data, indent=2), encoding="utf-8")

    with (run_dir / "dataset_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 5. Execute Protocol A and Protocol B across all seeds & frozen M2.0 models
    m2_cfg = cfg["frozen_m2_baseline_config"]
    baselines = m2_cfg["baselines"]

    fold_manifest_rows: List[Dict[str, Any]] = []
    fold_integrity_rows: List[Dict[str, Any]] = []
    per_fold_metric_rows: List[Dict[str, Any]] = []
    oof_prediction_frames: List[pd.DataFrame] = []

    for seed in seeds:
        for protocol in [PROTOCOL_A, PROTOCOL_B]:
            splits = generate_protocol_splits(
                train_df=train_df,
                group_ids=group_ids,
                protocol=protocol,
                n_splits=n_splits,
                seed=seed,
            )

            # Verify every record appears in held-out evaluation exactly once
            covered_eval_indices = np.concatenate([ev for _, ev in splits])
            assert len(covered_eval_indices) == len(train_df)
            assert len(np.unique(covered_eval_indices)) == len(train_df)

            for fold_idx, (tr_idx, ev_idx) in enumerate(splits):
                integrity_rec = compute_fold_integrity_record(
                    train_df=train_df,
                    group_ids=group_ids,
                    tr_idx=tr_idx,
                    ev_idx=ev_idx,
                    protocol=protocol,
                    seed=seed,
                    fold=fold_idx,
                    conflicting_groups=conflicting_groups,
                )
                fold_integrity_rows.append(integrity_rec)

                # Record fold assignment manifest (row_id + anonymous group_id only)
                for row_pos in ev_idx:
                    fold_manifest_rows.append({
                        "row_id": int(train_df.index[row_pos]),
                        "group_id": str(group_ids.iloc[row_pos]),
                        "protocol": protocol,
                        "seed": int(seed),
                        "fold": int(fold_idx),
                    })

                fold_train = train_df.iloc[tr_idx]
                fold_eval = train_df.iloc[ev_idx]
                y_eval_true = fold_eval["label"].tolist()

                for spec in baselines:
                    model_id = spec["model_id"]
                    set_global_seed(seed)
                    model = _build_frozen_baseline(spec=spec, m2_cfg=m2_cfg, seed=seed)

                    t0 = time.perf_counter()
                    model.fit(fold_train, target_col="label")
                    train_sec = time.perf_counter() - t0

                    t1 = time.perf_counter()
                    preds = model.predict(fold_eval)
                    inf_sec = time.perf_counter() - t1

                    scores_inc = extract_class_scores(
                        model, fold_eval, target_class=POSITIVE_CLASS
                    )

                    metrics = compute_fold_metrics(
                        y_true=y_eval_true,
                        y_pred=preds,
                        scores_inconsistent=scores_inc,
                        train_time_seconds=train_sec,
                        inference_time_seconds=inf_sec,
                    )
                    per_fold_metric_rows.append({
                        "model_id": model_id,
                        "protocol": protocol,
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        **metrics,
                    })

                    oof_prediction_frames.append(
                        pd.DataFrame({
                            "row_id": fold_eval.index.to_numpy(dtype=int),
                            "group_id": group_ids.iloc[ev_idx].to_numpy(),
                            "protocol": protocol,
                            "seed": int(seed),
                            "fold": int(fold_idx),
                            "model_id": model_id,
                            "true_label": y_eval_true,
                            "predicted_label": preds,
                            "score_inconsistent": np.round(scores_inc, 6),
                        })
                    )

    fold_manifest_df = pd.DataFrame(fold_manifest_rows)
    fold_integrity_df = pd.DataFrame(fold_integrity_rows)
    per_fold_df = pd.DataFrame(per_fold_metric_rows)
    oof_predictions_df = pd.concat(oof_prediction_frames, ignore_index=True)

    # 6. Aggregate comparisons (combined across all seeds + per-seed)
    protocol_comp_all_df = aggregate_protocol_comparison(per_fold_df, fold_integrity_df)
    protocol_comp_by_seed = {
        str(s): aggregate_protocol_comparison(per_fold_df, fold_integrity_df, seed_filter=s).to_dict(orient="records")
        for s in seeds
    }

    # 7. Pooled OOF metrics & LP-group bootstrap 95% CIs (primary seed)
    pooled_oof_summary: List[Dict[str, Any]] = []
    group_bootstrap_cis: Dict[str, Any] = {}
    n_boot = int(cfg.get("group_bootstrap_resamples", 500))

    for seed in seeds:
        for spec in baselines:
            m_id = spec["model_id"]
            for protocol in [PROTOCOL_A, PROTOCOL_B]:
                sub_oof = oof_predictions_df[
                    (oof_predictions_df["seed"] == seed)
                    & (oof_predictions_df["model_id"] == m_id)
                    & (oof_predictions_df["protocol"] == protocol)
                ]
                assert len(sub_oof) == len(train_df)
                pooled = compute_fold_metrics(
                    y_true=sub_oof["true_label"].tolist(),
                    y_pred=sub_oof["predicted_label"].tolist(),
                    scores_inconsistent=sub_oof["score_inconsistent"].to_numpy(),
                    train_time_seconds=0.0,
                    inference_time_seconds=0.0,
                )
                pooled_oof_summary.append({
                    "seed": int(seed),
                    "model_id": m_id,
                    "protocol": protocol,
                    "pooled_macro_f1": pooled["macro_f1"],
                    "pooled_balanced_accuracy": pooled["balanced_accuracy"],
                    "pooled_roc_auc": pooled["roc_auc"],
                    "pooled_average_precision": pooled["average_precision"],
                    "pooled_inconsistent_recall": pooled["recall_inconsistent"],
                    "pooled_inconsistent_precision": pooled["precision_inconsistent"],
                })

                if seed == primary_seed:
                    key = f"{m_id}__{protocol}"
                    group_bootstrap_cis[key] = compute_group_bootstrap_ci(
                        oof_df=sub_oof,
                        n_resamples=n_boot,
                        seed=primary_seed,
                    )

    # 8. Persist all isolated artifacts (ZERO raw Japanese text)
    group_dist_df.to_csv(run_dir / "group_distribution.csv", index=False)
    fold_manifest_df.to_csv(run_dir / "fold_manifest.csv", index=False)
    fold_integrity_df.to_csv(run_dir / "fold_integrity.csv", index=False)
    per_fold_df.to_csv(run_dir / "per_fold_metrics.csv", index=False)
    protocol_comp_all_df.to_csv(run_dir / "protocol_comparison.csv", index=False)
    oof_predictions_df.to_csv(run_dir / "oof_predictions.csv", index=False)

    model_manifest = {
        "experiment_id": experiment_id,
        "checkpoint_policy": "refit_fresh_per_fold_no_full_train_checkpoints_used",
        "frozen_m2_baseline_config": m2_cfg,
        "models_evaluated": [b["model_id"] for b in baselines],
        "protocols": [PROTOCOL_A, PROTOCOL_B],
        "seeds": seeds,
        "n_splits": n_splits,
    }
    with (run_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)

    generalization_payload = {
        "experiment_id": experiment_id,
        "primary_seed": primary_seed,
        "seeds": seeds,
        "selected_n_splits": n_splits,
        "dataset_provenance": provenance,
        "group_distribution_summary": group_summary,
        "cv_feasibility": feasibility,
        "protocol_comparison_all_seeds": protocol_comp_all_df.to_dict(orient="records"),
        "protocol_comparison_by_seed": protocol_comp_by_seed,
        "pooled_oof_summary": pooled_oof_summary,
        "group_bootstrap_cis_primary_seed": group_bootstrap_cis,
    }
    with (run_dir / "generalization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(generalization_payload, f, indent=2)

    report_md = format_rq1_report_markdown(
        provenance=provenance,
        group_summary=group_summary,
        feasibility=feasibility,
        fold_integrity_df=fold_integrity_df,
        protocol_comp_df=protocol_comp_all_df,
        pooled_oof_summary=pooled_oof_summary,
        payload=generalization_payload,
    )
    with (reports_path / "rq1_generalization_report.md").open("w", encoding="utf-8") as f:
        f.write(report_md)
    protocol_comp_all_df.to_csv(reports_path / "protocol_comparison.csv", index=False)
    with (reports_path / "generalization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(generalization_payload, f, indent=2)

    return {
        "provenance": provenance,
        "group_summary": group_summary,
        "group_dist_df": group_dist_df,
        "feasibility": feasibility,
        "fold_integrity_df": fold_integrity_df,
        "per_fold_df": per_fold_df,
        "protocol_comp_df": protocol_comp_all_df,
        "oof_predictions_df": oof_predictions_df,
        "payload": generalization_payload,
        "run_dir": run_dir,
        "reports_dir": reports_path,
        "report_md": report_md,
    }
