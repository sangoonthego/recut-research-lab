"""Error analysis, model inspection, and slice diagnostic utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from recut_lab.baseline import TfidfLogRegBaseline
from recut_lab.features import calculate_lexical_overlap


def extract_misclassifications(
    df: pd.DataFrame,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probs: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Isolate misclassified instances and attach diagnostic metadata.

    Args:
        df: Input DataFrame.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        probs: Optional predicted probabilities.
        classes: Optional class names corresponding to probability columns.

    Returns:
        DataFrame containing error slices and diagnostic attributes.
    """
    records = []
    y_t = list(y_true)
    y_p = list(y_pred)

    for i, (yt, yp) in enumerate(zip(y_t, y_p)):
        is_error = yt != yp
        row = {
            "index": df.index[i] if i < len(df) else i,
            "true_label": yt,
            "pred_label": yp,
            "is_error": is_error,
            "error_type": "FP" if (yp == "consistent" and yt == "inconsistent") else ("FN" if (yp == "inconsistent" and yt == "consistent") else "CORRECT"),
        }
        if "title" in df.columns:
            row["title_len"] = len(str(df.iloc[i]["title"]))
        if "lp_text" in df.columns:
            row["lp_len"] = len(str(df.iloc[i]["lp_text"]))

        if probs is not None and classes is not None and i < len(probs):
            for c_idx, c_name in enumerate(classes):
                row[f"prob_{c_name}"] = float(probs[i, c_idx])

        records.append(row)

    out_df = pd.DataFrame(records)
    return out_df


def compute_error_breakdown(error_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary distributions of error types."""
    total = len(error_df)
    errors = error_df[error_df["is_error"]]
    num_errors = len(errors)

    fp_count = int((error_df["error_type"] == "FP").sum())
    fn_count = int((error_df["error_type"] == "FN").sum())

    return {
        "total_instances": total,
        "error_count": num_errors,
        "error_rate": round(num_errors / max(total, 1), 4),
        "false_positive_count": fp_count,
        "false_positive_pct_of_errors": round(fp_count / max(num_errors, 1) * 100, 2),
        "false_negative_count": fn_count,
        "false_negative_pct_of_errors": round(fn_count / max(num_errors, 1) * 100, 2),
    }


def get_top_tfidf_features(
    model: TfidfLogRegBaseline,
    top_k: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """Extract top positive and negative character n-grams from TF-IDF Logistic Regression.

    Returns:
        Dict mapping class to list of (feature_name, coefficient_weight).
    """
    if model.pipeline is None:
        raise RuntimeError("Model pipeline has not been fitted.")

    tfidf = model.pipeline.named_steps["tfidf"]
    clf = model.pipeline.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())
    classes = clf.classes_
    coefs = clf.coef_

    results: Dict[str, List[Tuple[str, float]]] = {}

    if len(classes) == 2:
        # Binary case: coefs is shape (1, n_features), corresponding to classes[1]
        c1_name = str(classes[1])
        c0_name = str(classes[0])

        c1_weights = coefs[0]
        c1_top_idx = np.argsort(c1_weights)[::-1][:top_k]
        c0_top_idx = np.argsort(c1_weights)[:top_k]

        results[c1_name] = [(str(feature_names[i]), round(float(c1_weights[i]), 4)) for i in c1_top_idx]
        results[c0_name] = [(str(feature_names[i]), round(float(-c1_weights[i]), 4)) for i in c0_top_idx]
    else:
        for idx, c_name in enumerate(classes):
            c_weights = coefs[idx]
            top_idx = np.argsort(c_weights)[::-1][:top_k]
            results[str(c_name)] = [(str(feature_names[i]), round(float(c_weights[i]), 4)) for i in top_idx]

    return results


def build_error_inspection_table(
    valid_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_id: str,
    error_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Join validation predictions with in-memory full Japanese text for inspection.

    Only operates on validation data in memory; does not persist source text to disk.
    Columns returned match M2.0 Table 2:
        sample_id, true_label, predicted_label, model_score, lp_text, title

    Note:
        ``model_score`` is the raw uncalibrated ``predict_proba`` score for the
        ``inconsistent`` class and must not be interpreted as a calibrated probability.
    """
    subset = predictions_df[
        (predictions_df["model_id"] == model_id) & (predictions_df["is_error"])
    ].copy()

    if error_type is not None:
        subset = subset[subset["error_type"] == error_type].copy()

    # Sort deterministically by confidence in the erroneous prediction (highest error score first)
    if "score_inconsistent" in subset.columns:
        subset["_sort_score"] = subset.apply(
            lambda r: (
                float(r["score_inconsistent"])
                if r["predicted_label"] == "inconsistent" and pd.notna(r["score_inconsistent"])
                else (
                    float(r["score_consistent"])
                    if pd.notna(r.get("score_consistent"))
                    else 0.0
                )
            ),
            axis=1,
        )
        subset = subset.sort_values(
            by=["_sort_score", "valid_row_index"], ascending=[False, True]
        )

    if limit is not None:
        subset = subset.head(limit)

    rows = []
    for _, pred_row in subset.iterrows():
        v_idx = int(pred_row["valid_row_index"])
        src_row = valid_df.loc[v_idx]
        score_val = pred_row.get("score_inconsistent")
        rows.append({
            "sample_id": pred_row["sample_id"],
            "true_label": pred_row["true_label"],
            "predicted_label": pred_row["predicted_label"],
            "model_score": round(float(score_val), 4) if pd.notna(score_val) else None,
            "lp_text": str(src_row["lp_text"]),
            "title": str(src_row["title"]),
        })

    return pd.DataFrame(
        rows,
        columns=["sample_id", "true_label", "predicted_label", "model_score", "lp_text", "title"],
    )

