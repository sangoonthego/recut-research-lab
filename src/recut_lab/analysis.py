"""Error analysis, model inspection, and slice diagnostic utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

