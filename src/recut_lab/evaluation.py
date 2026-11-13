"""Model evaluation metrics, confusion matrix analysis, and latency benchmarking."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_ORDER = ["consistent", "inconsistent"]


@dataclass
class ConfusionMatrixReport:
    """Confusion matrix decomposition."""

    labels: List[str]
    matrix: List[List[int]]
    true_positives: Dict[str, int]
    false_positives: Dict[str, int]
    false_negatives: Dict[str, int]
    true_negatives: Dict[str, int]


@dataclass
class EvaluationReport:
    """Complete evaluation report for a classification run."""

    num_samples: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: ConfusionMatrixReport
    inference_latency_ms_per_sample: float
    total_inference_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Return as serializable dictionary."""
        return asdict(self)


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ("consistent", "inconsistent"),
    latency_ms_per_sample: float = 0.0,
    total_latency_seconds: float = 0.0,
) -> EvaluationReport:
    """Compute standard classification metrics for binary or multi-class predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Label order for metrics and matrix indexing.
        latency_ms_per_sample: Benchmarked inference latency.
        total_latency_seconds: Total benchmark duration.

    Returns:
        EvaluationReport instance.
    """
    labels_list = list(labels)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0))
    macro_prec = float(
        precision_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )
    macro_rec = float(
        recall_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )

    per_f1_arr = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    per_prec_arr = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    per_rec_arr = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)

    per_class_f1 = {lbl: round(float(f), 4) for lbl, f in zip(labels_list, per_f1_arr)}
    per_class_prec = {lbl: round(float(p), 4) for lbl, p in zip(labels_list, per_prec_arr)}
    per_class_rec = {lbl: round(float(r), 4) for lbl, r in zip(labels_list, per_rec_arr)}

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_matrix = [[int(val) for val in row] for row in cm_raw]

    # Decompose TP, FP, FN, TN per class
    tp_map: Dict[str, int] = {}
    fp_map: Dict[str, int] = {}
    fn_map: Dict[str, int] = {}
    tn_map: Dict[str, int] = {}
    total_n = len(y_true)

    for i, lbl in enumerate(labels_list):
        tp = int(cm_raw[i, i])
        fn = int(np.sum(cm_raw[i, :]) - tp)
        fp = int(np.sum(cm_raw[:, i]) - tp)
        tn = int(total_n - tp - fn - fp)
        tp_map[lbl] = tp
        fp_map[lbl] = fp
        fn_map[lbl] = fn
        tn_map[lbl] = tn

    cm_report = ConfusionMatrixReport(
        labels=labels_list,
        matrix=cm_matrix,
        true_positives=tp_map,
        false_positives=fp_map,
        false_negatives=fn_map,
        true_negatives=tn_map,
    )

    return EvaluationReport(
        num_samples=len(y_true),
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        macro_precision=round(macro_prec, 4),
        macro_recall=round(macro_rec, 4),
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        confusion_matrix=cm_report,
        inference_latency_ms_per_sample=round(latency_ms_per_sample, 4),
        total_inference_seconds=round(total_latency_seconds, 4),
    )


def measure_inference_latency(
    predict_fn: Callable[[Any], Sequence[str]],
    inputs: Any,
    num_runs: int = 1,
) -> Tuple[Sequence[str], float, float]:
    """Benchmark prediction function latency.

    Args:
        predict_fn: Callable that accepts inputs and returns predictions.
        inputs: Batched or full inputs.
        num_runs: Repetitions for stable timing.

    Returns:
        Tuple of (predictions, latency_ms_per_sample, total_seconds).
    """
    # Warmup
    _ = predict_fn(inputs)

    start_time = time.perf_counter()
    preds = None
    for _ in range(num_runs):
        preds = predict_fn(inputs)
    total_time = time.perf_counter() - start_time

    n_samples = max(len(inputs), 1)
    avg_total_time = total_time / num_runs
    latency_ms = (avg_total_time / n_samples) * 1000.0

    return preds, latency_ms, avg_total_time


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    latency_ms: float = 0.0,
    total_time: float = 0.0,
) -> Dict[str, Any]:
    """Helper returning a clean dictionary representation of evaluation metrics."""
    rep = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        latency_ms_per_sample=latency_ms,
        total_latency_seconds=total_time,
    )
    return rep.to_dict()

