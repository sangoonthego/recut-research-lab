"""Model evaluation metrics, confusion matrix analysis, and latency benchmarking."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_ORDER: List[str] = ["consistent", "inconsistent"]
ZERO_DIVISION_POLICY: int = 0


@dataclass
class ConfusionMatrixReport:
    """Confusion matrix decomposition using fixed class ordering."""

    labels: List[str]
    matrix: List[List[int]]
    true_positives: Dict[str, int]
    false_positives: Dict[str, int]
    false_negatives: Dict[str, int]
    true_negatives: Dict[str, int]


@dataclass
class LatencyBenchmarkResult:
    """Structured inference latency benchmark measurement."""

    method: str
    batch_size: int
    warmup_runs: int
    timed_runs: int
    median_batch_seconds: float
    median_ms_per_sample: float
    mean_ms_per_sample: float
    min_ms_per_sample: float
    max_ms_per_sample: float
    std_ms_per_sample: float
    run_batch_seconds: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a classification run."""

    num_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: ConfusionMatrixReport
    inference_latency_ms_per_sample: float
    total_inference_seconds: float
    zero_division_policy: int = ZERO_DIVISION_POLICY
    label_order: List[str] = field(default_factory=lambda: list(LABEL_ORDER))

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
    """Compute standard classification metrics with consistent class ordering.

    Zero-division behavior:
        All precision, recall, and F1 computations use ``zero_division=0``,
        so classes receiving zero predictions yield 0.0 precision/F1 rather
        than raising warnings or undefined values.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Fixed label ordering (default: ['consistent', 'inconsistent']).
        latency_ms_per_sample: Benchmarked median inference latency in ms/sample.
        total_latency_seconds: Benchmarked median batch duration in seconds.

    Returns:
        EvaluationReport instance.
    """
    labels_list = list(labels)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels_list,
            average="macro",
            zero_division=ZERO_DIVISION_POLICY,
        )
    )
    macro_prec = float(
        precision_score(
            y_true,
            y_pred,
            labels=labels_list,
            average="macro",
            zero_division=ZERO_DIVISION_POLICY,
        )
    )
    macro_rec = float(
        recall_score(
            y_true,
            y_pred,
            labels=labels_list,
            average="macro",
            zero_division=ZERO_DIVISION_POLICY,
        )
    )

    per_f1_arr = f1_score(
        y_true, y_pred, labels=labels_list, average=None, zero_division=ZERO_DIVISION_POLICY
    )
    per_prec_arr = precision_score(
        y_true, y_pred, labels=labels_list, average=None, zero_division=ZERO_DIVISION_POLICY
    )
    per_rec_arr = recall_score(
        y_true, y_pred, labels=labels_list, average=None, zero_division=ZERO_DIVISION_POLICY
    )

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
        balanced_accuracy=round(bal_acc, 4),
        macro_f1=round(macro_f1, 4),
        macro_precision=round(macro_prec, 4),
        macro_recall=round(macro_rec, 4),
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        confusion_matrix=cm_report,
        inference_latency_ms_per_sample=round(latency_ms_per_sample, 4),
        total_inference_seconds=round(total_latency_seconds, 4),
        zero_division_policy=ZERO_DIVISION_POLICY,
        label_order=labels_list,
    )


def benchmark_inference_latency(
    predict_fn: Callable[[Any], Sequence[str]],
    inputs: Any,
    warmup_runs: int = 2,
    timed_runs: int = 7,
) -> Tuple[Sequence[str], LatencyBenchmarkResult]:
    """Benchmark full-batch inference latency with warm-up and repeated timing.

    Protocol:
        Runs ``warmup_runs`` unmeasured passes on the full input batch, then
        executes ``timed_runs`` repeated passes on the identical batch using
        ``time.perf_counter()``. Reports the median amortized per-sample latency
        (``median_batch_seconds / batch_size * 1000``) without mixing single-sample
        and batch-throughput latency definitions.

    Args:
        predict_fn: Callable accepting ``inputs`` and returning predictions.
        inputs: Full evaluation batch (e.g., validation DataFrame).
        warmup_runs: Number of unmeasured warm-up passes.
        timed_runs: Number of timed passes (median reported).

    Returns:
        Tuple of (predictions, LatencyBenchmarkResult).
    """
    n_samples = max(len(inputs), 1)

    for _ in range(max(warmup_runs, 0)):
        _ = predict_fn(inputs)

    run_durations: List[float] = []
    preds: Sequence[str] = []
    for _ in range(max(timed_runs, 1)):
        t0 = time.perf_counter()
        preds = predict_fn(inputs)
        elapsed = time.perf_counter() - t0
        run_durations.append(elapsed)

    per_sample_ms = [(d / n_samples) * 1000.0 for d in run_durations]
    median_batch = float(np.median(run_durations))
    median_ms = float(np.median(per_sample_ms))

    result = LatencyBenchmarkResult(
        method="full_validation_batch_amortized_median",
        batch_size=len(inputs),
        warmup_runs=warmup_runs,
        timed_runs=max(timed_runs, 1),
        median_batch_seconds=round(median_batch, 6),
        median_ms_per_sample=round(median_ms, 6),
        mean_ms_per_sample=round(float(np.mean(per_sample_ms)), 6),
        min_ms_per_sample=round(float(np.min(per_sample_ms)), 6),
        max_ms_per_sample=round(float(np.max(per_sample_ms)), 6),
        std_ms_per_sample=round(float(np.std(per_sample_ms)), 6),
        run_batch_seconds=[round(d, 6) for d in run_durations],
    )
    return preds, result


def measure_inference_latency(
    predict_fn: Callable[[Any], Sequence[str]],
    inputs: Any,
    num_runs: int = 5,
    warmup_runs: int = 1,
) -> Tuple[Sequence[str], float, float]:
    """Benchmark prediction function latency returning median amortized ms/sample.

    Args:
        predict_fn: Callable that accepts inputs and returns predictions.
        inputs: Batched or full inputs.
        num_runs: Repetitions for stable median timing.
        warmup_runs: Unmeasured warm-up passes.

    Returns:
        Tuple of (predictions, median_latency_ms_per_sample, median_batch_seconds).
    """
    preds, bench = benchmark_inference_latency(
        predict_fn=predict_fn,
        inputs=inputs,
        warmup_runs=warmup_runs,
        timed_runs=num_runs,
    )
    return preds, bench.median_ms_per_sample, bench.median_batch_seconds


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
