"""Unit tests for metric computation, latency benchmarking, and error slicing."""

import pytest

from recut_lab.evaluation import (
    compute_classification_metrics,
    evaluate_predictions,
    measure_inference_latency,
)


def test_compute_classification_metrics_perfect():
    y_true = ["consistent", "consistent", "inconsistent", "inconsistent"]
    y_pred = ["consistent", "consistent", "inconsistent", "inconsistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.macro_precision == 1.0
    assert rep.macro_recall == 1.0
    assert rep.per_class_f1["consistent"] == 1.0
    assert rep.per_class_f1["inconsistent"] == 1.0
    assert rep.confusion_matrix.matrix == [[2, 0], [0, 2]]


def test_compute_classification_metrics_mixed():
    # 2 TP consistent, 1 FN (pred inconsistent, true consistent)
    # 1 FP (pred consistent, true inconsistent), 1 TN
    y_true = ["consistent", "consistent", "consistent", "inconsistent", "inconsistent"]
    y_pred = ["consistent", "consistent", "inconsistent", "consistent", "inconsistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.num_samples == 5
    assert rep.accuracy == 0.6
    assert 0.0 < rep.macro_f1 < 1.0
    assert "consistent" in rep.per_class_f1
    assert "inconsistent" in rep.per_class_f1


def test_compute_classification_metrics_zero_division():
    # Model predicts only 'consistent'
    y_true = ["consistent", "inconsistent"]
    y_pred = ["consistent", "consistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.accuracy == 0.5
    # Should not raise ZeroDivisionError
    assert rep.per_class_precision["inconsistent"] == 0.0


def test_measure_inference_latency():
    def mock_predict(inputs):
        return ["consistent"] * len(inputs)

    inputs = ["input1", "input2", "input3"]
    preds, latency_ms, total_sec = measure_inference_latency(mock_predict, inputs, num_runs=2)
    assert len(preds) == 3
    assert latency_ms >= 0.0
    assert total_sec >= 0.0
