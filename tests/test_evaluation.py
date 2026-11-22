"""Unit tests for metric computation, label ordering, confusion matrices, and latency benchmarking."""

import pytest

from recut_lab.evaluation import (
    LABEL_ORDER,
    benchmark_inference_latency,
    compute_classification_metrics,
    evaluate_predictions,
    measure_inference_latency,
)


def test_compute_classification_metrics_perfect():
    y_true = ["consistent", "consistent", "inconsistent", "inconsistent"]
    y_pred = ["consistent", "consistent", "inconsistent", "inconsistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.label_order == LABEL_ORDER
    assert rep.accuracy == 1.0
    assert rep.balanced_accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.macro_precision == 1.0
    assert rep.macro_recall == 1.0
    assert rep.per_class_f1["consistent"] == 1.0
    assert rep.per_class_f1["inconsistent"] == 1.0
    assert rep.confusion_matrix.labels == ["consistent", "inconsistent"]
    assert rep.confusion_matrix.matrix == [[2, 0], [0, 2]]


def test_compute_classification_metrics_mixed_and_confusion_matrix():
    # Label order: 0 = consistent, 1 = inconsistent
    # True consistent (3): 2 predicted consistent, 1 predicted inconsistent
    # True inconsistent (2): 1 predicted consistent, 1 predicted inconsistent
    y_true = ["consistent", "consistent", "consistent", "inconsistent", "inconsistent"]
    y_pred = ["consistent", "consistent", "inconsistent", "consistent", "inconsistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.num_samples == 5
    assert rep.accuracy == 0.6
    # Recall consistent = 2/3, Recall inconsistent = 1/2 -> balanced_accuracy = (2/3 + 1/2)/2 = 7/12 = 0.5833
    assert rep.balanced_accuracy == pytest.approx(0.5833, abs=1e-4)
    assert rep.per_class_recall["consistent"] == pytest.approx(0.6667, abs=1e-4)
    assert rep.per_class_recall["inconsistent"] == pytest.approx(0.5, abs=1e-4)
    assert rep.per_class_precision["inconsistent"] == pytest.approx(0.5, abs=1e-4)
    assert rep.per_class_f1["inconsistent"] == pytest.approx(0.5, abs=1e-4)
    assert rep.confusion_matrix.matrix == [[2, 1], [1, 1]]
    assert rep.confusion_matrix.true_positives["inconsistent"] == 1
    assert rep.confusion_matrix.false_positives["inconsistent"] == 1
    assert rep.confusion_matrix.false_negatives["inconsistent"] == 1
    assert rep.confusion_matrix.true_negatives["inconsistent"] == 2


def test_compute_classification_metrics_zero_division():
    # Majority-style prediction: predicts only 'consistent'
    y_true = ["consistent", "consistent", "inconsistent"]
    y_pred = ["consistent", "consistent", "consistent"]

    rep = compute_classification_metrics(y_true, y_pred)
    assert rep.accuracy == pytest.approx(0.6667, abs=1e-4)
    assert rep.balanced_accuracy == 0.5
    assert rep.zero_division_policy == 0
    assert rep.per_class_precision["inconsistent"] == 0.0
    assert rep.per_class_recall["inconsistent"] == 0.0
    assert rep.per_class_f1["inconsistent"] == 0.0


def test_measure_and_benchmark_inference_latency():
    def mock_predict(inputs):
        return ["consistent"] * len(inputs)

    inputs = ["input1", "input2", "input3"]
    preds, latency_ms, total_sec = measure_inference_latency(mock_predict, inputs, num_runs=3)
    assert len(preds) == 3
    assert latency_ms >= 0.0
    assert total_sec >= 0.0

    preds2, bench = benchmark_inference_latency(mock_predict, inputs, warmup_runs=1, timed_runs=3)
    assert len(preds2) == 3
    assert bench.method == "full_validation_batch_amortized_median"
    assert bench.batch_size == 3
    assert bench.timed_runs == 3
    assert len(bench.run_batch_seconds) == 3
    assert bench.median_ms_per_sample >= 0.0


def test_evaluate_predictions_dict():
    out = evaluate_predictions(
        y_true=["consistent", "inconsistent"],
        y_pred=["consistent", "inconsistent"],
        latency_ms=0.1,
        total_time=0.0002,
    )
    assert out["accuracy"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
