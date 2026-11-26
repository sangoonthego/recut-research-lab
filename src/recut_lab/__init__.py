"""Recut Research Lab: Research-first workspace for Advertising Consistency Classification.

This library provides tooling for data loading, auditing, baseline modeling,
evaluation, error analysis, and reproducible experimentation.
"""

from recut_lab.data import (
    ConsistencyDataset,
    load_adtec_dataset,
    load_adtec_train_valid,
    validate_schema,
)
from recut_lab.evaluation import (
    LABEL_ORDER,
    benchmark_inference_latency,
    compute_classification_metrics,
    evaluate_predictions,
)
from recut_lab.baseline import (
    BaselineModel,
    M2_DEFAULT_CONFIG,
    MajorityBaseline,
    TfidfLogRegBaseline,
    run_m2_baseline_benchmark,
)
from recut_lab.experiment import ExperimentManager
from recut_lab.diagnostics import (
    M2_0_1_DEFAULT_CONFIG,
    SeparateBlocksTfidfLogReg,
    TruncatedLpPairBaseline,
    compute_d1_score_diagnostics,
    compute_seen_unseen_confounding,
    extract_class_scores,
    run_m2_failure_diagnostics,
)
from recut_lab.generalization import (
    M2_1_DEFAULT_CONFIG,
    PROTOCOL_A,
    PROTOCOL_B,
    assign_lp_group_ids,
    audit_lp_groups,
    load_adtec_train_only,
    run_rq1_generalization_study,
)

__version__ = "0.1.0"

__all__ = [
    "ConsistencyDataset",
    "load_adtec_dataset",
    "load_adtec_train_valid",
    "load_adtec_train_only",
    "validate_schema",
    "LABEL_ORDER",
    "benchmark_inference_latency",
    "evaluate_predictions",
    "compute_classification_metrics",
    "BaselineModel",
    "M2_DEFAULT_CONFIG",
    "MajorityBaseline",
    "TfidfLogRegBaseline",
    "run_m2_baseline_benchmark",
    "ExperimentManager",
    "M2_0_1_DEFAULT_CONFIG",
    "SeparateBlocksTfidfLogReg",
    "TruncatedLpPairBaseline",
    "compute_d1_score_diagnostics",
    "compute_seen_unseen_confounding",
    "extract_class_scores",
    "run_m2_failure_diagnostics",
    "M2_1_DEFAULT_CONFIG",
    "PROTOCOL_A",
    "PROTOCOL_B",
    "assign_lp_group_ids",
    "audit_lp_groups",
    "run_rq1_generalization_study",
]

