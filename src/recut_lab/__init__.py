"""Recut Research Lab: Research-first workspace for Advertising Consistency Classification.

This library provides tooling for data loading, auditing, baseline modeling,
evaluation, error analysis, and reproducible experimentation.
"""

from recut_lab.data import ConsistencyDataset, load_adtec_dataset, validate_schema
from recut_lab.evaluation import evaluate_predictions, compute_classification_metrics
from recut_lab.baseline import BaselineModel, MajorityBaseline, TfidfLogRegBaseline
from recut_lab.experiment import ExperimentManager

__version__ = "0.1.0"

__all__ = [
    "ConsistencyDataset",
    "load_adtec_dataset",
    "validate_schema",
    "evaluate_predictions",
    "compute_classification_metrics",
    "BaselineModel",
    "MajorityBaseline",
    "TfidfLogRegBaseline",
    "ExperimentManager",
]
