"""Reproducible experiment tracking, environment metadata logging, and artifact management."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TRACKED_LIBRARIES: List[str] = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "nbformat",
    "pytest",
]


def set_global_seed(seed: int = 42) -> None:
    """Set global random seed for python random, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_installed_library_versions(
    packages: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Collect installed versions for key scientific Python dependencies."""
    target_pkgs = packages or TRACKED_LIBRARIES
    versions: Dict[str, str] = {}
    for pkg in target_pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def get_environment_metadata() -> Dict[str, Any]:
    """Capture runtime environment and library versions for scientific reproducibility."""
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "os": os.name,
        "cpu_count": os.cpu_count(),
        "library_versions": get_installed_library_versions(),
    }


class ExperimentManager:
    """Manages local experiment runs, configurations, metadata, and metric persistence."""

    def __init__(
        self,
        base_runs_dir: Path | str = "experiments/runs",
        reports_dir: Path | str = "reports/generated",
    ):
        self.base_runs_dir = Path(base_runs_dir)
        self.reports_dir = Path(reports_dir)
        self.base_runs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Path:
        """Create a new timestamped experiment directory and record config and environment.

        Args:
            experiment_name: Descriptive name of the run.
            config: Full experiment configuration dictionary.
            seed: Random seed applied to the run.

        Returns:
            Path to the newly initialized run directory.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{experiment_name}"
        run_dir = self.base_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        effective_seed = seed if seed is not None else config.get("random_seed", 42)
        set_global_seed(effective_seed)

        metadata = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "random_seed": effective_seed,
            "environment": get_environment_metadata(),
        }

        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return run_dir

    def create_named_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """Initialize a named experiment directory (e.g., 'm2_baseline_v001').

        Writes config.json and environment.json including dataset provenance.
        """
        run_dir = self.base_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        effective_seed = seed if seed is not None else config.get("random_seed", 42)
        set_global_seed(effective_seed)

        env_payload: Dict[str, Any] = {
            "experiment_id": run_id,
            "random_seed": effective_seed,
            "training_split": "train",
            "evaluation_split": "valid",
            "test_split_accessed": False,
            "environment": get_environment_metadata(),
        }

        if dataset_metadata:
            env_payload["dataset_provenance"] = {
                "dataset_name": dataset_metadata.get("dataset_name"),
                "config_name": dataset_metadata.get("config_name"),
                "license": dataset_metadata.get("license"),
                "upstream_revision": dataset_metadata.get("upstream_revision"),
                "download_date_utc": dataset_metadata.get("download_date_utc"),
                "split_hashes": {
                    split: info.get("sha256")
                    for split, info in dataset_metadata.get("splits", {}).items()
                },
            }

        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        with (run_dir / "environment.json").open("w", encoding="utf-8") as f:
            json.dump(env_payload, f, indent=2)

        return run_dir

    def save_run_metrics(
        self,
        run_dir: Path | str,
        metrics: Dict[str, Any],
        also_save_to_reports: bool = True,
        report_filename: Optional[str] = None,
    ) -> None:
        """Persist metrics dictionary to run directory and optionally reports/generated."""
        target_dir = Path(run_dir)
        with (target_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if also_save_to_reports:
            name = report_filename or f"{target_dir.name}_metrics.json"
            rep_path = self.reports_dir / name
            with rep_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    def save_run_predictions(
        self,
        run_dir: Path | str,
        predictions_df: pd.DataFrame,
        filename: str = "predictions.csv",
    ) -> Path:
        """Save predictions without committing raw source text (preserves indices & labels)."""
        target_dir = Path(run_dir)
        pred_path = target_dir / filename
        predictions_df.to_csv(pred_path, index=False)
        return pred_path
