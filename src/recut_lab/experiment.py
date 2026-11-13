"""Reproducible experiment tracking, environment metadata logging, and artifact management."""

from __future__ import annotations

import datetime
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    """Set global random seed for python random, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_metadata() -> Dict[str, Any]:
    """Capture environment attributes for scientific reproducibility."""
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "os": os.name,
        "cpu_count": os.cpu_count(),
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

    def save_run_metrics(
        self,
        run_dir: Path | str,
        metrics: Dict[str, Any],
        also_save_to_reports: bool = True,
        report_filename: Optional[str] = None,
    ) -> None:
        """Persist metrics dictionary to run directory and optionally reports/generated.

        Args:
            run_dir: Path to the experiment run folder.
            metrics: Evaluated metrics.
            also_save_to_reports: If True, writes a copy to reports/generated.
            report_filename: Optional report filename.
        """
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
    ) -> Path:
        """Save predictions without committing raw source text (preserves indices & labels)."""
        target_dir = Path(run_dir)
        pred_path = target_dir / "predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        return pred_path
