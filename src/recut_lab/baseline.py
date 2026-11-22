"""Baseline models and M2.0 benchmark runner for advertising consistency classification.

Implements four CPU-compatible baseline configurations:
- B0_majority: DummyClassifier(strategy="most_frequent")
- B1_title_only: Title-only character n-gram TF-IDF + Logistic Regression
- B2_lp_only: Landing-page-only character n-gram TF-IDF + Logistic Regression
- B3_pair: Concatenated pair (title + [SEP] + lp_text) TF-IDF + Logistic Regression
"""

from __future__ import annotations

import json
import pickle
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from recut_lab.data import compute_file_sha256
from recut_lab.evaluation import (
    LABEL_ORDER,
    ZERO_DIVISION_POLICY,
    benchmark_inference_latency,
    compute_classification_metrics,
)
from recut_lab.experiment import ExperimentManager, set_global_seed
from recut_lab.features import build_char_tfidf_vectorizer
from recut_lab.preprocessing import TextInputSelector

M2_DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_id": "m2_baseline_v001",
    "random_seed": 42,
    "data": {
        "dataset_name": "cyberagent/AdTEC",
        "config_name": "ad-consistency",
        "raw_dir": "data/raw/adtec_consistency",
        "train_split": "train",
        "eval_split": "valid",
        "test_split_protected": True,
        "separator": " [SEP] ",
        "normalize_unicode_nfkc": True,
    },
    "vectorizer": {
        "analyzer": "char",
        "ngram_range": [2, 4],
        "max_features": 50000,
        "min_df": 2,
        "sublinear_tf": True,
    },
    "classifier": {
        "model_family": "LogisticRegression",
        "C": 1.0,
        "class_weight": None,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "latency_benchmark": {
        "warmup_runs": 2,
        "timed_runs": 7,
        "method": "full_validation_batch_amortized_median",
    },
    "label_order": list(LABEL_ORDER),
    "zero_division_policy": ZERO_DIVISION_POLICY,
    "baselines": [
        {
            "model_id": "B0_majority",
            "model_type": "majority",
            "input_features": "none (training label prior)",
            "input_mode": "none",
        },
        {
            "model_id": "B1_title_only",
            "model_type": "tfidf_logreg",
            "input_features": "title",
            "input_mode": "title",
        },
        {
            "model_id": "B2_lp_only",
            "model_type": "tfidf_logreg",
            "input_features": "lp_text",
            "input_mode": "lp_text",
        },
        {
            "model_id": "B3_pair",
            "model_type": "tfidf_logreg",
            "input_features": "title + [SEP] + lp_text",
            "input_mode": "pair",
        },
    ],
}


class BaselineModel(ABC):
    """Abstract interface for recut_lab baseline classifiers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "label") -> BaselineModel:
        """Fit model exclusively on training DataFrame."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> List[str]:
        """Generate predicted string labels."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate raw model probability scores if supported."""
        pass

    def save(self, file_path: Path | str) -> None:
        """Serialize model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path | str) -> BaselineModel:
        """Load serialized model from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path.resolve()}")
        with path.open("rb") as f:
            model = pickle.load(f)
        return model


class MajorityBaseline(BaselineModel):
    """B0: Baseline predicting the most frequent class observed in training data."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.dummy_clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
        self.pipeline: Pipeline = Pipeline([("clf", self.dummy_clf)])
        self.classes_: Optional[np.ndarray] = None
        self.majority_label_: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> MajorityBaseline:
        y = df[target_col].to_numpy()
        X_dummy = np.zeros((len(df), 1))
        self.pipeline.fit(X_dummy, y)
        self.classes_ = self.dummy_clf.classes_
        self.majority_label_ = str(self.pipeline.predict(np.zeros((1, 1)))[0])
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        if self.majority_label_ is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.pipeline.predict(np.zeros((len(df), 1)))
        return [str(p) for p in preds]

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.majority_label_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.pipeline.predict_proba(np.zeros((len(df), 1)))


class TfidfLogRegBaseline(BaselineModel):
    """Character n-gram TF-IDF Vectorizer + Logistic Regression inside a scikit-learn Pipeline.

    Supports three input modes:
        - ``"title"`` (B1 diagnostic control)
        - ``"lp_text"`` (B2 diagnostic control)
        - ``"pair"`` (B3 main baseline using deterministic separator)
    """

    def __init__(
        self,
        input_mode: str = "pair",
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 2,
        max_features: Optional[int] = 50000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        class_weight: Optional[str] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        title_col: str = "title",
        lp_col: str = "lp_text",
        separator: str = " [SEP] ",
        normalize: bool = True,
    ):
        self.input_mode = input_mode
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.title_col = title_col
        self.lp_col = lp_col
        self.separator = separator
        self.normalize = normalize

        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        selector = TextInputSelector(
            mode=self.input_mode,
            title_col=self.title_col,
            lp_col=self.lp_col,
            separator=self.separator,
            normalize=self.normalize,
        )
        vectorizer = build_char_tfidf_vectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        classifier = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        return Pipeline([
            ("selector", selector),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> TfidfLogRegBaseline:
        self.pipeline = self._build_pipeline()
        y = df[target_col].to_numpy()
        self.pipeline.fit(df, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.pipeline.predict(df)
        return [str(p) for p in preds]

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.pipeline.predict_proba(df)

    def get_vocabulary_size(self) -> int:
        """Return size of fitted TF-IDF character n-gram vocabulary."""
        if self.pipeline is None:
            return 0
        return len(self.pipeline.named_steps["tfidf"].vocabulary_)


def get_baseline_model(model_type: str, **kwargs) -> BaselineModel:
    """Factory function for instantiating baseline models."""
    normalized = model_type.lower().strip()
    if normalized in ("b0", "b0_majority", "majority", "majority_class", "dummy"):
        return MajorityBaseline(random_state=kwargs.get("random_state", 42))
    if normalized in ("b1", "b1_title_only", "title_only"):
        kwargs = {**kwargs, "input_mode": "title"}
        return TfidfLogRegBaseline(**kwargs)
    if normalized in ("b2", "b2_lp_only", "lp_only"):
        kwargs = {**kwargs, "input_mode": "lp_text"}
        return TfidfLogRegBaseline(**kwargs)
    if normalized in ("b3", "b3_pair", "tfidf_logreg", "tfidf_lr", "logistic_regression", "tfidf"):
        kwargs.setdefault("input_mode", "pair")
        return TfidfLogRegBaseline(**kwargs)
    raise ValueError(
        f"Unknown baseline model type: '{model_type}'. "
        "Expected 'majority', 'tfidf_logreg', 'B0_majority', 'B1_title_only', "
        "'B2_lp_only', or 'B3_pair'."
    )


def _extract_aligned_scores(
    probs: Optional[np.ndarray],
    classes: Optional[np.ndarray],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Extract per-class raw model scores aligned to LABEL_ORDER."""
    if probs is None or classes is None:
        return [], []
    class_list = [str(c) for c in classes]
    idx_c = class_list.index("consistent") if "consistent" in class_list else None
    idx_i = class_list.index("inconsistent") if "inconsistent" in class_list else None

    scores_c = [
        round(float(probs[r, idx_c]), 6) if idx_c is not None else None
        for r in range(len(probs))
    ]
    scores_i = [
        round(float(probs[r, idx_i]), 6) if idx_i is not None else None
        for r in range(len(probs))
    ]
    return scores_c, scores_i


def format_benchmark_summary_markdown(
    benchmark_df: pd.DataFrame,
    metrics_payload: Dict[str, Any],
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Render M2.0 benchmark summary report in GitHub Flavored Markdown."""
    best_row = benchmark_df.sort_values("macro_f1", ascending=False).iloc[0]
    revision = (dataset_metadata or {}).get("upstream_revision", "unknown")
    latency_proto = metrics_payload.get("latency_protocol", {})

    lines = [
        "# Milestone M2.0 — Baseline Benchmark Summary",
        "",
        f"**Experiment ID**: `{metrics_payload.get('experiment_id', 'm2_baseline_v001')}`  ",
        f"**Dataset**: CyberAgent AdTEC (`ad-consistency`, revision `{revision}`)  ",
        "**Evaluation Split**: Official `valid` split only (945 samples); `test` split strictly untouched.  ",
        f"**Random Seed**: `{metrics_payload.get('random_seed', 42)}`  ",
        f"**Label Order**: `{metrics_payload.get('label_order', LABEL_ORDER)}`  ",
        f"**Zero-Division Policy**: `zero_division={metrics_payload.get('zero_division_policy', 0)}`  ",
        "",
        "---",
        "",
        "## 1. Main Validation Benchmark Table",
        "",
        "| model_id | input_features | accuracy | balanced_accuracy | macro_f1 | precision_inconsistent | recall_inconsistent | f1_inconsistent | train_time_seconds | inference_latency_ms | model_size_mb |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in benchmark_df.iterrows():
        lines.append(
            f"| `{row['model_id']}` | {row['input_features']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"**{row['macro_f1']:.4f}** | {row['precision_inconsistent']:.4f} | "
            f"{row['recall_inconsistent']:.4f} | {row['f1_inconsistent']:.4f} | "
            f"{row['train_time_seconds']:.4f} | {row['inference_latency_ms']:.4f} | "
            f"{row['model_size_mb']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Key Observations [VERIFIED]",
        "",
        f"- **Highest Observed Validation Macro-F1**: `{best_row['model_id']}` achieved Macro-F1 = **{best_row['macro_f1']:.4f}** "
        f"(Accuracy = {best_row['accuracy']:.4f}, Balanced Accuracy = {best_row['balanced_accuracy']:.4f}, "
        f"Inconsistent Recall = {best_row['recall_inconsistent']:.4f}).",
        "- **Latency Measurement Protocol**: "
        f"`{latency_proto.get('method', 'full_validation_batch_amortized_median')}` "
        f"(batch_size={latency_proto.get('batch_size', 945)}, "
        f"warmup_runs={latency_proto.get('warmup_runs', 2)}, "
        f"timed_runs={latency_proto.get('timed_runs', 7)}).",
        "- **Calibration Notice**: Raw `predict_proba` outputs are uncalibrated model scores and must not be interpreted as calibrated probabilities without dedicated calibration assessment.",
        "- **Scope Boundary**: These measurements are initial default-configuration baselines on the validation split only; they do not establish universal optimality, test-set generalization, or RQ1–RQ3 conclusions.",
    ])

    return "\n".join(lines)


def run_m2_baseline_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    runs_base_dir: Path | str = "experiments/runs",
    models_base_dir: Path | str = "models",
    reports_dir: Path | str = "reports/generated/m2_baseline",
) -> Dict[str, Any]:
    """Execute the full M2.0 four-baseline benchmark (B0–B3) reproducibly.

    Fits all pipelines exclusively on ``train_df``, evaluates exclusively on
    ``valid_df``, persists artifacts without raw source texts, and returns
    in-memory results for downstream visualization and inspection.

    Args:
        train_df: Official training split DataFrame.
        valid_df: Official validation split DataFrame.
        dataset_metadata: Provenance dict loaded from M1 metadata.json.
        config: Benchmark configuration dictionary (defaults to M2_DEFAULT_CONFIG).
        runs_base_dir: Parent directory for experiment runs.
        models_base_dir: Parent directory for serialized model artifacts.
        reports_dir: Output directory for M2.0 reports and figures.

    Returns:
        Dictionary containing benchmark_df, metrics, predictions_df, manifest,
        models, run_dir, and reports_dir.
    """
    cfg = deepcopy(config) if config is not None else deepcopy(M2_DEFAULT_CONFIG)
    experiment_id = cfg.get("experiment_id", "m2_baseline_v001")
    seed = int(cfg.get("random_seed", 42))

    exp_mgr = ExperimentManager(base_runs_dir=runs_base_dir, reports_dir=reports_dir)
    run_dir = exp_mgr.create_named_run(
        run_id=experiment_id,
        config=cfg,
        dataset_metadata=dataset_metadata,
        seed=seed,
    )

    models_dir = Path(models_base_dir) / experiment_id
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    vec_cfg = cfg.get("vectorizer", {})
    clf_cfg = cfg.get("classifier", {})
    data_cfg = cfg.get("data", {})
    lat_cfg = cfg.get("latency_benchmark", {})

    warmup_runs = int(lat_cfg.get("warmup_runs", 2))
    timed_runs = int(lat_cfg.get("timed_runs", 7))
    y_valid_true = valid_df["label"].tolist()

    benchmark_rows: List[Dict[str, Any]] = []
    per_model_metrics: Dict[str, Any] = {}
    prediction_frames: List[pd.DataFrame] = []
    model_manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "random_seed": seed,
        "models": {},
    }
    fitted_models: Dict[str, BaselineModel] = {}

    for spec in cfg.get("baselines", M2_DEFAULT_CONFIG["baselines"]):
        model_id = spec["model_id"]
        model_type = spec["model_type"]
        input_features = spec["input_features"]
        input_mode = spec.get("input_mode", "pair")

        set_global_seed(seed)

        if model_type == "majority":
            model: BaselineModel = MajorityBaseline(random_state=seed)
        else:
            model = TfidfLogRegBaseline(
                input_mode=input_mode,
                ngram_range=tuple(vec_cfg.get("ngram_range", [2, 4])),
                min_df=int(vec_cfg.get("min_df", 2)),
                max_features=vec_cfg.get("max_features", 50000),
                sublinear_tf=bool(vec_cfg.get("sublinear_tf", True)),
                c_param=float(clf_cfg.get("C", 1.0)),
                class_weight=clf_cfg.get("class_weight", None),
                max_iter=int(clf_cfg.get("max_iter", 1000)),
                random_state=seed,
                solver=str(clf_cfg.get("solver", "lbfgs")),
                separator=str(data_cfg.get("separator", " [SEP] ")),
                normalize=bool(data_cfg.get("normalize_unicode_nfkc", True)),
            )

        # Fit exclusively on training split
        t_fit_start = time.perf_counter()
        model.fit(train_df, target_col="label")
        train_time_sec = time.perf_counter() - t_fit_start

        # Persist model checkpoint to dedicated M2.0 subfolder (never overwriting older models)
        model_path = models_dir / f"{model_id}.pkl"
        model.save(model_path)
        model_size_bytes = model_path.stat().st_size
        model_size_mb = model_size_bytes / (1024.0 * 1024.0)
        model_sha256 = compute_file_sha256(model_path)

        # Benchmark inference latency on validation split exclusively
        preds, latency_res = benchmark_inference_latency(
            predict_fn=model.predict,
            inputs=valid_df,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )
        probs = model.predict_proba(valid_df)

        eval_report = compute_classification_metrics(
            y_true=y_valid_true,
            y_pred=preds,
            labels=LABEL_ORDER,
            latency_ms_per_sample=latency_res.median_ms_per_sample,
            total_latency_seconds=latency_res.median_batch_seconds,
        )

        vocab_size = (
            model.get_vocabulary_size()
            if isinstance(model, TfidfLogRegBaseline)
            else 0
        )

        row = {
            "model_id": model_id,
            "input_features": input_features,
            "accuracy": eval_report.accuracy,
            "balanced_accuracy": eval_report.balanced_accuracy,
            "macro_f1": eval_report.macro_f1,
            "precision_inconsistent": eval_report.per_class_precision["inconsistent"],
            "recall_inconsistent": eval_report.per_class_recall["inconsistent"],
            "f1_inconsistent": eval_report.per_class_f1["inconsistent"],
            "train_time_seconds": round(train_time_sec, 4),
            "inference_latency_ms": round(latency_res.median_ms_per_sample, 4),
            "model_size_mb": round(model_size_mb, 4),
        }
        benchmark_rows.append(row)

        per_model_metrics[model_id] = {
            "model_id": model_id,
            "input_features": input_features,
            "input_mode": input_mode,
            "train_time_seconds": round(train_time_sec, 4),
            "model_size_mb": round(model_size_mb, 4),
            "model_size_bytes": model_size_bytes,
            "vocabulary_size": vocab_size,
            "evaluation": eval_report.to_dict(),
            "latency_benchmark": latency_res.to_dict(),
        }

        # Prediction records (NO raw Japanese source text persisted in CSV)
        scores_c, scores_i = _extract_aligned_scores(probs, getattr(model, "classes_", None))
        pred_records = []
        for idx_pos, (yt, yp) in enumerate(zip(y_valid_true, preds)):
            sample_idx = int(valid_df.index[idx_pos])
            if yp == "inconsistent" and yt == "consistent":
                err_type = "FP_inconsistent"
            elif yp == "consistent" and yt == "inconsistent":
                err_type = "FN_inconsistent"
            else:
                err_type = "CORRECT"
            pred_records.append({
                "sample_id": f"valid_{sample_idx:04d}",
                "valid_row_index": sample_idx,
                "model_id": model_id,
                "true_label": yt,
                "predicted_label": yp,
                "score_consistent": scores_c[idx_pos] if scores_c else None,
                "score_inconsistent": scores_i[idx_pos] if scores_i else None,
                "is_error": yt != yp,
                "error_type": err_type,
            })
        prediction_frames.append(pd.DataFrame(pred_records))

        model_manifest["models"][model_id] = {
            "model_id": model_id,
            "model_type": model_type,
            "input_mode": input_mode,
            "input_features": input_features,
            "artifact_path": str(model_path.as_posix()),
            "sha256": model_sha256,
            "file_size_bytes": model_size_bytes,
            "file_size_mb": round(model_size_mb, 4),
            "vocabulary_size": vocab_size,
        }
        fitted_models[model_id] = model

    benchmark_df = pd.DataFrame(benchmark_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "random_seed": seed,
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_split_accessed": False,
        "label_order": list(LABEL_ORDER),
        "zero_division_policy": ZERO_DIVISION_POLICY,
        "latency_protocol": {
            "method": "full_validation_batch_amortized_median",
            "batch_size": len(valid_df),
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
        },
        "models": per_model_metrics,
    }

    # Persist run artifacts under experiments/runs/<experiment_id>/
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    benchmark_df.to_csv(run_dir / "benchmark_comparison.csv", index=False)
    predictions_df.to_csv(run_dir / "predictions_valid.csv", index=False)

    with (run_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)

    # Persist summary artifacts under reports/generated/m2_baseline/
    summary_md = format_benchmark_summary_markdown(
        benchmark_df=benchmark_df,
        metrics_payload=metrics_payload,
        dataset_metadata=dataset_metadata,
    )
    with (reports_path / "benchmark_summary.md").open("w", encoding="utf-8") as f:
        f.write(summary_md)
    benchmark_df.to_csv(reports_path / "benchmark_comparison.csv", index=False)
    with (reports_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return {
        "benchmark_df": benchmark_df,
        "metrics": metrics_payload,
        "predictions_df": predictions_df,
        "manifest": model_manifest,
        "models": fitted_models,
        "run_dir": run_dir,
        "reports_dir": reports_path,
        "summary_md": summary_md,
    }
