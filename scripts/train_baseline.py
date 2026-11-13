"""Train baseline models on AdTEC ad-consistency training split and evaluate on validation split.

Usage:
    python scripts/train_baseline.py [--config configs/baseline.json] [--model-type tfidf_logreg]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recut_lab.analysis import extract_misclassifications
from recut_lab.baseline import get_baseline_model
from recut_lab.data import load_adtec_dataset
from recut_lab.evaluation import compute_classification_metrics, measure_inference_latency
from recut_lab.experiment import ExperimentManager, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline consistency models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.json",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override model type ('majority' or 'tfidf_logreg').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory (defaults to config or data/raw/adtec_consistency).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model pickle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"[!] Warning: Config file '{config_path}' not found. Using defaults.")
        config = {
            "experiment_name": "baseline_tfidf_logreg",
            "model_type": "tfidf_logreg",
            "random_seed": 42,
            "data": {"raw_dir": "data/raw/adtec_consistency"},
        }

    model_type = args.model_type or config.get("model_type", "tfidf_logreg")
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    data_dir = args.data_dir or config.get("data", {}).get("raw_dir", "data/raw/adtec_consistency")

    set_global_seed(seed)
    exp_mgr = ExperimentManager()
    run_dir = exp_mgr.create_run(
        experiment_name=f"{model_type}_eval",
        config=config,
        seed=seed,
    )

    print(f"[*] Initialized experiment run at {run_dir}")
    print(f"[*] Loading dataset from {data_dir}...")
    try:
        dataset = load_adtec_dataset(data_dir)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        print("Run 'python scripts/download_adtec.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Loaded splits: train ({len(dataset.train)}), valid ({len(dataset.valid)}), test ({len(dataset.test)})")
    print("    [PROTOCOL CHECK] Training exclusively on train split. Test split remains untouched.")

    # Instantiate model
    print(f"[*] Building {model_type} baseline model...")
    if model_type == "majority":
        model = get_baseline_model("majority")
    else:
        vec_cfg = config.get("vectorizer", {})
        clf_cfg = config.get("classifier", {})
        model = get_baseline_model(
            "tfidf_logreg",
            ngram_range=tuple(vec_cfg.get("ngram_range", [1, 3])),
            min_df=vec_cfg.get("min_df", 2),
            max_features=vec_cfg.get("max_features", 25000),
            sublinear_tf=vec_cfg.get("sublinear_tf", True),
            c_param=clf_cfg.get("C", 1.0),
            class_weight=clf_cfg.get("class_weight", "balanced"),
            max_iter=clf_cfg.get("max_iter", 1000),
            random_state=seed,
            solver=clf_cfg.get("solver", "lbfgs"),
        )

    # Train model
    print(f"[*] Fitting {model_type} model on training set ({len(dataset.train)} samples)...")
    model.fit(dataset.train, target_col="label")
    print("[+] Model fitting complete.")

    # Benchmark inference on validation split
    print(f"[*] Evaluating on validation split ({len(dataset.valid)} samples)...")
    preds, latency_ms, total_sec = measure_inference_latency(
        predict_fn=model.predict,
        inputs=dataset.valid,
        num_runs=3,
    )
    probs = model.predict_proba(dataset.valid)

    # Compute evaluation metrics
    y_valid_true = dataset.valid["label"].tolist()
    report = compute_classification_metrics(
        y_true=y_valid_true,
        y_pred=preds,
        latency_ms_per_sample=latency_ms,
        total_latency_seconds=total_sec,
    )
    metrics_dict = report.to_dict()

    # Save metrics to run directory and reports/generated
    exp_mgr.save_run_metrics(
        run_dir=run_dir,
        metrics=metrics_dict,
        also_save_to_reports=True,
        report_filename=f"{model_type}_validation_metrics.json",
    )

    # Save error analysis and predictions without raw corpus text
    classes = getattr(model, "classes_", ["consistent", "inconsistent"])
    pred_df = extract_misclassifications(
        df=dataset.valid,
        y_true=y_valid_true,
        y_pred=preds,
        probs=probs,
        classes=classes,
    )
    exp_mgr.save_run_predictions(run_dir=run_dir, predictions_df=pred_df)

    # Save model artifact
    save_path = Path(args.save_model) if args.save_model else Path(f"models/{model_type}_baseline.pkl")
    model.save(save_path)
    print(f"[+] Saved model checkpoint to {save_path.resolve()}")

    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY ({model_type.upper()} on VALIDATION SPLIT)")
    print("=" * 50)
    print(f"Samples:          {report.num_samples}")
    print(f"Accuracy:         {report.accuracy:.4f}")
    print(f"Macro-F1:         {report.macro_f1:.4f}")
    print(f"Macro-Precision:  {report.macro_precision:.4f}")
    print(f"Macro-Recall:     {report.macro_recall:.4f}")
    print(f"Per-Class F1:     {report.per_class_f1}")
    print(f"Latency:          {report.inference_latency_ms_per_sample:.4f} ms/sample")
    print(f"Confusion Matrix: {report.confusion_matrix.matrix} (Labels: {report.confusion_matrix.labels})")
    print("=" * 50)


if __name__ == "__main__":
    main()

