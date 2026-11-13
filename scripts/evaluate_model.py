"""Evaluate a trained model checkpoint on a dataset split with strict protocol checks.

Usage:
    python scripts/evaluate_model.py --model models/tfidf_logreg_baseline.pkl [--split valid]
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
from recut_lab.baseline import BaselineModel
from recut_lab.data import load_adtec_dataset
from recut_lab.evaluation import compute_classification_metrics, measure_inference_latency


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to serialized model pickle file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/adtec_consistency",
        help="Directory containing dataset splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Split to evaluate against (default: 'valid').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/generated",
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Explicit confirmation required to evaluate against the frozen test set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "test" and not args.confirm_test:
        print(
            "[!] PROTOCOL WARNING: The official test split must remain frozen until final model selection.\n"
            "    If you are running the final benchmark evaluation, pass '--confirm-test'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[*] Loading model from {model_path}...")
    model = BaselineModel.load(model_path)

    print(f"[*] Loading dataset from {args.data_dir}...")
    dataset = load_adtec_dataset(args.data_dir)
    eval_df = dataset.get_split(args.split)
    print(f"[+] Evaluating on split '{args.split}' ({len(eval_df)} samples)...")

    preds, latency_ms, total_sec = measure_inference_latency(
        predict_fn=model.predict,
        inputs=eval_df,
        num_runs=3,
    )
    probs = model.predict_proba(eval_df)
    y_true = eval_df["label"].tolist()

    report = compute_classification_metrics(
        y_true=y_true,
        y_pred=preds,
        latency_ms_per_sample=latency_ms,
        total_latency_seconds=total_sec,
    )

    # Save output artifacts
    stem = f"{model_path.stem}_{args.split}"
    metrics_path = output_dir / f"{stem}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    pred_df = extract_misclassifications(
        df=eval_df,
        y_true=y_true,
        y_pred=preds,
        probs=probs,
        classes=getattr(model, "classes_", None),
    )
    preds_csv_path = output_dir / f"{stem}_predictions.csv"
    pred_df.to_csv(preds_csv_path, index=False)

    print(f"[+] Evaluation finished successfully!")
    print(f"    - Metrics: {metrics_path}")
    print(f"    - Predictions: {preds_csv_path}")
    print("\n" + "=" * 50)
    print(f"RESULTS: {model_path.name} ON {args.split.upper()} SET")
    print("=" * 50)
    print(f"Accuracy:         {report.accuracy:.4f}")
    print(f"Macro-F1:         {report.macro_f1:.4f}")
    print(f"Macro-Precision:  {report.macro_precision:.4f}")
    print(f"Macro-Recall:     {report.macro_recall:.4f}")
    print(f"Per-Class F1:     {report.per_class_f1}")
    print(f"Latency:          {report.inference_latency_ms_per_sample:.4f} ms/sample")
    print("=" * 50)


if __name__ == "__main__":
    main()

