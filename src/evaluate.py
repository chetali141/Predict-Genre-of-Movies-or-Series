"""
evaluate.py - Evaluate a trained model.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)

from utils import load_artifact, save_artifact, ensure_dirs

OUTPUTS_DIR = "outputs"
PLOTS_DIR   = "plots"
RESULTS_STORE = os.path.join(OUTPUTS_DIR, "eval_results.pkl")

def load_test_data():
    """Load the held-out test split from disk."""
    X_test = load_npz(os.path.join(OUTPUTS_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(OUTPUTS_DIR, "y_test.npy"))
    mlb    = load_artifact(os.path.join(OUTPUTS_DIR, "mlb.pkl"))
    print(f"[evaluate] Loaded X_test {X_test.shape}, y_test {y_test.shape}")
    return X_test, y_test, mlb


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute all four evaluation metrics.

    - Sample F1     : partial credit; best for multi-label overall performance
    - Hamming Loss  : fraction of individual label predictions that are wrong
                      (lower is better)
    - Subset Acc    : exact match; all labels must be correct (strictest metric)
    """
    return {
        "Sample F1"   : f1_score(y_true, y_pred, average="samples",  zero_division=0),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Subset Acc"  : accuracy_score(y_true, y_pred),
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    """Pretty-print the metric summary."""
    print(f"\n{'='*50}")
    print(f"  Metrics — {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        direction = "(lower is better)" if k == "Hamming Loss" else ""
        print(f"  {k:<15}: {v:.4f}  {direction}")
    print(f"{'='*50}\n")


def plot_model_metrics(metrics: dict, model_name: str) -> None:
    """Bar chart of all metrics for a single model."""
    ensure_dirs(PLOTS_DIR)
    names  = list(metrics.keys())
    values = list(metrics.values())

    colors = ["#e74c3c" if n == "Hamming Loss" else "#2ecc71" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Evaluation Metrics — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"eval_{model_name}_metrics.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[evaluate] Saved → {path}")


def plot_comparison(all_results: dict) -> None:
    """
    Grouped bar chart comparing all evaluated models side-by-side.
    all_results = { model_name: { metric: value, ... }, ... }
    """
    ensure_dirs(PLOTS_DIR)

    # Only show upward metrics in comparison (exclude Hamming Loss)
    compare_metrics = ["Sample F1", "Macro F1", "Micro F1", "Subset Acc"]
    models = list(all_results.keys())
    x = np.arange(len(compare_metrics))
    width = 0.22
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(models):
        vals = [all_results[model].get(m, 0) for m in compare_metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[i % len(colors)], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(compare_metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (all metrics)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "eval_comparison.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[evaluate] Comparison chart saved → {path}")


def evaluate_model(model_path: str, model_name: str) -> None:
    ensure_dirs(OUTPUTS_DIR, PLOTS_DIR)

    model = load_artifact(model_path)
    X_test, y_test, mlb = load_test_data()

    print(f"[evaluate] Predicting with {model_name} ...")
    y_pred = model.predict(X_test)

    # Per-label report
    print(f"\n[evaluate] Classification Report — {model_name}")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

    # Summary metrics
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, model_name)

    # Single-model plot
    plot_model_metrics(metrics, model_name)

    # Persist result for comparison later
    try:
        all_results = load_artifact(RESULTS_STORE)
    except FileNotFoundError:
        all_results = {}
    all_results[model_name] = metrics
    save_artifact(all_results, RESULTS_STORE)
    print(f"[evaluate] Results stored in {RESULTS_STORE}")


def compare_models() -> None:
    """Load all stored evaluation results and produce the comparison chart."""
    try:
        all_results = load_artifact(RESULTS_STORE)
    except FileNotFoundError:
        print("[evaluate] No stored results found. Run evaluate_model first.")
        return
    if not all_results:
        print("[evaluate] Results store is empty.")
        return

    # Print summary table
    df = pd.DataFrame(all_results).T
    print("\n[evaluate] ── Summary Table ──")
    print(df.to_string(float_format="{:.4f}".format))

    plot_comparison(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained genre classifier.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model .pkl file (e.g. outputs/model_lr.pkl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Short label for this model (e.g. lr, svm, rf).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison chart from all previously evaluated models.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models()
    elif args.model_path and args.model_name:
        evaluate_model(args.model_path, args.model_name)
    else:
        parser.print_help()
