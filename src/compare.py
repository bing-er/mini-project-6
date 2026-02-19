"""
compare.py — Feature Extraction vs Fine-Tuning Comparison
COMP 9130 Applied Artificial Intelligence — Mini Project 6

Requirement 4: Comparison & Analysis

Usage:
    python src/compare.py

Expects:
    models/feature_extraction_model.keras
    models/fine_tuning_model.keras
    data/test/  (class subfolders)

Outputs (saved to figures/):
    comparison_metrics.png
    accuracy_comparison.png
    fe_confusion_matrix.png
    ft_confusion_matrix.png
    comparison_table.png
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from src.utils import (
    MODELS_DIR, FIGURES_DIR,
    load_dataset_from_directory,
    get_predictions,
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_accuracy_bar,
    count_parameters,
    print_summary_table,
)


# ── Config ────────────────────────────────────────────────────────────────────

FE_MODEL_PATH = os.path.join(MODELS_DIR, "feature_extraction_model.keras")
FT_MODEL_PATH = os.path.join(MODELS_DIR, "fine_tuning_model.keras")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_check_model(path, name):
    """Load a model and print a short summary."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Model not found: {path}\n"
            f"Make sure your partner has saved the {name} model from their notebook.\n"
            f"Expected path: {path}"
        )
    print(f"\nLoading {name} from: {path}")
    model = load_model(path)
    trainable, total = count_parameters(model)
    print(f"  Trainable params : {trainable:>12,}")
    print(f"  Total params     : {total:>12,}")
    return model, trainable, total


def timed_evaluate(model, dataset, name):
    """Evaluate model and measure inference time."""
    print(f"\nEvaluating {name}...")
    start = time.time()
    y_true, y_pred, _ = get_predictions(model, dataset)
    elapsed = time.time() - start
    print(f"  Inference time: {elapsed:.1f}s")
    metrics = compute_metrics(y_true, y_pred)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    return y_true, y_pred, metrics, elapsed


def plot_comparison_table(fe_row, ft_row, filename="comparison_table.png"):
    """
    Render the comparison table as a clean matplotlib figure
    so it can be included directly in the report.
    """
    columns = ["Method", "Accuracy", "Precision", "Recall", "F1 Score",
               "Trainable Params", "Inference Time (s)"]
    cell_data = [
        ["Feature Extraction"] + fe_row,
        ["Fine-Tuning"]         + ft_row,
    ]

    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style header row
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for j in range(len(columns)):
        table[1, j].set_facecolor("#d6eaf8")
        table[2, j].set_facecolor("#fdebd0")

    plt.title("Feature Extraction vs Fine-Tuning — Summary Comparison",
              fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved comparison table: {out_path}")
    plt.show()


def analyse_class_improvements(y_true_fe, y_pred_fe, y_true_ft, y_pred_ft, top_n=10):
    """
    Find classes that improved and worsened most between FE and FT.
    Prints a summary to help write the Discussion section.
    """
    from sklearn.metrics import confusion_matrix

    cm_fe = confusion_matrix(y_true_fe, y_pred_fe)
    cm_ft = confusion_matrix(y_true_ft, y_pred_ft)

    # Per-class accuracy
    acc_fe = np.diag(cm_fe) / cm_fe.sum(axis=1)
    acc_ft = np.diag(cm_ft) / cm_ft.sum(axis=1)
    delta  = acc_ft - acc_fe

    # Top improved
    top_improved = np.argsort(delta)[-top_n:][::-1]
    # Top worsened
    top_worsened = np.argsort(delta)[:top_n]

    print(f"\n{'─'*55}")
    print(f"  Per-Class Analysis (top {top_n} changes)")
    print(f"{'─'*55}")
    print(f"\n  Most Improved Classes (FE → FT):")
    print(f"  {'Class':<10} {'FE Acc':>8} {'FT Acc':>8} {'Delta':>8}")
    for idx in top_improved:
        print(f"  Class {idx:<5} {acc_fe[idx]:>8.3f} {acc_ft[idx]:>8.3f} {delta[idx]:>+8.3f}")

    print(f"\n  Most Worsened Classes (FE → FT):")
    print(f"  {'Class':<10} {'FE Acc':>8} {'FT Acc':>8} {'Delta':>8}")
    for idx in top_worsened:
        print(f"  Class {idx:<5} {acc_fe[idx]:>8.3f} {acc_ft[idx]:>8.3f} {delta[idx]:>+8.3f}")

    print(f"\n  ► Use these numbers in your Discussion section!")
    print(f"    Example: 'Fine-tuning improved Class X accuracy from")
    print(f"    {acc_fe[top_improved[0]]:.3f} to {acc_ft[top_improved[0]]:.3f}")
    print(f"    (+{delta[top_improved[0]]:.3f}), likely because...'")

    return delta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  COMP 9130 — Mini Project 6")
    print("  Comparison & Analysis: Feature Extraction vs Fine-Tuning")
    print("=" * 60)

    # 1. Load models
    fe_model, fe_trainable, fe_total = load_and_check_model(FE_MODEL_PATH, "Feature Extraction")
    ft_model, ft_trainable, ft_total = load_and_check_model(FT_MODEL_PATH, "Fine-Tuning")

    # 2. Load test dataset
    print("\nLoading test dataset...")
    test_ds = load_dataset_from_directory(split="test")

    # 3. Evaluate both models
    fe_true, fe_pred, fe_metrics, fe_time = timed_evaluate(fe_model, test_ds, "Feature Extraction")
    ft_true, ft_pred, ft_metrics, ft_time = timed_evaluate(ft_model, test_ds, "Fine-Tuning")

    # 4. Full classification reports
    print_classification_report(fe_true, fe_pred, "Feature Extraction")
    print_classification_report(ft_true, ft_pred, "Fine-Tuning")

    # 5. Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(fe_true, fe_pred,
                          title="Feature Extraction — Confusion Matrix",
                          filename="fe_confusion_matrix.png")
    plot_confusion_matrix(ft_true, ft_pred,
                          title="Fine-Tuning — Confusion Matrix",
                          filename="ft_confusion_matrix.png")

    # 6. Metric comparison bar chart
    print("\nGenerating comparison charts...")
    plot_metric_comparison(fe_metrics, ft_metrics,
                           filename="comparison_metrics.png")
    plot_accuracy_bar(
        {"Feature Extraction": fe_metrics["accuracy"],
         "Fine-Tuning":        ft_metrics["accuracy"]},
        filename="accuracy_comparison.png"
    )

    # 7. Summary comparison table
    fe_row = [
        f"{fe_metrics['accuracy']:.4f}",
        f"{fe_metrics['precision']:.4f}",
        f"{fe_metrics['recall']:.4f}",
        f"{fe_metrics['f1']:.4f}",
        f"{fe_trainable:,}",
        f"{fe_time:.1f}",
    ]
    ft_row = [
        f"{ft_metrics['accuracy']:.4f}",
        f"{ft_metrics['precision']:.4f}",
        f"{ft_metrics['recall']:.4f}",
        f"{ft_metrics['f1']:.4f}",
        f"{ft_trainable:,}",
        f"{ft_time:.1f}",
    ]
    print("\nSummary Table:")
    print_summary_table(
        rows=[["Feature Extraction"] + fe_row, ["Fine-Tuning"] + ft_row],
        headers=["Method", "Accuracy", "Precision", "Recall", "F1",
                 "Trainable Params", "Inference (s)"]
    )
    plot_comparison_table(fe_row, ft_row)

    # 8. Per-class improvement analysis (for Discussion section)
    analyse_class_improvements(fe_true, fe_pred, ft_true, ft_pred)

    print("\n" + "=" * 60)
    print("  Done! All figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
