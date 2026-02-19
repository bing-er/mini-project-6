"""
utils.py — Shared helper functions for Mini Project 6
COMP 9130 Applied Artificial Intelligence

Used by: compare.py, gradcam.py, arch_compare.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# ── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 102
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ── Dataset Loading ──────────────────────────────────────────────────────────

def preprocess_image(image, label):
    """Resize and normalise a single image."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_dataset_from_directory(split="test"):
    """
    Load a dataset split from data/<split>/ directory.
    Expects subfolders named by class index (0–101).

    Returns a batched, prefetched tf.data.Dataset.
    """
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(
            f"Directory not found: {split_dir}\n"
            f"Please follow data/DATA_INSTRUCTIONS.txt to set up the dataset."
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return ds


# ── Evaluation ───────────────────────────────────────────────────────────────

def get_predictions(model, dataset):
    """
    Run inference on a dataset and return true labels and predicted labels.

    Returns:
        y_true (np.array): ground truth labels
        y_pred (np.array): predicted labels
        y_prob (np.array): predicted probabilities (shape: n_samples x n_classes)
    """
    y_true, y_pred, y_prob = [], [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(probs, axis=1))
        y_prob.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1 (weighted).

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy":  round(float(acc),       4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "f1":        round(float(f1),        4),
    }


def print_classification_report(y_true, y_pred, model_name="Model"):
    """Print a full sklearn classification report."""
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, zero_division=0))


# ── Plotting — Training Curves ────────────────────────────────────────────────

def plot_training_curves(history, title, filename):
    """
    Plot accuracy and loss curves from a Keras History object.

    Args:
        history:  Keras History object (or dict with same keys)
        title:    Title prefix for the plot
        filename: Output filename (saved to figures/)
    """
    h = history.history if hasattr(history, "history") else history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Accuracy
    ax1.plot(h["accuracy"],     label="Train",      color="steelblue")
    ax1.plot(h["val_accuracy"], label="Validation", color="darkorange", linestyle="--")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(h["loss"],     label="Train",      color="steelblue")
    ax2.plot(h["val_loss"], label="Validation", color="darkorange", linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


# ── Plotting — Confusion Matrix ───────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, filename, top_n_classes=20):
    """
    Plot confusion matrix. With 102 classes the full matrix is very large,
    so by default we show only the top_n_classes most confused classes.

    Args:
        y_true:        ground truth labels
        y_pred:        predicted labels
        title:         plot title
        filename:      output filename (saved to figures/)
        top_n_classes: number of most-confused classes to show (default 20)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Select top_n most confused classes (off-diagonal errors)
    errors = cm.sum(axis=1) - np.diag(cm)
    top_idx = np.argsort(errors)[-top_n_classes:][::-1]
    cm_subset = cm[np.ix_(top_idx, top_idx)]
    labels = [f"Class {i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm_subset,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"{title}\n(Top {top_n_classes} most confused classes)", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()

    return cm


# ── Plotting — Comparison Chart ───────────────────────────────────────────────

def plot_metric_comparison(metrics_fe, metrics_ft, filename="comparison_metrics.png"):
    """
    Side-by-side bar chart comparing Feature Extraction vs Fine-Tuning
    across accuracy, precision, recall, and F1.

    Args:
        metrics_fe: dict from compute_metrics() for Feature Extraction
        metrics_ft: dict from compute_metrics() for Fine-Tuning
        filename:   output filename (saved to figures/)
    """
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels       = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fe_values    = [metrics_fe[m] for m in metric_names]
    ft_values    = [metrics_ft[m] for m in metric_names]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_fe = ax.bar(x - width/2, fe_values, width, label="Feature Extraction", color="steelblue")
    bars_ft = ax.bar(x + width/2, ft_values, width, label="Fine-Tuning",        color="darkorange")

    ax.set_title("Feature Extraction vs Fine-Tuning — Performance Comparison", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars_fe:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_ft:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


def plot_accuracy_bar(results_dict, filename="accuracy_comparison.png", title="Test Accuracy Comparison"):
    """
    Simple accuracy bar chart for any number of models.

    Args:
        results_dict: e.g. {"Feature Extraction": 0.823, "Fine-Tuning": 0.871}
        filename:     output filename (saved to figures/)
        title:        chart title
    """
    names  = list(results_dict.keys())
    values = list(results_dict.values())
    colors = ["steelblue", "darkorange", "seagreen", "tomato"][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, width=0.4)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


# ── Misc ──────────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Return (trainable_params, total_params) for a Keras model."""
    trainable = int(np.sum([tf.size(w).numpy() for w in model.trainable_variables]))
    total     = int(np.sum([tf.size(w).numpy() for w in model.variables]))
    return trainable, total


def print_summary_table(rows, headers):
    """
    Print a simple ASCII comparison table.

    Args:
        rows:    list of lists, e.g. [["Feature Extraction", "82.3%", "120s"], ...]
        headers: list of column names
    """
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    separator  = "+" + "+".join("-" * w for w in col_widths) + "+"
    row_fmt    = "|" + "|".join(f" {{:<{w-1}}}" for w in col_widths) + "|"

    print(separator)
    print(row_fmt.format(*headers))
    print(separator)
    for row in rows:
        print(row_fmt.format(*[str(v) for v in row]))
    print(separator)
