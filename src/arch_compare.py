"""
arch_compare.py — Architecture Comparison: ResNet50 vs EfficientNetB0 (Bonus)
COMP 9130 Applied Artificial Intelligence — Mini Project 6

Bonus: Compare two pre-trained architectures as feature extractors
and fine-tuned models.

Usage:
    python src/arch_compare.py

Expects:
    data/train/, data/valid/, data/test/

Outputs (saved to figures/):
    arch_accuracy_comparison.png
    arch_training_curves_resnet50.png
    arch_training_curves_efficientnet.png
    arch_summary_table.png

Outputs (saved to models/):
    arch_resnet50_fe.keras
    arch_efficientnet_fe.keras
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import load_model
from src.utils import (
    MODELS_DIR, FIGURES_DIR, NUM_CLASSES, IMG_SIZE, BATCH_SIZE,
    load_dataset_from_directory,
    get_predictions,
    compute_metrics,
    plot_training_curves,
    plot_accuracy_bar,
    count_parameters,
    print_summary_table,
)


# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS      = 20   # keep lower for bonus comparison (speed vs depth)
UNFREEZE_N  = 30   # layers to unfreeze for fine-tuning phase

ARCHITECTURES = {
    "ResNet50":       ResNet50,
    "EfficientNetB0": EfficientNetB0,
}


# ── Model Building ────────────────────────────────────────────────────────────

def build_model(base_fn, name, input_shape=(IMG_SIZE, IMG_SIZE, 3), frozen=True):
    """
    Build a transfer learning model with a custom classification head.

    Args:
        base_fn:     Keras application function (e.g. ResNet50)
        name:        model name string
        input_shape: input image shape
        frozen:      if True, freeze base (Feature Extraction);
                     if False, partially unfreeze (Fine-Tuning)

    Returns:
        model, base_model
    """
    base = base_fn(weights="imagenet", include_top=False, input_shape=input_shape)

    if frozen:
        base.trainable = False
    else:
        base.trainable = True
        # Freeze all except last UNFREEZE_N layers
        for layer in base.layers[:-UNFREEZE_N]:
            layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name=name)

    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, model_name, epochs=EPOCHS, lr=1e-3):
    """Train a model with early stopping and return history + training time."""
    compile_model(model, lr)

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"arch_{model_name.lower().replace(' ', '_')}.keras"),
            save_best_only=True, verbose=0
        )
    ]

    print(f"\n  Training {model_name}  (lr={lr}, max epochs={epochs})")
    trainable, total = count_parameters(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    start   = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=cb, verbose=1
    )
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")
    return history, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  COMP 9130 — Mini Project 6")
    print("  Bonus: Architecture Comparison")
    print("  ResNet50 vs EfficientNetB0")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset_from_directory(split="train")
    val_ds   = load_dataset_from_directory(split="valid")
    test_ds  = load_dataset_from_directory(split="test")

    results = {}   # store all results for final comparison

    for arch_name, arch_fn in ARCHITECTURES.items():
        print(f"\n{'─'*55}")
        print(f"  Architecture: {arch_name}")
        print(f"{'─'*55}")

        # ── Feature Extraction ─────────────────────────────────────
        print(f"\n[1/2] Feature Extraction — {arch_name}")
        fe_model, base = build_model(arch_fn, f"{arch_name}_FE", frozen=True)
        fe_history, fe_time = train_model(
            fe_model, train_ds, val_ds,
            model_name=f"{arch_name}_fe", epochs=EPOCHS, lr=1e-3
        )
        plot_training_curves(
            fe_history,
            title=f"{arch_name} — Feature Extraction",
            filename=f"arch_{arch_name.lower()}_fe_curves.png"
        )
        fe_true, fe_pred, _ = get_predictions(fe_model, test_ds)
        fe_metrics           = compute_metrics(fe_true, fe_pred)
        fe_trainable, _      = count_parameters(fe_model)

        print(f"  FE Test Accuracy: {fe_metrics['accuracy']:.4f}")

        # ── Fine-Tuning ────────────────────────────────────────────
        print(f"\n[2/2] Fine-Tuning — {arch_name}")
        # Unfreeze layers in the already-trained model
        base.trainable = True
        for layer in base.layers[:-UNFREEZE_N]:
            layer.trainable = False

        ft_history, ft_time = train_model(
            fe_model, train_ds, val_ds,
            model_name=f"{arch_name}_ft", epochs=EPOCHS, lr=1e-5
        )
        plot_training_curves(
            ft_history,
            title=f"{arch_name} — Fine-Tuning",
            filename=f"arch_{arch_name.lower()}_ft_curves.png"
        )
        ft_true, ft_pred, _ = get_predictions(fe_model, test_ds)
        ft_metrics           = compute_metrics(ft_true, ft_pred)
        ft_trainable, _      = count_parameters(fe_model)

        print(f"  FT Test Accuracy: {ft_metrics['accuracy']:.4f}")

        results[arch_name] = {
            "fe": {"metrics": fe_metrics, "time": fe_time, "trainable": fe_trainable},
            "ft": {"metrics": ft_metrics, "time": ft_time, "trainable": ft_trainable},
        }

    # ── Final Comparison ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Architecture Comparison Summary")
    print("=" * 60)

    rows = []
    acc_dict = {}
    for arch, data in results.items():
        fe = data["fe"]
        ft = data["ft"]
        rows.append([
            f"{arch} (FE)",
            f"{fe['metrics']['accuracy']:.4f}",
            f"{fe['metrics']['f1']:.4f}",
            f"{fe['trainable']:,}",
            f"{fe['time']:.1f}s"
        ])
        rows.append([
            f"{arch} (FT)",
            f"{ft['metrics']['accuracy']:.4f}",
            f"{ft['metrics']['f1']:.4f}",
            f"{ft['trainable']:,}",
            f"{ft['time']:.1f}s"
        ])
        acc_dict[f"{arch} FE"] = fe["metrics"]["accuracy"]
        acc_dict[f"{arch} FT"] = ft["metrics"]["accuracy"]

    print_summary_table(
        rows=rows,
        headers=["Model", "Test Accuracy", "F1 Score", "Trainable Params", "Train Time"]
    )

    # Accuracy bar chart across all 4 combinations
    plot_accuracy_bar(
        acc_dict,
        filename="arch_accuracy_comparison.png",
        title="Architecture Comparison — Test Accuracy\n(ResNet50 vs EfficientNetB0)"
    )

    print("\n" + "=" * 60)
    print("  Done! All figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
