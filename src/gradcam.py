"""
gradcam.py — Grad-CAM Heatmap Visualizations (Bonus)
COMP 9130 Applied Artificial Intelligence — Mini Project 6

Bonus: Visualize what the model focuses on using Grad-CAM.
Shows heatmaps for correctly and incorrectly classified samples.

Usage:
    python src/gradcam.py

Expects:
    models/fine_tuning_model.keras
    data/test/  (class subfolders)

Outputs (saved to figures/):
    gradcam_correct.png
    gradcam_incorrect.png
    gradcam_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model, Model
from src.utils import (
    MODELS_DIR, FIGURES_DIR, DATA_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES,
    load_dataset_from_directory,
    get_predictions,
)


# ── Config ────────────────────────────────────────────────────────────────────

FT_MODEL_PATH  = os.path.join(MODELS_DIR, "fine_tuning_model.keras")
N_SAMPLES      = 8   # number of images to visualize per grid


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def find_last_conv_layer(model):
    """
    Automatically find the last convolutional layer in the model.
    Works for ResNet50 and EfficientNetB0.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"  Found last conv layer: {layer.name}")
            return layer.name
        # Handle nested models (e.g. ResNet50 is a sub-model)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    print(f"  Found last conv layer (nested): {sublayer.name}")
                    return sublayer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        img_array:           preprocessed image, shape (1, H, W, 3)
        model:               Keras model
        last_conv_layer_name: name of the target convolutional layer
        pred_index:          class index to explain (None = use top prediction)

    Returns:
        heatmap: np.array of shape (h, w), values in [0, 1]
    """
    # Build a model that outputs: conv feature maps + final predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap="jet"):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        img:      original image, np.array shape (H, W, 3), values [0, 1]
        heatmap:  Grad-CAM heatmap, np.array shape (h, w), values [0, 1]
        alpha:    heatmap transparency
        colormap: matplotlib colormap name

    Returns:
        superimposed image as np.array [0, 1]
    """
    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [img.shape[0], img.shape[1]]
    ).numpy().squeeze()

    # Apply colormap
    cmap   = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[..., :3]  # drop alpha channel

    # Superimpose
    superimposed = (1 - alpha) * img + alpha * heatmap_colored
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


# ── Sampling ─────────────────────────────────────────────────────────────────

def collect_samples(model, test_ds, n=8):
    """
    Collect n correctly and n incorrectly classified samples.

    Returns:
        correct_samples:   list of (image, true_label, pred_label)
        incorrect_samples: list of (image, true_label, pred_label)
    """
    correct, incorrect = [], []

    for images, labels in test_ds:
        probs  = model.predict(images, verbose=0)
        preds  = np.argmax(probs, axis=1)
        imgs_np = images.numpy()
        lbls_np = labels.numpy()

        for img, true, pred in zip(imgs_np, lbls_np, preds):
            if true == pred and len(correct) < n:
                correct.append((img, int(true), int(pred)))
            elif true != pred and len(incorrect) < n:
                incorrect.append((img, int(true), int(pred)))

        if len(correct) >= n and len(incorrect) >= n:
            break

    return correct, incorrect


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_gradcam_grid(samples, model, last_conv_layer, title, filename, cols=4):
    """
    Plot a grid of: original image | Grad-CAM overlay
    for a list of samples.
    """
    n    = len(samples)
    rows = n  # one row per sample, two columns: original + overlay

    fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (img, true_label, pred_label) in enumerate(samples):
        img_tensor = tf.expand_dims(img, axis=0)

        # Compute heatmap
        try:
            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer,
                                           pred_index=pred_label)
            overlay = overlay_heatmap(img, heatmap)
        except Exception as e:
            print(f"  [WARN] Grad-CAM failed for sample {i}: {e}")
            overlay = img

        ax_orig    = axes[i, 0]
        ax_overlay = axes[i, 1]

        ax_orig.imshow(img)
        ax_orig.set_title(f"True: {true_label}", fontsize=8)
        ax_orig.axis("off")

        ax_overlay.imshow(overlay)
        status = "✓ Correct" if true_label == pred_label else f"✗ Pred: {pred_label}"
        ax_overlay.set_title(f"Grad-CAM | {status}", fontsize=8)
        ax_overlay.axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


def plot_fe_vs_ft_gradcam(sample, fe_model, ft_model, fe_conv, ft_conv,
                           filename="gradcam_comparison.png"):
    """
    For a single image, show: original | FE Grad-CAM | FT Grad-CAM
    This directly shows how fine-tuning changes what the model focuses on.
    """
    img, true_label, _ = sample
    img_tensor = tf.expand_dims(img, axis=0)

    fe_probs = fe_model.predict(img_tensor, verbose=0)
    ft_probs = ft_model.predict(img_tensor, verbose=0)
    fe_pred  = int(np.argmax(fe_probs))
    ft_pred  = int(np.argmax(ft_probs))

    fe_heatmap = make_gradcam_heatmap(img_tensor, fe_model, fe_conv, pred_index=fe_pred)
    ft_heatmap = make_gradcam_heatmap(img_tensor, ft_model, ft_conv, pred_index=ft_pred)

    fe_overlay = overlay_heatmap(img, fe_heatmap)
    ft_overlay = overlay_heatmap(img, ft_heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Grad-CAM Comparison — True Class: {true_label}", fontsize=12)

    axes[0].imshow(img);        axes[0].set_title("Original Image");           axes[0].axis("off")
    axes[1].imshow(fe_overlay); axes[1].set_title(f"Feature Extraction\nPred: {fe_pred}"); axes[1].axis("off")
    axes[2].imshow(ft_overlay); axes[2].set_title(f"Fine-Tuning\nPred: {ft_pred}");        axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  COMP 9130 — Mini Project 6")
    print("  Bonus: Grad-CAM Heatmap Visualizations")
    print("=" * 60)

    # Load fine-tuned model (primary model for Grad-CAM)
    print(f"\nLoading Fine-Tuning model from: {FT_MODEL_PATH}")
    ft_model = load_model(FT_MODEL_PATH)

    # Also load FE model for comparison plot
    fe_model_path = os.path.join(MODELS_DIR, "feature_extraction_model.keras")
    fe_model = load_model(fe_model_path)

    # Find last conv layers
    print("\nFinding last conv layers...")
    ft_conv = find_last_conv_layer(ft_model)
    fe_conv = find_last_conv_layer(fe_model)

    # Load test data
    print("\nLoading test dataset...")
    test_ds = load_dataset_from_directory(split="test")

    # Collect correct and incorrect samples
    print(f"\nCollecting {N_SAMPLES} correct and {N_SAMPLES} incorrect samples...")
    correct_samples, incorrect_samples = collect_samples(ft_model, test_ds, n=N_SAMPLES)
    print(f"  Collected {len(correct_samples)} correct, {len(incorrect_samples)} incorrect")

    # Plot Grad-CAM grids
    print("\nGenerating Grad-CAM visualizations...")
    plot_gradcam_grid(
        correct_samples, ft_model, ft_conv,
        title="Grad-CAM — Correctly Classified Samples (Fine-Tuning Model)",
        filename="gradcam_correct.png"
    )
    plot_gradcam_grid(
        incorrect_samples, ft_model, ft_conv,
        title="Grad-CAM — Incorrectly Classified Samples (Fine-Tuning Model)",
        filename="gradcam_incorrect.png"
    )

    # FE vs FT comparison on a single image
    print("\nGenerating FE vs FT Grad-CAM comparison...")
    if correct_samples:
        plot_fe_vs_ft_gradcam(
            correct_samples[0], fe_model, ft_model, fe_conv, ft_conv,
            filename="gradcam_comparison.png"
        )

    print("\n" + "=" * 60)
    print("  Done! All Grad-CAM figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
