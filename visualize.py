from __future__ import annotations

import json
from math import ceil, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import load_checkpoint


def plot_training_curves(history: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_accuracy"], label="val accuracy", color="#1f77b4")
    if "train_accuracy" in history:
        axes[1].plot(epochs, history["train_accuracy"], label="train accuracy", color="#ff7f0e")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _normalize_image(array: np.ndarray) -> np.ndarray:
    lo = np.percentile(array, 1)
    hi = np.percentile(array, 99)
    if hi <= lo:
        lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def visualize_first_layer_weights(
    checkpoint_path: str | Path,
    out_path: str | Path,
    max_units: int = 64,
) -> None:
    model, metadata = load_checkpoint(checkpoint_path)
    image_size = int(metadata.get("image_size", 64))
    expected_dim = image_size * image_size * 3
    if model.fc1.W.shape[0] != expected_dim:
        raise ValueError(f"Cannot reshape first-layer weights with input_dim={model.fc1.W.shape[0]}")

    weights = model.fc1.W
    norms = np.linalg.norm(weights, axis=0)
    unit_indices = np.argsort(norms)[::-1][: min(max_units, weights.shape[1])]
    images = []
    for unit_index in unit_indices:
        image = weights[:, unit_index].reshape(image_size, image_size, 3)
        images.append(_normalize_image(image))

    cols = int(ceil(sqrt(len(images))))
    rows = int(ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    for ax, image, unit_index in zip(axes, images, unit_indices):
        ax.imshow(image)
        ax.set_title(str(int(unit_index)), fontsize=7)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = matrix.max() * 0.6 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=color, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_error_examples(
    errors: list[tuple[str, int, int]],
    data_root: str | Path,
    classes: list[str],
    out_path: str | Path,
    max_examples: int = 12,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = errors[:max_examples]
    if not selected:
        payload = {"message": "No misclassified examples found."}
        out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    cols = min(4, len(selected))
    rows = int(ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6), dpi=150)
    axes = np.asarray(axes).reshape(-1)

    for ax, (rel_path, true_label, pred_label) in zip(axes, selected):
        with Image.open(Path(data_root) / rel_path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"T: {classes[true_label]}\nP: {classes[pred_label]}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(selected) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
