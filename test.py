from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data import load_image_flat, load_split_file, scan_dataset, stratified_split
from metrics import accuracy, confusion_matrix, format_confusion_matrix
from model import load_checkpoint
from visualize import plot_confusion_matrix, save_error_examples


def load_normalization_stats(checkpoint: Path, explicit_path: str | None):
    stats_path = Path(explicit_path) if explicit_path else checkpoint.parent / "norm_stats.npz"
    if not stats_path.exists():
        return None
    stats = np.load(stats_path, allow_pickle=False)
    mode = str(stats["mode"].item())
    return {
        "mode": mode,
        "mean": stats["mean"].astype(np.float32),
        "std": stats["std"].astype(np.float32),
        "path": stats_path,
    }


def apply_normalization(x: np.ndarray, stats: dict | None, image_size: int) -> np.ndarray:
    if stats is None:
        return x
    mode = stats["mode"]
    mean = stats["mean"]
    std = stats["std"]
    if mode == "channel":
        images = x.reshape(x.shape[0], image_size, image_size, 3)
        images = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
        return images.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    if mode == "feature":
        return ((x - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32, copy=False)
    if mode == "none":
        return x
    raise ValueError(f"Unsupported normalization mode in stats file: {mode}")


def run_test(config: dict) -> dict:
    checkpoint = Path(config["checkpoint"])
    model, metadata = load_checkpoint(checkpoint)
    data_root = Path(config["data_root"])
    image_size = int(metadata.get("image_size", config["image_size"]))
    norm_stats = load_normalization_stats(checkpoint, config.get("norm_stats"))
    if norm_stats is not None:
        print(f"using normalization stats from {norm_stats['path']}")

    if config.get("split_file"):
        splits, classes, _ = load_split_file(config["split_file"])
        test_samples = splits["test"]
    else:
        samples, classes = scan_dataset(data_root)
        splits = stratified_split(
            samples,
            val_ratio=float(metadata.get("val_ratio", 0.15)),
            test_ratio=float(metadata.get("test_ratio", 0.15)),
            seed=int(metadata.get("seed", 42)),
        )
        test_samples = splits["test"]

    if "classes" in metadata:
        classes = list(metadata["classes"])

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    errors: list[tuple[str, int, int]] = []

    batch_size = int(config["batch_size"])
    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start : start + batch_size]
        x = np.stack([load_image_flat(data_root, rel_path, image_size) for rel_path, _ in batch], axis=0)
        x = apply_normalization(x, norm_stats, image_size)
        y = np.asarray([label for _, label in batch], dtype=np.int64)
        pred = model.predict(x)
        y_true_all.append(y)
        y_pred_all.append(pred)
        for (rel_path, true_label), pred_label in zip(batch, pred):
            if int(true_label) != int(pred_label):
                errors.append((rel_path, int(true_label), int(pred_label)))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = accuracy(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, len(classes))

    out_dir = Path(config["out_dir"]) if config.get("out_dir") else checkpoint.parent / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *[int(x) for x in row]])

    plot_confusion_matrix(matrix, classes, out_dir / "confusion_matrix.png")
    save_error_examples(errors, data_root, classes, out_dir / "error_examples.png", config["max_errors"])

    result = {
        "checkpoint": str(checkpoint),
        "accuracy": acc,
        "num_test": int(len(test_samples)),
        "num_errors": int(len(errors)),
        "classes": classes,
    }
    (out_dir / "test_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"test_accuracy={acc:.4f} ({int((y_true == y_pred).sum())}/{len(y_true)})")
    print(format_confusion_matrix(matrix, classes))
    print(f"saved test artifacts to {out_dir}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP checkpoint.")
    parser.add_argument("--data-root", default="/root/hw1/EuroSAT_RGB")
    parser.add_argument("--checkpoint", default="/root/hw1_solution/outputs/run/best_model.npz")
    parser.add_argument("--split-file", default="/root/hw1_solution/outputs/run/splits.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-errors", type=int, default=12)
    parser.add_argument("--norm-stats", default=None, help="Optional path to norm_stats.npz.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run_test(vars(parser.parse_args()))


if __name__ == "__main__":
    main()
