from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend import get_backend
from data import (
    ImageBatcher,
    limit_samples_per_class,
    load_samples_to_arrays,
    load_split_file,
    save_split_file,
    scan_dataset,
    stratified_split,
)
from metrics import accuracy
from model import MLP, softmax_cross_entropy
from optim import SGD, StepDecay


def compute_normalization_stats(x: np.ndarray, image_size: int, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    if mode == "none":
        return None
    if mode == "channel":
        images = x.reshape(x.shape[0], image_size, image_size, 3)
        mean = images.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
        std = images.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    elif mode == "feature":
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray, image_size: int, mode: str) -> np.ndarray:
    if mode == "channel":
        images = x.reshape(x.shape[0], image_size, image_size, 3)
        images = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
        return images.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    if mode == "feature":
        return ((x - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32, copy=False)
    if mode == "none":
        return x
    raise ValueError(f"Unsupported normalize mode: {mode}")


def apply_normalization_backend(x, mean, std, image_size: int, mode: str, xp):
    if mode == "channel":
        images = x.reshape(x.shape[0], image_size, image_size, 3)
        images = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
        return images.reshape(x.shape[0], -1).astype(xp.float32, copy=False)
    if mode == "feature":
        return ((x - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(xp.float32, copy=False)
    if mode == "none":
        return x
    raise ValueError(f"Unsupported normalize mode: {mode}")


def augment_batch(x, image_size: int, xp, rng: np.random.Generator, config: dict):
    images = x.reshape(x.shape[0], image_size, image_size, 3).copy()
    batch_size = images.shape[0]

    if config.get("aug_hflip", True):
        mask = rng.random(batch_size) < 0.5
        if mask.any():
            mask_xp = xp.asarray(mask) if xp is not np else mask
            images[mask_xp] = images[mask_xp, :, ::-1, :]

    if config.get("aug_vflip", True):
        mask = rng.random(batch_size) < 0.5
        if mask.any():
            mask_xp = xp.asarray(mask) if xp is not np else mask
            images[mask_xp] = images[mask_xp, ::-1, :, :]

    if config.get("aug_rot90", True):
        rotations = rng.integers(0, 4, size=batch_size)
        for k in (1, 2, 3):
            mask = rotations == k
            if mask.any():
                mask_xp = xp.asarray(mask) if xp is not np else mask
                images[mask_xp] = xp.rot90(images[mask_xp], k=int(k), axes=(1, 2))

    brightness = float(config.get("aug_brightness", 0.0))
    if brightness > 0:
        factors = rng.uniform(1.0 - brightness, 1.0 + brightness, size=(batch_size, 1, 1, 1)).astype(np.float32)
        factors = xp.asarray(factors) if xp is not np else factors
        images = images * factors

    contrast = float(config.get("aug_contrast", 0.0))
    if contrast > 0:
        factors = rng.uniform(1.0 - contrast, 1.0 + contrast, size=(batch_size, 1, 1, 1)).astype(np.float32)
        factors = xp.asarray(factors) if xp is not np else factors
        images = (images - 0.5) * factors + 0.5

    images = xp.clip(images, 0.0, 1.0)
    return images.reshape(batch_size, -1).astype(xp.float32, copy=False)


def iter_array_batches(
    x,
    y,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    xp,
):
    order = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        if xp is np:
            indexer = batch_indices
        else:
            indexer = xp.asarray(batch_indices)
        yield x[indexer], y[indexer]


def evaluate(
    model: MLP,
    samples: list[tuple[str, int]],
    data_root: str | Path,
    batch_size: int,
    image_size: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for x, y in ImageBatcher(samples, data_root, batch_size, image_size=image_size, shuffle=False):
        logits = model.forward(x)
        loss, _, _ = softmax_cross_entropy(logits, y)
        pred = np.argmax(logits, axis=1)
        total_loss += loss * len(y)
        total_seen += len(y)
        y_true_parts.append(y)
        y_pred_parts.append(pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": accuracy(y_true, y_pred),
    }


def evaluate_arrays(model: MLP, x, y, batch_size: int, xp) -> dict:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0

    rng = np.random.default_rng(0)
    for xb, yb in iter_array_batches(x, y, batch_size, shuffle=False, rng=rng, xp=xp):
        logits = model.forward(xb)
        loss, _, _ = softmax_cross_entropy(logits, yb)
        pred = xp.argmax(logits, axis=1)
        total_loss += loss * int(yb.shape[0])
        total_seen += int(yb.shape[0])
        total_correct += int(xp.sum(pred == yb).item())

    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
    }


def prepare_splits(config: dict):
    split_file = config.get("split_file")
    if split_file:
        splits, classes, split_meta = load_split_file(split_file)
        return splits, classes, split_meta

    samples, classes = scan_dataset(config["data_root"])
    samples = limit_samples_per_class(samples, config.get("max_per_class"), seed=config["seed"])
    splits = stratified_split(
        samples,
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
    )
    split_meta = {
        "data_root": str(Path(config["data_root"]).resolve()),
        "classes": classes,
        "seed": config["seed"],
        "val_ratio": config["val_ratio"],
        "test_ratio": config["test_ratio"],
    }
    return splits, classes, split_meta


def run_training(config: dict) -> dict:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_name = config.get("backend", "numpy")
    xp = get_backend(backend_name)
    if backend_name == "cupy":
        xp.cuda.Device(config.get("device", 0)).use()
        xp.random.seed(config["seed"])

    splits, classes, split_meta = prepare_splits(config)
    save_split_file(
        out_dir / "splits.json",
        splits,
        classes,
        config["data_root"],
        seed=int(split_meta.get("seed", config["seed"])),
        val_ratio=float(split_meta.get("val_ratio", config["val_ratio"])),
        test_ratio=float(split_meta.get("test_ratio", config["test_ratio"])),
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    input_dim = config["image_size"] * config["image_size"] * 3
    model = MLP(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=len(classes),
        activation=config["activation"],
        seed=config["seed"],
        xp=xp,
        dropout=config.get("dropout", 0.0),
    )
    optimizer = SGD(config["lr"])
    scheduler = StepDecay(config["lr"], config["lr_decay"], config["lr_decay_every"])

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    verbose = bool(config.get("verbose", True))
    cache_data = config.get("cache_data", "auto")
    normalize_mode = config.get("normalize", "none")
    augment_enabled = bool(config.get("augment", False))
    use_cached_arrays = cache_data == "ram" or backend_name == "cupy" or normalize_mode != "none" or augment_enabled

    train_x = train_y = val_x = val_y = None
    norm_mean = norm_std = None
    if use_cached_arrays:
        if verbose:
            print("loading train/val images into RAM")
        train_x_cpu, train_y_cpu = load_samples_to_arrays(
            splits["train"], config["data_root"], config["image_size"], progress=verbose
        )
        val_x_cpu, val_y_cpu = load_samples_to_arrays(
            splits["val"], config["data_root"], config["image_size"], progress=verbose
        )
        stats = compute_normalization_stats(train_x_cpu, config["image_size"], normalize_mode)
        if stats is not None:
            mean, std = stats
            if verbose:
                print(f"using {normalize_mode} normalization")
            if not augment_enabled:
                train_x_cpu = apply_normalization(train_x_cpu, mean, std, config["image_size"], normalize_mode)
            val_x_cpu = apply_normalization(val_x_cpu, mean, std, config["image_size"], normalize_mode)
            np.savez_compressed(out_dir / "norm_stats.npz", mode=normalize_mode, mean=mean, std=std)
            norm_mean, norm_std = mean, std
        else:
            stats_path = out_dir / "norm_stats.npz"
            if stats_path.exists():
                stats_path.unlink()

        if backend_name == "cupy":
            if verbose:
                print("moving train/val arrays to GPU")
            train_x = xp.asarray(train_x_cpu)
            train_y = xp.asarray(train_y_cpu)
            val_x = xp.asarray(val_x_cpu)
            val_y = xp.asarray(val_y_cpu)
            if norm_mean is not None:
                norm_mean = xp.asarray(norm_mean)
                norm_std = xp.asarray(norm_std)
            xp.cuda.Stream.null.synchronize()
            if verbose:
                free, total = xp.cuda.runtime.memGetInfo()
                print(f"gpu memory free={free / 1024**3:.2f}GB total={total / 1024**3:.2f}GB")
        else:
            train_x, train_y = train_x_cpu, train_y_cpu
            val_x, val_y = val_x_cpu, val_y_cpu

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        lr = scheduler.lr_for_epoch(epoch - 1)
        optimizer.lr = lr

        train_loss_sum = 0.0
        train_seen = 0
        train_correct = 0

        if use_cached_arrays:
            batcher = iter_array_batches(
                train_x,
                train_y,
                config["batch_size"],
                shuffle=True,
                rng=np.random.default_rng(config["seed"] + epoch),
                xp=xp,
            )
        else:
            batcher = ImageBatcher(
                splits["train"],
                config["data_root"],
                config["batch_size"],
                image_size=config["image_size"],
                shuffle=True,
                seed=config["seed"] + epoch,
            )

        batch_rng = np.random.default_rng(config["seed"] + 100000 + epoch)
        for x, y in batcher:
            if not use_cached_arrays and backend_name != "numpy":
                x = xp.asarray(x)
                y = xp.asarray(y)
            if augment_enabled:
                x = augment_batch(x, config["image_size"], xp, batch_rng, config)
                if norm_mean is not None:
                    x = apply_normalization_backend(x, norm_mean, norm_std, config["image_size"], normalize_mode, xp)
            loss, logits = model.loss_and_backward(x, y, weight_decay=config["weight_decay"])
            optimizer.step(model.parameters_and_grads())

            batch_count = int(y.shape[0])
            train_loss_sum += loss * batch_count
            train_seen += batch_count
            pred = xp.argmax(logits, axis=1)
            train_correct += int(xp.sum(pred == y).item())

        train_acc = train_correct / max(1, train_seen)
        train_loss = train_loss_sum / max(1, train_seen)
        if use_cached_arrays:
            val_metrics = evaluate_arrays(model, val_x, val_y, config["batch_size"], xp)
        else:
            val_metrics = evaluate(
                model,
                splits["val"],
                config["data_root"],
                config["batch_size"],
                config["image_size"],
            )

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["lr"].append(float(lr))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            model.save(
                out_dir / "best_model.npz",
                metadata={
                    "classes": classes,
                    "image_size": config["image_size"],
                    "seed": config["seed"],
                    "val_ratio": config["val_ratio"],
                    "test_ratio": config["test_ratio"],
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "normalize": normalize_mode,
                    "dropout": config.get("dropout", 0.0),
                    "augment": augment_enabled,
                },
            )

        if verbose:
            print(
                f"epoch {epoch:03d}/{config['epochs']:03d} "
                f"lr={lr:.6f} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model.save(
        out_dir / "final_model.npz",
        metadata={
            "classes": classes,
            "image_size": config["image_size"],
            "seed": config["seed"],
            "val_ratio": config["val_ratio"],
            "test_ratio": config["test_ratio"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "normalize": normalize_mode,
            "dropout": config.get("dropout", 0.0),
            "augment": augment_enabled,
        },
    )

    try:
        from visualize import plot_training_curves, visualize_first_layer_weights

        plot_training_curves(history, out_dir / "training_curves.png")
        visualize_first_layer_weights(out_dir / "best_model.npz", out_dir / "first_layer_weights.png")
    except Exception as exc:
        if verbose:
            print(f"warning: failed to create visualizations: {exc}")

    result = {
        "out_dir": str(out_dir),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "history": history,
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "classes": classes,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on EuroSAT RGB.")
    parser.add_argument("--data-root", default="/root/hw1/EuroSAT_RGB")
    parser.add_argument("--out-dir", default="/root/hw1_solution/outputs/run")
    parser.add_argument("--split-file", default=None, help="Optional existing splits.json to reuse.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--lr-decay-every", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability after hidden activation.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="Use a subset for quick debugging.")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--device", type=int, default=0, help="CuPy CUDA device index after CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--normalize",
        choices=["none", "channel", "feature"],
        default="none",
        help="Normalize inputs using statistics computed from the training split.",
    )
    parser.add_argument(
        "--cache-data",
        choices=["auto", "ram", "none"],
        default="auto",
        help="Preload train/val arrays. CuPy uses preloading automatically.",
    )
    parser.add_argument("--augment", action="store_true", help="Apply train-time random image augmentation.")
    parser.add_argument("--no-aug-hflip", action="store_false", dest="aug_hflip")
    parser.add_argument("--no-aug-vflip", action="store_false", dest="aug_vflip")
    parser.add_argument("--no-aug-rot90", action="store_false", dest="aug_rot90")
    parser.add_argument("--aug-brightness", type=float, default=0.0)
    parser.add_argument("--aug-contrast", type=float, default=0.0)
    parser.set_defaults(aug_hflip=True, aug_vflip=True, aug_rot90=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    config = vars(parser.parse_args())
    run_training(config)


if __name__ == "__main__":
    main()
