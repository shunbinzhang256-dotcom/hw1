from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_dataset(data_root: str | Path) -> tuple[list[tuple[str, int]], list[str]]:
    """Return samples as (relative_path, label_index), plus sorted class names."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class folders found under {root}")

    samples: list[tuple[str, int]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        image_paths = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No images found in class folder: {class_dir}")
        for path in image_paths:
            samples.append((str(path.relative_to(root)), label))
    return samples, classes


def stratified_split(
    samples: Iterable[tuple[str, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Expected val_ratio > 0, test_ratio > 0, and val_ratio + test_ratio < 1")

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for rel_path, label in samples:
        by_label[int(label)].append((rel_path, int(label)))

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        class_samples = by_label[label]
        order = rng.permutation(len(class_samples))
        shuffled = [class_samples[i] for i in order]

        n = len(shuffled)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n - n_test - n_val <= 0:
            raise ValueError(f"Class {label} has too few samples for the requested split ratios")

        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def limit_samples_per_class(
    samples: Iterable[tuple[str, int]],
    max_per_class: int | None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    if max_per_class is None or max_per_class <= 0:
        return list(samples)

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample[1])].append((sample[0], int(sample[1])))

    limited: list[tuple[str, int]] = []
    for label in sorted(by_label):
        class_samples = by_label[label]
        order = rng.permutation(len(class_samples))[:max_per_class]
        limited.extend(class_samples[i] for i in order)
    rng.shuffle(limited)
    return limited


def save_split_file(
    path: str | Path,
    splits: dict[str, list[tuple[str, int]]],
    classes: list[str],
    data_root: str | Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    payload = {
        "data_root": str(Path(data_root).resolve()),
        "classes": classes,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "splits": {key: [[p, int(y)] for p, y in value] for key, value in splits.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_file(path: str | Path) -> tuple[dict[str, list[tuple[str, int]]], list[str], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {
        key: [(str(p), int(y)) for p, y in value]
        for key, value in payload["splits"].items()
    }
    return splits, list(payload["classes"]), payload


def load_image_flat(
    data_root: str | Path,
    rel_path: str,
    image_size: int = 64,
) -> np.ndarray:
    path = Path(data_root) / rel_path
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(-1)


class ImageBatcher:
    def __init__(
        self,
        samples: list[tuple[str, int]],
        data_root: str | Path,
        batch_size: int,
        image_size: int = 64,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.data_root = Path(data_root)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = np.arange(len(self.samples))
        if self.shuffle:
            self.rng.shuffle(order)

        for start in range(0, len(order), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            batch_samples = [self.samples[i] for i in batch_indices]
            x = np.stack(
                [load_image_flat(self.data_root, rel_path, self.image_size) for rel_path, _ in batch_samples],
                axis=0,
            )
            y = np.asarray([label for _, label in batch_samples], dtype=np.int64)
            yield x, y


def load_samples_to_arrays(
    samples: list[tuple[str, int]],
    data_root: str | Path,
    image_size: int = 64,
    dtype=np.float32,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(samples), image_size * image_size * 3), dtype=dtype)
    y = np.empty((len(samples),), dtype=np.int64)
    for index, (rel_path, label) in enumerate(samples):
        x[index] = load_image_flat(data_root, rel_path, image_size).astype(dtype, copy=False)
        y[index] = int(label)
        if progress and (index + 1) % 5000 == 0:
            print(f"loaded {index + 1}/{len(samples)} images")
    return x, y
