from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def format_confusion_matrix(matrix: np.ndarray, classes: list[str]) -> str:
    name_width = max(12, max(len(name) for name in classes))
    cell_width = max(6, len(str(int(matrix.max()))) + 2)
    header = " " * (name_width + 1) + "".join(f"{name[:cell_width-1]:>{cell_width}}" for name in classes)
    lines = [header]
    for name, row in zip(classes, matrix):
        lines.append(f"{name:<{name_width}} " + "".join(f"{int(value):>{cell_width}d}" for value in row))
    return "\n".join(lines)
