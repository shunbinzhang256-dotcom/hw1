from __future__ import annotations

import numpy as np


def get_backend(name: str):
    normalized = name.lower()
    if normalized == "numpy":
        return np
    if normalized == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CuPy backend requested, but cupy is not installed.") from exc
        return cp
    raise ValueError(f"Unsupported backend: {name}")


def to_numpy(array):
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def scalar_to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
