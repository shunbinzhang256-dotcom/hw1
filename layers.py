from __future__ import annotations

import numpy as np


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "xavier",
        rng=None,
        xp=np,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rng = np.random.default_rng() if rng is None else rng
        self.xp = xp

        if init == "he":
            scale = np.sqrt(2.0 / self.in_features)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (self.in_features + self.out_features))
        else:
            raise ValueError(f"Unknown init: {init}")

        w_cpu = (self.rng.normal(0.0, scale, size=(self.in_features, self.out_features))).astype(np.float32)
        b_cpu = np.zeros(self.out_features, dtype=np.float32)
        self.W = self.xp.asarray(w_cpu)
        self.b = self.xp.asarray(b_cpu)
        self.dW = self.xp.zeros_like(self.W)
        self.db = self.xp.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Linear.backward called before forward")
        self.dW[...] = self._x.T @ grad_output
        self.db[...] = self.xp.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def parameters_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    name = "relu"

    def __init__(self, xp=np) -> None:
        self.xp = xp
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return self.xp.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad_output * self._mask


class Tanh:
    name = "tanh"

    def __init__(self, xp=np) -> None:
        self.xp = xp
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = self.xp.tanh(x)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Tanh.backward called before forward")
        return grad_output * (1.0 - self._output ** 2)


class Sigmoid:
    name = "sigmoid"

    def __init__(self, xp=np) -> None:
        self.xp = xp
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clip = self.xp.clip(x, -50, 50)
        self._output = 1.0 / (1.0 + self.xp.exp(-x_clip))
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return grad_output * self._output * (1.0 - self._output)


class Dropout:
    def __init__(self, p: float = 0.0, xp=np, seed: int = 42) -> None:
        self.p = float(p)
        self.xp = xp
        self.rng = np.random.default_rng(seed)
        self._mask = None
        if self.p < 0.0 or self.p >= 1.0:
            raise ValueError("dropout p must satisfy 0 <= p < 1")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.p == 0.0 or not training:
            self._mask = None
            return x

        if self.xp is np:
            keep = self.rng.random(x.shape) >= self.p
            self._mask = keep.astype(x.dtype, copy=False) / (1.0 - self.p)
        else:
            random_values = self.xp.random.random(x.shape).astype(x.dtype, copy=False)
            self._mask = (random_values >= self.p).astype(x.dtype, copy=False) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_output
        return grad_output * self._mask


def make_activation(name: str, xp=np):
    normalized = name.lower()
    if normalized == "relu":
        return ReLU(xp=xp)
    if normalized == "tanh":
        return Tanh(xp=xp)
    if normalized == "sigmoid":
        return Sigmoid(xp=xp)
    raise ValueError(f"Unsupported activation: {name}")
