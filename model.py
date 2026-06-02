from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend import to_numpy
from layers import Dropout, Linear, make_activation


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if hasattr(logits, "get"):
        import cupy as xp
    else:
        xp = np

    shifted = logits - xp.max(logits, axis=1, keepdims=True)
    exp_scores = xp.exp(shifted)
    probs = exp_scores / xp.sum(exp_scores, axis=1, keepdims=True)

    batch_size = labels.shape[0]
    rows = xp.arange(batch_size)
    loss = -xp.log(probs[rows, labels] + 1e-12).mean()

    grad = probs.copy()
    grad[rows, labels] -= 1.0
    grad /= batch_size
    return float(loss.item()), grad.astype(xp.float32), probs


class MLP:
    """Three-layer classifier in the input-hidden-output sense."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
        xp=np,
        dropout: float = 0.0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.activation_name = activation.lower()
        self.xp = xp
        self.dropout_p = float(dropout)
        self.training = True

        rng = np.random.default_rng(seed)
        init = "he" if self.activation_name == "relu" else "xavier"
        self.fc1 = Linear(self.input_dim, self.hidden_dim, init=init, rng=rng, xp=self.xp)
        self.activation = make_activation(self.activation_name, xp=self.xp)
        self.dropout = Dropout(self.dropout_p, xp=self.xp, seed=seed + 997)
        self.fc2 = Linear(self.hidden_dim, self.num_classes, init="xavier", rng=rng, xp=self.xp)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = self.fc1.forward(x)
        hidden = self.activation.forward(hidden)
        hidden = self.dropout.forward(hidden, training=self.training)
        return self.fc2.forward(hidden)

    def loss_and_backward(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        weight_decay: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        logits = self.forward(x)
        data_loss, grad_logits, _ = softmax_cross_entropy(logits, labels)

        grad_hidden = self.fc2.backward(grad_logits)
        grad_hidden = self.dropout.backward(grad_hidden)
        grad_hidden = self.activation.backward(grad_hidden)
        self.fc1.backward(grad_hidden)

        reg_loss = 0.0
        if weight_decay > 0:
            reg_loss = 0.5 * weight_decay * (
                self.xp.sum(self.fc1.W * self.fc1.W) + self.xp.sum(self.fc2.W * self.fc2.W)
            )
            self.fc1.dW += weight_decay * self.fc1.W
            self.fc2.dW += weight_decay * self.fc2.W

        return data_loss + float(reg_loss.item() if hasattr(reg_loss, "item") else reg_loss), logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        if was_training:
            self.train()
        return np.argmax(logits, axis=1)

    def parameters_and_grads(self):
        params = []
        params.extend(self.fc1.parameters_and_grads())
        params.extend(self.fc2.parameters_and_grads())
        return params

    def metadata(self, extra: dict | None = None) -> dict:
        payload = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "activation": self.activation_name,
            "dropout": self.dropout_p,
        }
        if extra:
            payload.update(extra)
        return payload

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            fc1_W=to_numpy(self.fc1.W),
            fc1_b=to_numpy(self.fc1.b),
            fc2_W=to_numpy(self.fc2.W),
            fc2_b=to_numpy(self.fc2.b),
            metadata=np.asarray(json.dumps(self.metadata(metadata))),
        )

    def load_arrays(self, arrays) -> None:
        self.fc1.W[...] = arrays["fc1_W"]
        self.fc1.b[...] = arrays["fc1_b"]
        self.fc2.W[...] = arrays["fc2_W"]
        self.fc2.b[...] = arrays["fc2_b"]


def load_checkpoint(path: str | Path) -> tuple[MLP, dict]:
    arrays = np.load(path, allow_pickle=False)
    metadata = json.loads(str(arrays["metadata"].item()))
    model = MLP(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_classes=int(metadata["num_classes"]),
        activation=str(metadata["activation"]),
        seed=0,
        dropout=float(metadata.get("dropout", 0.0)),
    )
    model.load_arrays(arrays)
    return model, metadata
