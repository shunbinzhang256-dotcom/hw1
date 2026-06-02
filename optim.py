from __future__ import annotations


class SGD:
    def __init__(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self, parameters_and_grads) -> None:
        for param, grad in parameters_and_grads:
            param -= self.lr * grad


class StepDecay:
    def __init__(self, initial_lr: float, decay_rate: float = 0.95, decay_every: int = 1) -> None:
        self.initial_lr = float(initial_lr)
        self.decay_rate = float(decay_rate)
        self.decay_every = int(decay_every)
        if self.decay_every <= 0:
            raise ValueError("decay_every must be positive")

    def lr_for_epoch(self, epoch_index: int) -> float:
        steps = epoch_index // self.decay_every
        return self.initial_lr * (self.decay_rate ** steps)
