from typing_extensions import override
from model.layers.base_layer import BaseLayer
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


class LayerNormLayer(BaseLayer):
    def __init__(self, prev_layer: BaseLayer) -> None:
        super().__init__(prev_layer, prev_layer.neurons)
        self.D = prev_layer.neurons
        self.eps = 1e-5
        self.gamma = np.ones((1, self.D))
        self.beta = np.zeros((1, self.D))

        self.x: np.ndarray
        self.mean: np.double
        self.var: np.double
        self.x_norm: np.ndarray

        self.id = f"LayerNorm{self.number}"

    @override
    def forward(self) -> None:
        assert self.prev_layer != None

        x = self.prev_layer.output
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)  # Normalize
        out = self.gamma * x_norm + self.beta  # Scale and shift

        # Store values for backprop
        self.x = x
        self.mean = mean
        self.var = var
        self.x_norm = x_norm

        self.output = out

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        d_gamma = np.sum(dA * self.x_norm, axis=(0, 1), keepdims=True)  # (1, 1, D)
        d_beta = np.sum(dA, axis=(0, 1), keepdims=True)  # (1, 1, D)

        optimizer.update(self.gamma, d_gamma, self.id + "gamma")
        optimizer.update(self.beta, d_beta, self.id + "beta")

        # Gradient w.r.t. normalized input
        d_x_norm = dA * self.gamma  # (N, T, D)

        # Compute gradients for mean and variance
        std_inv = 1.0 / np.sqrt(self.var + self.eps)  # (N, T, 1)
        d_var = np.sum(
            d_x_norm * (self.x - self.mean) * -0.5 * std_inv**3, axis=-1, keepdims=True
        )  # (N, T, 1)
        d_mean = (
            np.sum(d_x_norm * -std_inv, axis=-1, keepdims=True)
            + d_var * np.sum(-2 * (self.x - self.mean), axis=-1, keepdims=True) / self.D
        )  # (N, T, 1)

        # Compute gradient w.r.t. input
        d_x = (
            d_x_norm * std_inv
            + d_var * 2 * (self.x - self.mean) / self.D
            + d_mean / self.D
        )  # (N, T, D)

        return d_x
