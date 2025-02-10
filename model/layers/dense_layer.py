from typing_extensions import override
from model.layers.base_layer import BaseLayer
from model.layers.activation import activation_forward, activation_backward
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


class DenseLayer(BaseLayer):
    def __init__(
        self,
        prev_layer: BaseLayer,
        neurons: int,
        act_forward: activation_forward,
        act_backward: activation_backward,
    ) -> None:
        super().__init__(prev_layer, neurons)

        self.act_forward = act_forward
        self.act_backward = act_backward

        self.weights = np.random.random((prev_layer.neurons, neurons))
        self.bias = np.random.random((1, neurons))

        self.z: np.ndarray

    @override
    def forward(self) -> None:
        assert self.prev_layer != None

        x = self.prev_layer.output

        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.act_forward(self.z)

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        assert self.prev_layer != None

        dZ = self.act_backward(self.z, dA)

        dX = np.dot(dZ, self.weights.T)

        x = self.prev_layer.output
        if len(x.shape) == 2:  # (N, D)
            dW = np.dot(x.T, dZ)
        else:  # (N, T, D)
            dW = np.einsum("ntd,ntD->dD", x, dZ)  # Sum over N and T
        self.weights = optimizer.update(self.weights, dW)

        dB = np.sum(
            dZ, axis=(0, 1) if len(x.shape) == 3 else 0
        )  # Sum over N (and T if 3D)

        self.bias = optimizer.update(self.bias, dB)

        return dX

    @override
    def reset(self) -> None:
        assert self.prev_layer != None

        self.weights = np.random.random((self.prev_layer.neurons, self.neurons))
        self.bias = np.random.random((1, self.neurons))
