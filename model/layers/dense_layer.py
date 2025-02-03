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
        dW = np.dot(x.T, dZ)
        self.weights = optimizer.update(self.weights, dW)
        self.bias = optimizer.update(self.bias, dZ)

        return dX
