from typing_extensions import override
from model.layers.base_layer import BaseLayer
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


class AverageLayer(BaseLayer):
    def __init__(self, prev_layer: BaseLayer) -> None:
        neurons = prev_layer.neurons
        super().__init__(prev_layer, neurons)

    @override
    def forward(self) -> None:
        assert self.prev_layer != None

        self.output = np.mean(self.prev_layer.output, axis=1)
        self.T = self.prev_layer.output.shape[1]

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        return np.repeat((1 / self.T) * np.expand_dims(dA, axis=1), self.T, axis=1)
