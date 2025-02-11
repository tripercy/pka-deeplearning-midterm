from typing import List, Optional
from typing_extensions import override
from model.layers.base_layer import BaseLayer
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


# Assuming the layers skipped always output the same dimension as input
class ResidualLayer(BaseLayer):

    def __init__(self, prev_layer: BaseLayer) -> None:
        super().__init__(prev_layer, prev_layer.neurons)

        self.skipped_layers: List[BaseLayer] = []

    @override
    def forward(self) -> None:
        assert self.prev_layer is not None
        assert len(self.skipped_layers) > 0, "Residual layer empty!"

        x = self.prev_layer.output
        self.output = x
        for layer in self.skipped_layers:
            layer.forward()

        self.output = x + self.skipped_layers[-1].output

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        dY = dA
        for layer in self.skipped_layers[::-1]:
            dA = layer.backward(dA, optimizer)

        return dY + dA

    def add_skipped_layers(self, last: Optional[BaseLayer]) -> None:

        self.skipped_layers = []
        while last is not None and last != self:
            self.skipped_layers.append(last)
            last = last.prev_layer

        assert (
            last == self
        ), "The first layer of skipped connection should be connected to ResidualLayer"
        self.skipped_layers.reverse()
