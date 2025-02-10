from typing import Optional
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


class BaseLayer:
    def __init__(self, prev_layer: Optional["BaseLayer"], neurons: int) -> None:
        self.prev_layer = prev_layer
        self.neurons = neurons

        self.output: np.ndarray

    def forward(self) -> None:
        pass

    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        pass

        return np.zeros((0))

    def reset(self) -> None:
        pass
