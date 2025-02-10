from typing_extensions import override
from model.optimizers.base_optimizer import BaseOptimizer
import numpy as np


class GradientDescentOpt(BaseOptimizer):
    def __init__(self, learning_rate: np.double) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    @override
    def update(self, weights: np.ndarray, grads: np.ndarray, id: str) -> np.ndarray:
        return weights - self.learning_rate * grads
