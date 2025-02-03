import numpy as np


class BaseOptimizer:
    def __init__(self) -> None:
        self.iterations = 0

    def update(self, weights: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Update weights

        Args:
            weights (ndarray): initial weigths
            grads (ndarray): gradients of the initial weights

        Returns:
            ndarray: the updated weights
        """
        pass
        return np.zeros((0, 0))
