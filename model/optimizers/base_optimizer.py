import numpy as np


class BaseOptimizer:
    def __init__(self) -> None:
        self.iterations = 0

    def update(self, weights: np.ndarray, grads: np.ndarray, id: str) -> np.ndarray:
        """
        Update weights

        Args:
            weights (ndarray): initial weigths
            grads (ndarray): gradients of the initial weights
            id (str): the weights' ID

        Returns:
            ndarray: the updated weights
        """
        pass
        return np.zeros((0, 0))

    def tick(self) -> None:
        self.iterations += 1
