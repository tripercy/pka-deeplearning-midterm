from typing_extensions import override
from model.optimizers.base_optimizer import BaseOptimizer
import numpy as np


class AdamOpt(BaseOptimizer):
    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.momentum = {}
        self.variance = {}

    @override
    def update(self, weights: np.ndarray, grads: np.ndarray, id: str) -> np.ndarray:
        if id not in self.momentum:
            self.momentum[id] = np.zeros_like(weights)
            self.variance[id] = np.zeros_like(weights)

        m_t = self.momentum[id]
        v_t = self.variance[id]

        # Update biased first moment estimate
        m_t = self.beta1 * m_t + (1 - self.beta1) * grads

        # Update biased second moment estimate
        v_t = self.beta2 * v_t + (1 - self.beta2) * (grads**2)

        # Bias correction
        m_hat = m_t / (1 - self.beta1**self.iterations)
        v_hat = v_t / (1 - self.beta2**self.iterations)

        # Compute weight update
        weight_update = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        updated_weight = weights - weight_update
        self.momentum[id] = m_t
        self.variance[id] = v_t

        return updated_weight
