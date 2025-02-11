from typing import Tuple
from typing_extensions import override
from model.layers.base_layer import BaseLayer
import numpy as np

from model.optimizers.base_optimizer import BaseOptimizer


class MultiheadAttentionLayer(BaseLayer):
    def __init__(self, prev_layer: BaseLayer, neurons: int, num_heads: int) -> None:
        super().__init__(prev_layer, neurons)

        assert neurons % num_heads == 0, "neurons must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = neurons // num_heads

        # Initialize learnable weights
        self.W_q = np.random.randn(neurons, neurons)
        self.W_k = np.random.randn(neurons, neurons)
        self.W_v = np.random.randn(neurons, neurons)
        self.W_o = np.random.randn(neurons, neurons)

        self.cache: Tuple

        self.id = f"MHA{self.number}"

    @override
    def forward(self) -> None:
        assert self.prev_layer is not None

        X = self.prev_layer.output
        N, T, D = X.shape
        h, d_k = self.num_heads, self.d_k

        # Compute Q, K, V
        Q = X @ self.W_q  # (N, T, D)
        K = X @ self.W_k  # (N, T, D)
        V = X @ self.W_v  # (N, T, D)

        # Split into multiple heads: (N, T, D) -> (N, h, T, d_k)
        Q = Q.reshape(N, T, h, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(N, T, h, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(N, T, h, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # (N, h, T, T)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

        # Apply attention weights
        context = attn_weights @ V  # (N, h, T, d_k)

        # Concatenate heads and apply final linear transformation
        context = context.transpose(0, 2, 1, 3).reshape(N, T, D)
        self.output = context @ self.W_o  # (N, T, D)

        self.cache = (X, Q, K, V, attn_weights, context)

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:

        X, Q, K, V, attn_weights, context = self.cache
        N, T, D = X.shape
        h, d_k = self.num_heads, self.d_k

        # Gradient w.r.t. W_o
        dL_dW_o = context.reshape(N * T, D).T @ dA.reshape(N * T, D)

        # Gradient w.r.t. concatenated context
        dL_dContext = dA @ self.W_o.T
        dL_dContext = dL_dContext.reshape(N, T, h, d_k).transpose(0, 2, 1, 3)

        # Gradients w.r.t. attention weights
        dL_dAttn = dL_dContext @ V.transpose(0, 1, 3, 2)

        # Gradient w.r.t. V
        dL_dV = attn_weights.transpose(0, 1, 3, 2) @ dL_dContext

        # Gradient w.r.t. Q, K
        dL_dQ = dL_dAttn @ K
        dL_dK = Q.transpose(0, 1, 3, 2) @ dL_dAttn

        # Backprop to weight matrices
        dL_dW_q = X.reshape(N * T, D).T @ dL_dQ.reshape(N * T, D)
        dL_dW_k = X.reshape(N * T, D).T @ dL_dK.reshape(N * T, D)
        dL_dW_v = X.reshape(N * T, D).T @ dL_dV.reshape(N * T, D)

        optimizer.update(self.W_q, dL_dW_q, self.id + "WQ")
        optimizer.update(self.W_k, dL_dW_k, self.id + "WK")
        optimizer.update(self.W_v, dL_dW_v, self.id + "WV")
        optimizer.update(self.W_o, dL_dW_o, self.id + "WO")

        dL_dQ = dL_dQ.transpose(0, 2, 1, 3).reshape(
            N, T, D
        )  # Reshape back to (N, T, D)
        dL_dK = dL_dK.transpose(0, 2, 1, 3).reshape(N, T, D)
        dL_dV = dL_dV.transpose(0, 2, 1, 3).reshape(N, T, D)

        dL_dX = (
            np.dot(dL_dQ, self.W_q.T)
            + np.dot(dL_dK, self.W_k.T)
            + np.dot(dL_dV, self.W_v.T)
        )
        # dL_dX = dL_dX.reshape(N, T, D)  # Reshape back to input shape

        return dL_dX
