from typing_extensions import override
from model.layers.activation import *
from model.layers.base_layer import BaseLayer
import numpy as np

from model.layers.dense_layer import DenseLayer
from model.layers.layernorm_layer import LayerNormLayer
from model.layers.multiheadattention_layer import MultiheadAttentionLayer
from model.layers.residual_layer import ResidualLayer
from model.optimizers.base_optimizer import BaseOptimizer


class TransformerEncoderBlock(BaseLayer):
    def __init__(self, prev_layer: BaseLayer, num_heads: int) -> None:

        neurons = prev_layer.neurons
        super().__init__(prev_layer, neurons)
        self.res1 = ResidualLayer(prev_layer)
        mha = MultiheadAttentionLayer(self.res1, neurons, num_heads)
        self.res1.add_skipped_layers(mha)
        self.norm1 = LayerNormLayer(self.res1)

        self.res2 = ResidualLayer(self.norm1)
        ff1 = DenseLayer(self.res2, neurons, relu_forward, relu_backward)
        ff2 = DenseLayer(ff1, neurons, relu_forward, relu_backward)
        self.res2.add_skipped_layers(ff2)
        self.norm2 = LayerNormLayer(self.res2)

    @override
    def forward(self) -> None:
        self.res1.forward()
        self.norm1.forward()
        self.res2.forward()
        self.norm2.forward()

        self.output = self.norm2.output

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        dA = self.norm2.backward(dA, optimizer)
        dA = self.res2.backward(dA, optimizer)
        dA = self.norm1.backward(dA, optimizer)
        dA = self.res1.backward(dA, optimizer)

        return dA


class TransformerEncoderLayer(BaseLayer):
    def __init__(self, prev_layer: BaseLayer, num_heads: int, num_blocks) -> None:
        neurons = prev_layer.neurons
        super().__init__(prev_layer, neurons)
        self.blocks = []

        last = prev_layer
        for _ in range(num_blocks):
            block = TransformerEncoderBlock(last, num_heads)
            self.blocks.append(block)
            last = block

    @override
    def forward(self) -> None:
        for block in self.blocks:
            block.forward()

        self.output = self.blocks[-1].output

    @override
    def backward(self, dA: np.ndarray, optimizer: BaseOptimizer) -> np.ndarray:
        for block in self.blocks[::-1]:
            dA = block.backward(dA, optimizer)

        return dA
