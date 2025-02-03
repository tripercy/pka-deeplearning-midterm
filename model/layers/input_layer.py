import numpy as np

from model.layers.base_layer import BaseLayer


class InputLayer(BaseLayer):
    def __init__(self, neurons: int) -> None:
        super().__init__(None, neurons)
        self.data = None

    def feed_input(self, data: np.ndarray) -> None:
        """
        Feed an input batch into the layer

        Args:
            data (np.ndarray): the input batch
        Returns:
            None
        """
        assert data.shape[1] == self.neurons
        self.output = data
