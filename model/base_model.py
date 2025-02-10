from typing import Any, Generator, List, Tuple

from model.layers.base_layer import BaseLayer
import numpy as np
import tqdm

from model.layers.input_layer import InputLayer
from model.layers.loss_function import loss_func, loss_grad
from model.metrics import accuracy, precision, recall
from model.optimizers.base_optimizer import BaseOptimizer


class BaseModel:
    def __init__(
        self,
        input_layer: InputLayer,
        output_layer: BaseLayer,
        batch_size: int,
        optimizer: BaseOptimizer,
        loss: loss_func,
        loss_grad: loss_grad,
    ) -> None:
        layer = output_layer
        layers: List[BaseLayer] = [layer]

        while layer.prev_layer != None:
            layer = layer.prev_layer
            layers.append(layer)

        self.layers: List[BaseLayer] = list(reversed(layers))

        assert layer == input_layer
        self.input_layer: InputLayer = input_layer
        self.output_layer: BaseLayer = output_layer

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_func = loss
        self.loss_grad = loss_grad
        self.history = []

    def get_batches(
        self, x: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], Any, Any]:
        N = x.shape[0]
        for i in range(0, N, self.batch_size):
            yield x[i : i + self.batch_size], y[i : i + self.batch_size]

    def reset(self) -> None:
        self.history = []
        for layer in self.layers:
            layer.reset()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 100) -> None:
        assert x_train.shape[-1] == self.input_layer.neurons
        self.reset()

        for i in range(epochs):
            epoch_loss = 0
            n_batches = np.ceil(x_train.shape[0] / self.batch_size)

            print(f"Epoch {i + 1}/{epochs}")
            # for x, y in self.get_batches(x_train, y_train):
            for x, y in tqdm.tqdm(self.get_batches(x_train, y_train), total=n_batches):
                self.input_layer.feed_input(x)

                # Forward pass
                for layer in self.layers[1:]:
                    layer.forward()

                epoch_loss += self.loss_func(y, self.output_layer.output)
                dA = self.loss_grad(y, self.output_layer.output)

                # Back propagation
                self.optimizer.tick()
                for layer in reversed(self.layers):
                    dA = layer.backward(dA, self.optimizer)

            epoch_loss /= n_batches
            print(f"Loss: {epoch_loss}")
            self.history.append(epoch_loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == self.input_layer.neurons

        self.input_layer.feed_input(x)
        for layer in self.layers:
            layer.forward()
            # print(layer.output.shape)

        return self.output_layer.output

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)

        return (
            accuracy(y_test, y_pred),
            precision(y_test, y_pred),
            recall(y_test, y_pred),
        )
