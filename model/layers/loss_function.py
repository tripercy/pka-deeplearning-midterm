from typing import Callable, TypeAlias
import numpy as np

loss_func: TypeAlias = Callable[[np.ndarray, np.ndarray], np.double]
loss_grad: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.double:

    assert (
        y_true.shape == y_pred.shape
    ), f"Expected y_pred to be shape {y_true.shape}, got {y_pred.shape}"

    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]


def cross_entropy_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    assert (
        y_true.shape == y_pred.shape
    ), f"Expected y_pred to be shape {y_true.shape}, got {y_pred.shape}"

    return -1 * y_true / (y_pred * y_pred.shape[0])
