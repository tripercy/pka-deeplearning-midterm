from typing import TypeAlias, Callable
import numpy as np

activation_forward: TypeAlias = Callable[[np.ndarray], np.ndarray]
activation_backward: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]


def relu_forward(x: np.ndarray) -> np.ndarray:
    """
    Forward RELU function

    Args:
        x (np.ndarray): numpy array of any shape
    Returns:
        np.ndarray: numpy array of the same shape as x
    """

    return np.maximum(0, x)


def relu_backward(z: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """
    Derivative of RELU

    Args:
        z (np.ndarray): numpy array of any shape, the cached input of RELU
        dA (np.ndarray): numpy array of the same shape as z, result of dL/dA
    Returns:
        np.ndarray: numpy array of the same shape as z
    """

    assert z.shape == dA.shape

    dz = np.array(dA, copy=True)
    dz[z <= 0] = 0

    return dz


def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """
    Forward Sigmoid function

    Args:
        x (np.ndarray): numpy array of any shape
    Returns:
        np.ndarray: numpy array of the same shape as x
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_backward(z: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """
    Derivative of Sigmoid

    Args:
        z (np.ndarray): numpy array of any shape, the cached input of Sigmoid
        dA (np.ndarray): numpy array of the same shape as z, result of dL/dA
    Returns:
        np.ndarray: numpy array of the same shape as z
    """

    assert z.shape == dA.shape, f"Shape mismatched: z: {z.shape} and dA: {dA.shape}"

    s = sigmoid_forward(z)
    dz = dA * s * (1 - s)

    return dz


def softmax_forward(x: np.ndarray) -> np.ndarray:
    """
    Forward Softmax function

    Args:
        x (np.ndarray): numpy array of any shape
    Returns:
        np.ndarray: numpy array of the same shape as x
    """

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_backward(z: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """
    Derivative of Softmax

    Args:
        z (np.ndarray): numpy array of any shape, the cached input of Softmax
        dA (np.ndarray): numpy array of the same shape as z, result of dL/dA
    Returns:
        np.ndarray: numpy array of the same shape as z
    """
    assert z.shape == dA.shape

    y = softmax_forward(z)
    dL_dz = y * (dA - np.sum(dA * y, axis=1, keepdims=True))
    return dL_dz
