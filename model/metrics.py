import numpy as np


def accuracy(y_true, y_pred):
    """
    Compute accuracy for multiclass classification.

    Args:
        y_true: One-hot encoded ground truth labels, shape (N, C)
        y_pred: One-hot encoded predicted labels, shape (N, C)
    Returns:
        (double): Accuracy score
    """
    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total_samples = y_true.shape[0]
    return correct_predictions / total_samples


def precision(y_true, y_pred):
    """
    Compute precision for each class in multiclass classification.

    Args:
        y_true: One-hot encoded ground truth labels, shape (N, C)
        y_pred: One-hot encoded predicted labels, shape (N, C)
    Returns:
        ndarray:Array of precision values for each class
    """
    true_positives = np.sum(y_true * y_pred, axis=0)
    predicted_positives = np.sum(y_pred, axis=0)
    return np.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)


def recall(y_true, y_pred):
    """
    Compute recall for each class in multiclass classification.

    Args:
        y_true: One-hot encoded ground truth labels, shape (N, C)
        y_pred: One-hot encoded predicted labels, shape (N, C)
    Returns:
        ndarray: Array of recall values for each class
    """
    true_positives = np.sum(y_true * y_pred, axis=0)
    actual_positives = np.sum(y_true, axis=0)
    return np.where(actual_positives > 0, true_positives / actual_positives, 0.0)
