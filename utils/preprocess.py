from typing import List, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from numpy.core.multiarray import ndarray
import sklearn.utils
import numpy as np

stemmer = PorterStemmer()


def preprocess_sequence(sequence: str) -> List[str]:
    """
    Preprocesses a text sequence:
    - Tokenizes
    - Removes punctuation & stopwords

    Args:
        sequence (str): Input text sequence

    Returns:
        np.ndarray: Preprocessed sequence of shape (max_len, embedding_dim)
    """

    stop_words = set(stopwords.words("english"))  # Get stopwords

    # Tokenize & Lowercase
    words = word_tokenize(sequence.lower())

    # Remove Punctuation & Stopwords, Stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and word not in string.punctuation
    ]

    return words


def preprocess_ds(sequences: List[str]) -> List[List[str]]:
    """
    Preprocess a list of sequences

    Args:
        sequences: the list of sequences to be processed
    return:
        (List[List[str]]): the processed list
    """
    return [preprocess_sequence(seq) for seq in sequences]


def train_test_split(
    x: np.ndarray, y: np.ndarray, train_ratio: float, shuffle: bool = True
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Split data into train and test set

    Args:
        x, y: input data
        train_ratio: train/test ratio
        shuffle: whether or not to shuffle the dataset first
    Returns:
        x_train, y_train, x_test, y_test
    """
    assert x.shape[0] == y.shape[0], f"Different number of samples in x and y!"

    N = x.shape[0]
    Nx = int(np.ceil(N * train_ratio))

    if shuffle:
        p = np.random.permutation(N)
        x = x[p]
        y = y[p]

    return x[:Nx], y[:Nx], x[Nx:], y[Nx:]
