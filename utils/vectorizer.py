from typing import Any, List
from nltk.tokenize import word_tokenize
import numpy as np


def vectorize_sequence_average(sequence: List[str], model: Any) -> np.ndarray:
    """
    Vectorize a sequence by averaging the embedding vectors of words in that sequence

    Args:
        sequence (List[str]): the list of word in the sequence to vectorize
        model (Any): gensim loaded word embedding model with embedding of size D

    Returns:
        (ndarray): array of shape (1, D): the embedding of the sequence
    """
    word_vectors = [model[word] for word in sequence if word in model]  # Get vectors
    return (
        np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    )


def vectorize_ds_average(sequences: List[List[str]], model: Any) -> np.ndarray:
    """
    Vectorize a list of sequences by averaging the embedding vectors of words in that sequence

    Args:
        sequences (List[List[str]]): the list of N sequences
        model (Any): gensim loaded word embedding model with embedding of size D

    Returns:
        (ndarray): array of shape (N, D): the embedding of the sequences
    """

    return np.array(
        list(
            map(lambda sequence: vectorize_sequence_average(sequence, model), sequences)
        )
    )


def vectorize_sequence(sequence: List[str], model: Any, max_len: int) -> np.ndarray:
    """
    Vectorize a sequence

    Args:
        sequence (List[str]): the list of words in the sequence to vectorize
        model (Any): gensim loaded word embedding model with embedding of size D
        max_len(int): maximum number of tokens

    Returns:
        (ndarray): array of shape (max_len, D): the embedding of the sequence
    """
    word_vectors = [model[word] for word in sequence if word in model]

    if len(word_vectors) < max_len:
        padding = np.zeros(
            (max_len - len(word_vectors), model.vector_size)
        )  # Zero padding
        word_vectors = np.vstack((word_vectors, padding))
    else:
        word_vectors = word_vectors[:max_len]

    return np.array(word_vectors)  # Shape: (T, D)


def vectorize_ds(sequences: List[List[str]], model: Any, max_len: int) -> np.ndarray:
    """
    Vectorize a list of sequences

    Args:
        sequences (List[List[str]]): the list of N sequences
        model (Any): gensim loaded word embedding model with embedding of size D
        max_len(int): maximum number of tokens for each sequence

    Returns:
        (ndarray): array of shape (N, max_len, D): the embedding of the sequences
    """

    return np.array(
        list(
            map(
                lambda sequence: vectorize_sequence(sequence, model, max_len), sequences
            )
        )
    )
