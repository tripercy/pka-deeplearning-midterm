from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

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
