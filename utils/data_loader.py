from typing import Any, Tuple
import pandas as pd
import numpy as np
import os
from numpy.typing import ArrayLike


def read_imdb(path: str) -> Tuple[ArrayLike | Any, ArrayLike, ArrayLike]:
    """
    Load the IMDb genre classification dataset, shoule have been downloaded from
    "https://www.kaggle.com/code/soundslikedata/genre-classification-notebook"

    Args:
        path: the path to a data file, whose format should be: ID ::: TITLE ::: GENRE ::: DESCRIPTION

    Returns:
        ArrayLike: ndarray of strings, the descriptions
        ArrayLike: 2D ndarray of double, the one-hot encoded labels
        Index    : the list of labels' names, correspond to the one-hot encoded y
    """
    df = pd.read_csv(
        path, sep=" ::: ", header=None, names=["id", "title", "genre", "description"]
    )
    one_hot = pd.get_dummies(df["genre"], dtype=np.double)

    return df["description"].values, one_hot.values, one_hot.columns


def read_bbc(base_path: str) -> Tuple[ArrayLike | Any, ArrayLike, ArrayLike]:
    """
    Load the BBC Text classification dataset, should have been downloaded from
    "https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data"

    Args:
        path: the path to the base directory of the dataset, should contains the directories listed in
        labels below

    Returns:
        ArrayLike: ndarray of strings, the descriptions
        ArrayLike: 2D ndarray of double, the one-hot encoded labels
        Index    : the list of labels' names, correspond to the one-hot encoded y
    """
    labels = ["business", "entertainment", "politics", "sport", "tech"]
    x = []
    y = []
    C = len(labels)
    for i, label in enumerate(labels):
        path = os.path.join(base_path, label)
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            with open(filepath, encoding="utf8", errors="ignore") as f:
                content = f.readlines()
                content = " ".join(content)
                x.append(content)
                onehot = [0 if j != i else 1 for j in range(C)]
                y.append(onehot)

    return np.array(x), np.array(y), labels
