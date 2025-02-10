from typing import Tuple
import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import ArrayLike


def read_imdb(path: str) -> Tuple[ArrayLike, ArrayLike, pd.Index]:
    """
    Load the IMDb gener classification dataset, shoule have been downloaded from
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
