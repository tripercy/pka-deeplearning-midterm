import numpy as np


def add_positional_encoding(X):
    """
    Adds sinusoidal positional encoding to input tensor X, handling both even and odd D.

    Args:
        X: Input array of shape (N, T, D)

    Returns:
        ndarray: Input array with added positional encoding, shape (N, T, D)
    """
    N, T, D = X.shape
    D_pe = D if D % 2 == 0 else D + 1  # Ensure D_pe is even

    # Create position indices (shape: T)
    positions = np.arange(T)[:, np.newaxis]  # Shape (T, 1)

    # Compute frequency scaling
    div_term = np.exp(np.arange(0, D_pe, 2) * (-np.log(10000.0) / D_pe))

    # Compute positional encoding
    pos_enc = np.zeros((T, D_pe))
    pos_enc[:, 0::2] = np.sin(positions * div_term)  # Sin for even indices
    pos_enc[:, 1::2] = np.cos(positions * div_term)  # Cos for odd indices

    # Truncate if original D was odd
    pos_enc = pos_enc[:, :D]

    # Expand to match batch size (N, T, D)
    pos_enc = np.expand_dims(pos_enc, axis=0)  # Shape (1, T, D)

    return X + pos_enc
