"""A complete plain Python extension, without product or UI imports."""

import numpy as np


def fit(X, epsilon=1e-8):
    """Standardize features using the reference data."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), epsilon)
    return lambda X_new: (X_new - mean) / scale
