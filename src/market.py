"""Convert betting odds into race-normalised probabilities."""

import numpy as np


def implied_probs(odds) -> np.ndarray:
    """Win probabilities from decimal odds, with the overround removed."""
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("odds must be finite decimal odds greater than 1")
    raw = 1.0 / odds
    return raw / raw.sum()
