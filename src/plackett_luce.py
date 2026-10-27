"""Plackett-Luce (exploded conditional logit) truncated at third place."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

DEPTH = 3


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def race_nll(beta: np.ndarray, X: np.ndarray, positions: np.ndarray, depth: int = DEPTH) -> float:
    """Negative log-likelihood of the first `depth` finishers in one race."""
    scores = X @ beta
    order = np.argsort(positions, kind="stable")[:depth]
    remaining = np.ones(len(scores), dtype=bool)
    nll = 0.0
    for idx in order:
        s = scores[remaining]
        m = s.max()
        log_z = m + np.log(np.exp(s - m).sum())
        nll -= scores[idx] - log_z
        remaining[idx] = False
    return float(nll)


def fit(races: List[Tuple[np.ndarray, np.ndarray]], l2: float = 1.0) -> np.ndarray:
    """Fit coefficients on (X, positions) races with an L2 penalty."""
    n_features = races[0][0].shape[1]
    if n_features == 0:
        return np.zeros(0)

    def objective(beta):
        return sum(race_nll(beta, X, pos) for X, pos in races) + 0.5 * l2 * float(beta @ beta)

    return minimize(objective, np.zeros(n_features), method="L-BFGS-B").x


def top3_probs(p: np.ndarray) -> np.ndarray:
    """P(horse finishes in the top three) by exact enumeration of the PL model."""
    n = len(p)
    out = p.copy()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            out[i] += p[j] * p[i] / (1.0 - p[j])
            for k in range(n):
                if k == i or k == j:
                    continue
                out[i] += p[j] * p[k] / (1.0 - p[j]) * p[i] / max(1.0 - p[j] - p[k], 1e-12)
    return out
