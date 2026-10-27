"""Per-race scoring rules and a race-level paired bootstrap."""

from typing import Sequence, Tuple

import numpy as np


def winner_log_loss(p: np.ndarray, positions: np.ndarray) -> float:
    """-log P(actual winner). Uniform over 24 runners is ln 24, about 3.18."""
    return float(-np.log(p[int(np.argmin(positions))]))


def top3_log_loss(p: np.ndarray, positions: np.ndarray) -> float:
    """-log PL likelihood of the actual 1-2-3 finishers, in order."""
    first, second, third = np.argsort(positions, kind="stable")[:3]
    ll = np.log(p[first])
    ll += np.log(p[second] / (1.0 - p[first]))
    ll += np.log(p[third] / (1.0 - p[first] - p[second]))
    return float(-ll)


def paired_bootstrap(diffs: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean and 95% CI of per-race differences, resampling races (never rows)."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
