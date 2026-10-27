import numpy as np
import pytest

from src.plackett_luce import fit, softmax, top3_probs


def simulate(n_races=300, n=12, true=(1.5, -0.5), seed=1):
    rng = np.random.default_rng(seed)
    true = np.array(true)
    races = []
    for _ in range(n_races):
        X = rng.normal(size=(n, len(true)))
        gumbel = -np.log(-np.log(rng.random(n)))
        positions = np.argsort(np.argsort(-(X @ true + gumbel))) + 1
        races.append((X, positions.astype(float)))
    return races


def test_fit_recovers_known_coefficients():
    beta = fit(simulate(), l2=0.1)
    assert beta == pytest.approx([1.5, -0.5], abs=0.15)


@pytest.mark.parametrize("n", [5, 24])
def test_top3_probs_sum_to_three_and_dominate_win_probs(n):
    p = softmax(np.linspace(0, 3, n))
    t = top3_probs(p)
    assert t.sum() == pytest.approx(3.0, abs=1e-9)
    assert (t >= p - 1e-12).all() and (t <= 1.0 + 1e-12).all()


def test_three_runner_field_everyone_places():
    assert top3_probs(np.array([0.5, 0.3, 0.2])) == pytest.approx([1, 1, 1])


def test_fit_with_no_features_returns_empty():
    races = [(np.zeros((6, 0)), np.arange(1.0, 7.0))]
    assert fit(races).shape == (0,)


def test_fit_handles_dead_heat_outside_top3():
    X = np.random.default_rng(0).normal(size=(8, 2))
    positions = np.array([1, 2, 3, 4, 5, 6, 6, 8], dtype=float)
    beta = fit([(X, positions)] * 5)
    assert np.isfinite(beta).all()
