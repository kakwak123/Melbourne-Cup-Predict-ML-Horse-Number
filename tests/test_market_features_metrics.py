import numpy as np
import pandas as pd
import pytest

from src.features import add_race_features
from src.market import implied_probs
from src.metrics import paired_bootstrap, top3_log_loss, winner_log_loss


def test_implied_probs_removes_overround():
    p = implied_probs([2, 4, 4, 8])
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx([4 / 9, 2 / 9, 2 / 9, 1 / 9])


@pytest.mark.parametrize("bad", [[2, 1.0], [2, 0.5], [2, float("nan")], [2, float("inf")]])
def test_implied_probs_rejects_bad_odds(bad):
    with pytest.raises(ValueError):
        implied_probs(bad)


def test_zscore_constant_column_is_zero():
    df = pd.DataFrame({"year": [2025] * 3, "weight": [54.0] * 3, "barrier": [1, 2, 3]})
    out = add_race_features(df)
    assert (out["w_z"] == 0).all()
    assert out["b_z"].mean() == pytest.approx(0.0)


def test_log_mkt_only_for_complete_years():
    df = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "weight": [50, 52, 50, 52], "barrier": [1, 2, 1, 2],
        "bsp": [2.0, 4.0, 3.0, np.nan],
    })
    out = add_race_features(df)
    assert out.loc[out.year == 2024, "log_mkt"].notna().all()
    assert out.loc[out.year == 2025, "log_mkt"].isna().all()


def test_winner_log_loss_uniform_is_ln_24():
    p = np.full(24, 1 / 24)
    positions = np.arange(1, 25)
    assert winner_log_loss(p, positions) == pytest.approx(np.log(24))


def test_top3_log_loss_matches_formula():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    positions = np.array([2, 1, 3, 4])        # winner idx1, second idx0, third idx2
    expected = -(np.log(0.3) + np.log(0.5 / 0.7) + np.log(0.15 / 0.2))
    assert top3_log_loss(p, positions) == pytest.approx(expected)


def test_paired_bootstrap_is_deterministic_and_ordered():
    d = [0.1, -0.2, 0.05, 0.3, -0.1]
    a = paired_bootstrap(d, n_boot=2000, seed=1)
    assert a == paired_bootstrap(d, n_boot=2000, seed=1)
    mean, lo, hi = a
    assert lo <= mean <= hi
