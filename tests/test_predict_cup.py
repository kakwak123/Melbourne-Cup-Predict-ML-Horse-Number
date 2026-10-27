import numpy as np
import pandas as pd
import pytest

from src import predict_cup
from tests.test_evaluate import synthetic


def lineup(n=10, odds=None):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "horse_number": range(1, n + 1),
        "horse_name": [f"H{i}" for i in range(1, n + 1)],
        "barrier": rng.permutation(n) + 1,
        "weight": rng.normal(54, 2, n),
        "odds": odds if odds is not None else rng.uniform(3, 60, n),
    })


def test_probabilities_sum_correctly():
    out = predict_cup.predict_lineup(lineup(), synthetic())
    assert out["p_win"].sum() == pytest.approx(1.0)
    assert out["p_top3"].sum() == pytest.approx(3.0, abs=1e-9)
    assert out["model"].nunique() == 1


def test_lineup_with_missing_odds_raises():
    odds = [5.0] * 9 + [np.nan]
    with pytest.raises(ValueError, match="H10"):
        predict_cup.predict_lineup(lineup(odds=odds), synthetic())


def test_lineup_with_odds_of_one_raises():
    odds = [5.0] * 9 + [1.0]
    with pytest.raises(ValueError, match="H10"):
        predict_cup.predict_lineup(lineup(odds=odds), synthetic())


def test_market_only_when_combined_does_not_beat_it(monkeypatch):
    monkeypatch.setattr(predict_cup, "choose_model", lambda df: "market")
    out = predict_cup.predict_lineup(lineup(), synthetic())
    assert (out["model"] == "market").all()
