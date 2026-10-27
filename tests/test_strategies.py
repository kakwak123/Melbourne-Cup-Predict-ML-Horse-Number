import numpy as np
import pandas as pd
import pytest

from src import strategies
from src.strategies import SCENARIOS, build_frame, placegetter_ranks, settle, strategy_bets


def make_year(order, n=20, year=2025):
    """Field of n runners ranked by price (rank 0 = favourite). `order` = ranks of 1st, 2nd, 3rd."""
    pos = {r: i + 1 for i, r in enumerate(order)}
    rest = iter(range(4, n + 1))
    positions = [pos[r] if r in pos else next(rest) for r in range(n)]
    return pd.DataFrame({
        "year": year, "horse_number": np.arange(1, n + 1), "horse_name": [f"H{r}" for r in range(n)],
        "finishing_position": positions,
        "bsp": 3.0 + 3.0 * np.arange(n), "place_bsp": 1.5 + 0.5 * np.arange(n),
    })


def fair_win_price(g, rank, m):
    inv = 1 / g["bsp"]
    return 1 / (inv.iloc[rank] / inv.sum() * m)


def test_favourite_win_pays_normalised_price_when_it_wins():
    g = make_year([0, 5, 9])
    sc = SCENARIOS["exchange"]
    staked, returned = settle([("win", (0,), 100.0)], g, sc)
    assert staked == 100.0
    assert returned == pytest.approx(100.0 * fair_win_price(g, 0, sc["win_m"]))


def test_losing_favourite_returns_nothing():
    g = make_year([4, 5, 9])
    assert settle([("win", (0,), 100.0)], g, SCENARIOS["exchange"])[1] == 0.0


def test_place_bet_pays_when_runner_finishes_top_three():
    g = make_year([7, 0, 9])
    assert settle([("place", (0,), 100.0)], g, SCENARIOS["exchange"])[1] > 100.0 * 0.5
    g2 = make_year([7, 8, 9])
    assert settle([("place", (0,), 100.0)], g2, SCENARIOS["exchange"])[1] == 0.0


def test_bookmaker_margin_lowers_every_payout():
    g = make_year([0, 1, 2])
    for bets in ([("win", (0,), 100.0)], [("place", (0,), 100.0)], [("tri", (0, 1, 2, 3), 100.0)]):
        assert settle(bets, g, SCENARIOS["bookmaker"])[1] < settle(bets, g, SCENARIOS["exchange"])[1]


@pytest.mark.parametrize("n", [17, 20, 24])
def test_every_strategy_stakes_exactly_100(n):
    g = make_year([0, 1, 2], n=n)
    for name, bets in strategy_bets(n).items():
        staked, _ = settle(bets, g, SCENARIOS["exchange"])
        assert staked == pytest.approx(100.0), name


def test_trifecta_box_pays_only_when_all_three_finishers_are_in_the_box():
    g = make_year([1, 3, 2])
    sc = SCENARIOS["exchange"]
    hit = settle([("tri", (0, 1, 2, 3), 120.0)], g, sc)[1]
    assert hit > 0
    inv = 1 / g["bsp"]
    pw = (inv / inv.sum()).to_numpy()
    p = pw[1] * pw[3] / (1 - pw[1]) * pw[2] / (1 - pw[1] - pw[3])
    assert hit == pytest.approx(120.0 / 24 * (1 - sc["takeout"]) / p)
    assert settle([("tri", (0, 1, 2), 120.0)], g, sc)[1] == 0.0   # rank 3 is missing from the box


def test_field_smaller_than_17_raises():
    with pytest.raises(ValueError, match="at least 17"):
        strategy_bets(16)


def test_unknown_bet_kind_raises():
    with pytest.raises(ValueError, match="unknown bet kind"):
        settle([("exacta", (0, 1), 10.0)], make_year([0, 1, 2]), SCENARIOS["exchange"])


def test_build_frame_raises_on_runner_without_prices():
    g = make_year([0, 1, 2])
    market = g[["year", "horse_number", "bsp", "place_bsp"]].iloc[:-1]
    with pytest.raises(ValueError, match="2025 #20"):
        build_frame(g[["year", "horse_number", "horse_name", "finishing_position"]], market)


def test_placegetter_ranks_reports_market_rank_of_top_three():
    out = placegetter_ranks(make_year([4, 0, 9]))
    assert out["market_rank"].tolist() == [5, 1, 10]
