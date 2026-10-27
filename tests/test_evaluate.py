import numpy as np
import pandas as pd

from src import evaluate
from src.features import add_race_features


def synthetic(years=range(2015, 2026), n=10, market_from=2020, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        w = rng.normal(54, 2, n)
        b = rng.permutation(n) + 1
        strength = -0.3 * (w - 54) + rng.normal(0, 1, n)
        pos = np.argsort(np.argsort(-(strength - np.log(-np.log(rng.random(n)))))) + 1
        bsp = np.exp(-strength) * 8 + 1.5 if y >= market_from else np.full(n, np.nan)
        for i in range(n):
            rows.append(dict(year=y, horse_number=i + 1, weight=w[i], barrier=b[i],
                             finishing_position=pos[i], bsp=bsp[i]))
    return add_race_features(pd.DataFrame(rows))


def test_logo_never_trains_on_the_held_out_year(monkeypatch):
    df = synthetic()
    seen = []
    real_fit = evaluate.fit

    def spy(races, l2=1.0):
        seen.append(len(races))
        return real_fit(races, l2=l2)

    monkeypatch.setattr(evaluate, "fit", spy)
    evaluate.logo_pl(df, ["w_z", "b_z"])
    assert set(seen) == {df["year"].nunique() - 1}


def test_incomplete_market_year_is_excluded():
    df = synthetic()
    df.loc[(df.year == 2023) & (df.horse_number == 1), "bsp"] = np.nan
    df = add_race_features(df.drop(columns=["w_z", "b_z", "log_mkt"]))
    years = evaluate.market_years(df)
    assert 2023 not in years and 2024 in years


def test_summarise_has_baseline_row_of_zero():
    df = synthetic()
    m = df[df.year.isin(evaluate.market_years(df))]
    scores = {"market": evaluate.market_scores(m), "uniform": evaluate.logo_pl(m, [])}
    table = evaluate.summarise(scores, "market")
    base = table[table.model == "market"].iloc[0]
    assert base["top3_ll_minus_market"] == 0
