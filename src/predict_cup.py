"""Frozen Melbourne Cup prediction: market-only or combined, chosen by a rule fixed in code."""

import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import L2, _race_arrays, load_verified, logo_pl, market_scores, market_years
from src.market import implied_probs
from src.plackett_luce import fit, softmax, top3_probs

COMBINED_COLS = ["w_z", "b_z", "log_mkt"]


def choose_model(train: pd.DataFrame) -> str:
    """'combined' only if it beats the market on mean top-3 log loss under LOGO."""
    m = train[train["year"].isin(market_years(train))]
    combined = logo_pl(m, COMBINED_COLS)["top3_ll"].mean()
    market = market_scores(m)["top3_ll"].mean()
    return "combined" if combined < market else "market"


def _check_lineup(lineup: pd.DataFrame) -> None:
    bad = lineup[~(np.isfinite(lineup["odds"].astype(float)) & (lineup["odds"].astype(float) > 1.0))]
    if not bad.empty:
        raise ValueError(f"missing or invalid odds for: {', '.join(bad['horse_name'])}")
    for col in ("barrier", "weight"):
        missing = lineup[lineup[col].isna()]
        if not missing.empty:
            raise ValueError(f"missing {col} for: {', '.join(missing['horse_name'])}")


def predict_lineup(lineup: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    _check_lineup(lineup)
    model = choose_model(train)
    market_p = implied_probs(lineup["odds"].astype(float).values)

    if model == "market":
        p = market_p
    else:
        m = train[train["year"].isin(market_years(train))]
        beta = fit(list(_race_arrays(m, COMBINED_COLS).values()), l2=L2)
        z = lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0.0
        X = np.column_stack([z(lineup["weight"].astype(float)).to_numpy(),
                             z(lineup["barrier"].astype(float)).to_numpy(),
                             np.log(market_p)])
        p = softmax(X @ beta)

    out = pd.DataFrame({
        "horse_number": lineup["horse_number"].values,
        "horse_name": lineup["horse_name"].values,
        "p_win": p,
        "p_top3": top3_probs(p),
        "model": model,
    })
    return out.sort_values("p_top3", ascending=False).reset_index(drop=True)


def main(path: str) -> None:
    lineup = pd.read_csv(path)
    out = predict_lineup(lineup, load_verified())
    os.makedirs("predictions", exist_ok=True)
    out.to_csv("predictions/2026_melbourne_cup.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
