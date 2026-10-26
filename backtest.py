"""
Leave-one-year-out backtest for Melbourne Cup top-3 prediction.

Uses only fields that are real in every year (weight, barrier, finishing position),
so every strategy is scored on the same inputs. Compares simple baselines against
a logistic model that is trained only on years before the one being tested.
"""

import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def load() -> pd.DataFrame:
    files = sorted(glob.glob("data/historical/*.csv") + glob.glob("data/backtest/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[["year", "horse_name", "barrier", "weight", "finishing_position"]].dropna()
    df["top3"] = (df["finishing_position"] <= 3).astype(int)
    # Weight relative to the field is what matters: handicaps shift year to year.
    g = df.groupby("year")
    df["w_rel"] = (df["weight"] - g["weight"].transform("mean")) / g["weight"].transform("std")
    df["b_rel"] = df["barrier"] / g["barrier"].transform("max")
    return df

def score(year_df: pd.DataFrame, picks: pd.Series) -> dict:
    top3_pred = year_df.loc[picks.sort_values(ascending=False).index[:3]]
    winner = year_df[year_df["finishing_position"] == 1]
    win_rank = int(picks.rank(ascending=False)[winner.index[0]])
    return {"hits": int(top3_pred["top3"].sum()), "winner_rank": win_rank,
            "winner_in_top3": int(win_rank <= 3), "n": len(year_df)}

def main() -> None:
    df = load()
    years = sorted(df["year"].unique())
    rng = np.random.default_rng(0)
    strategies = {
        "random": lambda tr, te: pd.Series(rng.random(len(te)), index=te.index),
        "lowest weight": lambda tr, te: -te["weight"],
        "highest weight": lambda tr, te: te["weight"],
        "low barrier": lambda tr, te: -te["barrier"],
        "wide barrier": lambda tr, te: te["barrier"],
    }

    def logistic(tr, te):
        m = LogisticRegression(C=1.0).fit(tr[["w_rel", "b_rel"]], tr["top3"])
        return pd.Series(m.predict_proba(te[["w_rel", "b_rel"]])[:, 1], index=te.index)
    strategies["logistic(weight,barrier)"] = logistic

    rows = []
    for name, fn in strategies.items():
        for y in years:
            te = df[df["year"] == y]
            tr = df[df["year"] < y]
            if name.startswith("logistic") and tr["year"].nunique() < 3:
                continue  # not enough history yet
            rows.append({"strategy": name, "year": y, **score(te, fn(tr, te))})
    res = pd.DataFrame(rows)
    out = res.groupby("strategy").agg(
        races=("year", "count"),
        avg_top3_hits=("hits", "mean"),
        winner_in_top3=("winner_in_top3", "mean"),
        avg_winner_rank=("winner_rank", "mean"),
    ).round(2)
    print(out.sort_values("avg_top3_hits", ascending=False).to_string())
    print(f"\nRandom expectation: {3*3/df.groupby('year').size().mean():.2f} top-3 hits, "
          f"winner in top 3 {3/df.groupby('year').size().mean():.0%}, winner rank {(df.groupby('year').size().mean()+1)/2:.1f}")

if __name__ == "__main__":
    main()
