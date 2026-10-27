"""Leave-one-year-out evaluation of Cup models against the market."""

import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics
from src.betfair import attach_bsp
from src.features import add_race_features
from src.plackett_luce import fit, softmax
from src.validate import validate_race

L2 = 1.0


def load_verified(with_market: bool = True) -> pd.DataFrame:
    """Verified results, plus BSP for the years Betfair covers (NaN elsewhere)."""
    frames = []
    for path in sorted(glob.glob("data/verified/melbourne_cup_*.csv")):
        df = pd.read_csv(path)
        problems = validate_race(df)
        if problems:
            raise ValueError(f"{path}: {problems}")
        frames.append(df)
    results = pd.concat(frames, ignore_index=True)
    if with_market and os.path.exists("data/market/cup_bsp.csv"):
        bsp = pd.read_csv("data/market/cup_bsp.csv")
        covered = results["year"].isin(bsp["year"])
        with_bsp = attach_bsp(results[covered], bsp)
        results = pd.concat([results[~covered], with_bsp], ignore_index=True)
    return add_race_features(results)


def market_years(df: pd.DataFrame) -> List[int]:
    """Years in which every runner has a usable market probability."""
    ok = df.groupby("year")["log_mkt"].apply(lambda s: s.notna().all())
    return sorted(int(y) for y in ok[ok].index)


def _race_arrays(df: pd.DataFrame, cols: List[str]) -> Dict[int, tuple]:
    return {
        int(y): (g[cols].to_numpy(dtype=float), g["finishing_position"].to_numpy(dtype=float))
        for y, g in df.groupby("year")
    }


def logo_pl(df: pd.DataFrame, cols: List[str], l2: float = L2) -> pd.DataFrame:
    """Train on every other year, score the held-out year. No test-year leakage."""
    races = _race_arrays(df, cols)
    rows = []
    for y, (X, pos) in races.items():
        train = [r for yy, r in races.items() if yy != y]
        beta = fit(train, l2=l2)
        p = softmax(X @ beta) if len(beta) else np.full(len(pos), 1.0 / len(pos))
        rows.append({"year": y, "win_ll": metrics.winner_log_loss(p, pos),
                     "top3_ll": metrics.top3_log_loss(p, pos)})
    return pd.DataFrame(rows).set_index("year").sort_index()


def market_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in df.groupby("year"):
        p = np.exp(g["log_mkt"].to_numpy())
        pos = g["finishing_position"].to_numpy(dtype=float)
        rows.append({"year": int(y), "win_ll": metrics.winner_log_loss(p, pos),
                     "top3_ll": metrics.top3_log_loss(p, pos)})
    return pd.DataFrame(rows).set_index("year").sort_index()


def summarise(scores: Dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        d = s["top3_ll"] - scores[baseline]["top3_ll"]
        mean, lo, hi = metrics.paired_bootstrap(d.values)
        rows.append({"model": name, "races": len(s), "win_ll": s["win_ll"].mean(),
                     "top3_ll": s["top3_ll"].mean(), f"top3_ll_minus_{baseline}": mean,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).round(3)


def build_report(df: pd.DataFrame) -> str:
    all_scores = {"uniform": logo_pl(df, []), "fundamental": logo_pl(df, ["w_z", "b_z"])}
    all_table = summarise(all_scores, "uniform")

    m = df[df["year"].isin(market_years(df))]
    mk_scores = {
        "uniform": logo_pl(m, []),
        "market": market_scores(m),
        "fundamental": logo_pl(m, ["w_z", "b_z"]),
        "combined": logo_pl(m, ["w_z", "b_z", "log_mkt"]),
    }
    mk_table = summarise(mk_scores, "market")

    comb = mk_table[mk_table.model == "combined"].iloc[0]
    diff = comb["top3_ll_minus_market"]
    verdict = (
        f"combined {'beat' if diff < 0 else 'did not beat'} the market on mean top-3 log loss "
        f"(difference {diff:+.3f} nats, 95% CI [{comb['ci_lo']:+.3f}, {comb['ci_hi']:+.3f}]); "
        f"the CI {'excludes' if (comb['ci_lo'] > 0 or comb['ci_hi'] < 0) else 'includes'} 0."
    )
    return "\n".join([
        "Melbourne Cup evaluation (leave-one-year-out; lower log loss is better)",
        f"Uniform winner log loss for a 24-runner field is ln 24 = {np.log(24):.3f}.",
        "",
        f"All verified years ({df['year'].nunique()} races), baseline = uniform",
        all_table.to_string(index=False),
        "",
        f"Market years {market_years(df)}, baseline = market",
        mk_table.to_string(index=False),
        "",
        "Verdict: " + verdict,
    ])


def main() -> None:
    report = build_report(load_verified())
    print(report)
    os.makedirs("docs", exist_ok=True)
    with open("docs/evaluation_report.md", "w") as fh:
        fh.write("```\n" + report + "\n```\n")


if __name__ == "__main__":
    main()
