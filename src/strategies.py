"""Replay betting strategies on past Melbourne Cups using market-derived prices.

Runners are addressed by market rank (0 = favourite, by Betfair BSP). Prices come from
the race-normalised BSP with a margin applied, so the exchange and bookmaker scenarios
show what the same bets would have returned at different price qualities.
"""

import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# win_m / place_m divide the fair price; takeout is the tote's cut of the trifecta pool.
SCENARIOS = {
    "exchange": {"win_m": 1.02, "place_m": 1.02, "takeout": 0.10},
    "bookmaker": {"win_m": 1.15, "place_m": 1.25, "takeout": 0.25},
}

Bet = Tuple[str, Tuple[int, ...], float]   # (kind, market ranks, stake); kind: win | place | tri


def build_frame(results: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join verified results to BSP / place BSP; every runner of a market year must match."""
    frame = results[results["year"].isin(market["year"])].merge(
        market[["year", "horse_number", "bsp", "place_bsp"]],
        on=["year", "horse_number"], how="left")
    missing = frame[frame["bsp"].isna() | frame["place_bsp"].isna()]
    if not missing.empty:
        names = ", ".join(f"{r.year} #{r.horse_number}" for r in missing.itertuples())
        raise ValueError(f"no market prices for: {names}")
    return frame


def load_frame() -> pd.DataFrame:
    results = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob("data/verified/melbourne_cup_*.csv"))],
        ignore_index=True)
    return build_frame(results, pd.read_csv("data/market/cup_bsp.csv"))


def strategy_bets(n_runners: int) -> Dict[str, List[Bet]]:
    """Every strategy stakes exactly $100. Names use 1-based market ranks."""
    if n_runners < 17:
        raise ValueError("strategies need a field of at least 17 runners")
    long_ranks = range(7, 17)
    return {
        "favourite win $100": [("win", (0,), 100.0)],
        "favourite place $100": [("place", (0,), 100.0)],
        "longshot spread $10 x ranks 8-17": [("win", (i,), 10.0) for i in long_ranks],
        "trifecta box ranks 1-5": [("tri", tuple(range(0, 5)), 100.0)],
        "trifecta box ranks 4-9": [("tri", tuple(range(3, 9)), 100.0)],
        "mixed: $40 longshots, $30 place 4-6, $30 tri 2-6":
            [("win", (i,), 4.0) for i in long_ranks]
            + [("place", (i,), 10.0) for i in (3, 4, 5)]
            + [("tri", tuple(range(1, 6)), 30.0)],
    }


def _prices(g: pd.DataFrame, sc: Dict[str, float]):
    pw = ((1 / g["bsp"]) / (1 / g["bsp"]).sum()).to_numpy()
    pp = (3 * (1 / g["place_bsp"]) / (1 / g["place_bsp"]).sum()).to_numpy()
    win_price = 1 / (pw * sc["win_m"])
    place_price = np.maximum(1 / (pp * sc["place_m"]), 1.0)
    return pw, win_price, place_price


def settle(bets: List[Bet], year_df: pd.DataFrame, sc: Dict[str, float]) -> Tuple[float, float]:
    """Return (staked, returned) for one Cup. Trifecta dividends are simulated:
    (1 - takeout) / P(1-2-3) under the market's Plackett-Luce probabilities."""
    g = year_df.sort_values("bsp", kind="stable").reset_index(drop=True)
    pw, win_price, place_price = _prices(g, sc)
    pos = g["finishing_position"].to_numpy(dtype=float)
    first, second, third = np.argsort(pos, kind="stable")[:3]
    staked = returned = 0.0
    for kind, ranks, stake in bets:
        staked += stake
        if kind == "win":
            if pos[ranks[0]] == 1:
                returned += stake * win_price[ranks[0]]
        elif kind == "place":
            if pos[ranks[0]] <= 3:
                returned += stake * place_price[ranks[0]]
        elif kind == "tri":
            k = len(ranks)
            unit = stake / (k * (k - 1) * (k - 2))
            if {first, second, third} <= set(ranks):
                p = pw[first] * pw[second] / (1 - pw[first]) * pw[third] / (1 - pw[first] - pw[second])
                returned += unit * (1 - sc["takeout"]) / p
        else:
            raise ValueError(f"unknown bet kind: {kind}")
    return staked, returned


def backtest(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, sc in SCENARIOS.items():
        for year, g in frame.groupby("year"):
            for name, bets in strategy_bets(len(g)).items():
                staked, returned = settle(bets, g, sc)
                rows.append({"scenario": scenario, "year": int(year), "strategy": name,
                             "staked": staked, "returned": returned})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    out = table.groupby(["scenario", "strategy"], sort=False).agg(
        staked=("staked", "sum"), returned=("returned", "sum"),
        years_paid=("returned", lambda s: int((s > 0).sum())), cups=("year", "count"),
        best_year=("returned", "max"), worst_year=("returned", "min")).reset_index()
    out["roi_pct"] = (out["returned"] / out["staked"] - 1) * 100
    return out.round(1)


def placegetter_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Top-3 finishers with their pre-race market rank (1 = favourite)."""
    f = frame.copy()
    f["market_rank"] = f.groupby("year")["bsp"].rank(method="first").astype(int)
    return f[f["finishing_position"] <= 3].sort_values(["year", "finishing_position"])[
        ["year", "finishing_position", "horse_name", "bsp", "market_rank"]]


def build_report(frame: pd.DataFrame) -> str:
    table = backtest(frame)
    summary = summarise(table)
    years = sorted(frame["year"].unique())
    placed = placegetter_ranks(frame)
    buckets = [(1, 3), (4, 6), (7, 10), (11, 16), (17, 24)]
    bucket_lines = [f"  market ranks {lo}-{hi}: {int(placed['market_rank'].between(lo, hi).sum())}"
                    for lo, hi in buckets]
    winners = sorted(placed[placed["finishing_position"] == 1]["market_rank"].tolist())
    parts = [
        f"Melbourne Cup strategy replay, {len(years)} Cups {years[0]}-{years[-1]}, $100 per Cup per strategy",
        "Prices: race-normalised Betfair BSP with a margin. Trifecta dividends are simulated from the",
        "market's Plackett-Luce probabilities (real tote dividends are not in the data).",
        "Six Cups is a tiny sample: results show what a shape of bet looks like, not an edge.",
        "",
        f"Winners' market ranks (1 = favourite): {winners}",
        "Top-3 finishers by market rank bucket (18 in total):",
        *bucket_lines,
    ]
    for scenario, sc in SCENARIOS.items():
        parts += ["", f"== {scenario} {sc}"]
        parts.append(summary[summary["scenario"] == scenario].drop(columns="scenario").to_string(index=False))
        piv = table[table["scenario"] == scenario].pivot(index="strategy", columns="year", values="returned").round(0)
        parts += ["Return per Cup ($):", piv.to_string()]
    return "\n".join(parts)


def main() -> None:
    report = build_report(load_frame())
    print(report)
    os.makedirs("docs", exist_ok=True)
    with open("docs/strategy_report.md", "w") as fh:
        fh.write("```\n" + report + "\n```\n")


if __name__ == "__main__":
    main()
