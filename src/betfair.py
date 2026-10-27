"""Extract Melbourne Cup runners and Betfair Starting Price from Betfair's free ANZ CSVs."""

from typing import Iterable, Optional

import pandas as pd

USECOLS = ["LOCAL_MEETING_DATE", "TRACK", "WIN_MARKET_NAME", "DISTANCE",
           "TAB_NUMBER", "SELECTION_NAME", "WIN_BSP", "PLACE_BSP"]


def load_cup_bsp(paths: Iterable, expected_years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Flemington, 3200m, November, Group 1 -> one row per runner per year."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=USECOLS, parse_dates=["LOCAL_MEETING_DATE"])
        mask = (
            (df["TRACK"] == "Flemington")
            & (df["DISTANCE"] == 3200)
            & (df["LOCAL_MEETING_DATE"].dt.month == 11)
            & df["WIN_MARKET_NAME"].str.contains("Grp1", na=False)
        )
        frames.append(df[mask])
    cup = pd.concat(frames, ignore_index=True)
    cup["year"] = cup["LOCAL_MEETING_DATE"].dt.year
    if expected_years is not None:
        missing = sorted(set(expected_years) - set(cup["year"]))
        if missing:
            raise ValueError(f"no Cup rows found for years {missing}")
    dates = cup.groupby("year")["LOCAL_MEETING_DATE"].nunique()
    if (dates > 1).any():
        raise ValueError(f"more than one candidate Cup date in years {list(dates[dates > 1].index)}")
    out = cup.rename(columns={"TAB_NUMBER": "horse_number", "SELECTION_NAME": "horse_name",
                              "WIN_BSP": "bsp", "PLACE_BSP": "place_bsp"})
    out = out[["year", "horse_number", "horse_name", "bsp", "place_bsp"]].dropna(subset=["bsp"])
    out["horse_number"] = out["horse_number"].astype(int)
    return out.sort_values(["year", "horse_number"]).reset_index(drop=True)


def attach_bsp(results: pd.DataFrame, bsp: pd.DataFrame) -> pd.DataFrame:
    """Add a `bsp` column to verified results; raise if any runner has no match."""
    merged = results.merge(bsp[["year", "horse_number", "bsp"]], on=["year", "horse_number"], how="left")
    unmatched = merged[merged["bsp"].isna()]
    if not unmatched.empty:
        names = ", ".join(f"{r.year} #{r.horse_number} {r.horse_name}" for r in unmatched.itertuples())
        raise ValueError(f"no BSP match for: {names}")
    return merged
