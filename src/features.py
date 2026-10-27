"""Within-race features. Every runner is described relative to its own field."""

import numpy as np
import pandas as pd

from src.market import implied_probs


def _z(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add w_z, b_z (and log_mkt when a `bsp` column exists) per year."""
    out = df.copy()
    out["w_z"] = out.groupby("year")["weight"].transform(_z)
    out["b_z"] = out.groupby("year")["barrier"].transform(_z)
    if "bsp" in out.columns:
        def log_mkt(g):
            if g["bsp"].isna().any():
                return pd.Series(np.nan, index=g.index)
            return pd.Series(np.log(implied_probs(g["bsp"].values)), index=g.index)
        out["log_mkt"] = out.groupby("year", group_keys=False).apply(log_mkt)
    return out
