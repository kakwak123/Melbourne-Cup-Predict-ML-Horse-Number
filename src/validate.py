"""Checks that a Melbourne Cup results table is a real, self-consistent race."""

import re
from typing import List

import pandas as pd

REQUIRED = ["year", "horse_number", "horse_name", "barrier", "weight", "finishing_position"]
MAX_FIELD = 24


def validate_race(df: pd.DataFrame) -> List[str]:
    """Return a list of problems for one race (empty list means valid)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    problems = []
    if len(df) > MAX_FIELD:
        problems.append(f"more than 24 runners ({len(df)})")
    if (df["finishing_position"] == 1).sum() != 1:
        problems.append("expected exactly one winner (finishing_position == 1)")
    if df["horse_number"].duplicated().any():
        problems.append("duplicate horse_number")
    if (~df["barrier"].between(1, MAX_FIELD)).any():
        problems.append("barrier outside 1-24")
    if (~df["weight"].between(45, 65)).any():
        problems.append("weight outside 45-65kg")

    # Competition ranking: a tie repeats a position and skips the next one (1,2,2,4).
    ranks = sorted(df["finishing_position"].tolist())
    for i, r in enumerate(ranks):
        if i == 0:
            ok = r == 1
        else:
            ok = r == ranks[i - 1] or r == i + 1
        if not ok:
            problems.append("finishing positions are not competition-ranked (1,2,2,4,...)")
            break
    return problems


def _norm_name(name: str) -> str:
    name = re.sub(r"\s*\([A-Za-z]+\)\s*$", "", str(name))
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def compare_to_verified(existing: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """One row per disagreement between an existing table and the verified one."""
    cols = ["year", "horse_number", "field", "existing", "verified"]
    e = existing.set_index("horse_number")
    v = verified.set_index("horse_number")
    rows = []
    year = int(verified["year"].iloc[0])
    for num in sorted(set(e.index) - set(v.index)):
        rows.append((year, num, "only_in_existing", e.loc[num, "horse_name"], None))
    for num in sorted(set(v.index) - set(e.index)):
        rows.append((year, num, "only_in_verified", None, v.loc[num, "horse_name"]))
    for num in sorted(set(e.index) & set(v.index)):
        if _norm_name(e.loc[num, "horse_name"]) != _norm_name(v.loc[num, "horse_name"]):
            rows.append((year, num, "horse_name", e.loc[num, "horse_name"], v.loc[num, "horse_name"]))
        for field in ("barrier", "weight", "finishing_position"):
            if abs(float(e.loc[num, field]) - float(v.loc[num, field])) > 1e-6:
                rows.append((year, num, field, e.loc[num, field], v.loc[num, field]))
    return pd.DataFrame(rows, columns=cols)
