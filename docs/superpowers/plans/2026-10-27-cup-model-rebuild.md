# Melbourne Cup Model Rebuild (Steps 1-3 + frozen 2026 prediction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the position-regression pipeline with a per-race Plackett-Luce model scored against the Betfair market, using only verified data, and freeze a logged 2026 Melbourne Cup prediction before the race (Tuesday 3 November 2026).

**Architecture:** New small modules in `src/` (validation, market probabilities, Plackett-Luce fit and top-3 enumeration, metrics, within-race features, leave-one-year-out evaluation). Legacy modules (`train_models.py`, `predict.py`, `advanced_models.py`, XGBoost/LSTM code) are left untouched and marked legacy. Every runner is described relative to its own field, and the race is the unit of validation.

**Tech Stack:** Python 3, numpy, pandas, scipy (`L-BFGS-B`), pytest. No TensorFlow, no XGBoost.

**Spec:** `docs/Melbourne Cup predictor alternatives.md` (migration steps 1-3, plus the Betfair data from step 4 restricted to the Cup races themselves so the market baseline exists). Steps 4-7 of that doc are **out of scope** here; they need their own plans (see "Follow-up plans" at the end).

## Global Constraints

- **Unit of analysis is a race.** Validation, standardisation and scoring are always per `year`. Never split rows of one race across train and test. Use leave-one-year-out (LOGO).
- **No test-year leakage.** Anything fitted (coefficients, scalers) is fitted on other years only. Features are standardised _within_ a race, so no scaler is shared across years.
- **Verified data only.** Models read `data/verified/*.csv`. The old `data/historical/*.csv` (partly synthetic) is used for the comparison report only, never for fitting or scoring.
- **Required columns of a verified results file:** `year,horse_number,horse_name,barrier,weight,finishing_position` (`trainer`, `jockey`, `odds` are optional extras). `horse_number` is the saddlecloth number and equals Betfair's `TAB_NUMBER`.
- **Probabilities:** per-race win probabilities sum to 1; top-3 probabilities sum to 3 (within 1e-9).
- **Odds are decimal win odds** (`bsp` = Betfair `WIN_BSP`), must be finite and > 1.
- **Do not tune on 2025 or earlier results to chase winners.** Hyperparameters are fixed in code: `l2=1.0`, depth 3. No grid search in this plan.
- **Licensing:** raw Betfair zips live only in `data/raw/` (gitignored). Only the extracted per-Cup BSP file `data/market/cup_bsp.csv` (24 rows per year) is committed. Keep scraped data out of any public push.
- **Interpreter:** commands assume the repo's virtualenv is active (`python -m venv venv && source venv/bin/activate`); `venv/` is already gitignored.
- **`.gitignore` ignores `*.csv`.** New data directories need explicit `!` exceptions (Task 1).

**User decisions (already made):** "plan it out like what is mentioned here" (the alternatives doc). Earlier in the session: leave-one-year-out backtest first, real per-runner odds are the most valuable data, nothing pushed until asked.

## Review Focus

Failure modes the spec implies but a straight reading would miss, most likely first. Each has a test in the owning task.

1. **Dead heat in a verified race** (2025 had two horses on 16th and no 17th): validation must accept it and the Plackett-Luce fit must not crash. Tests: Task 1 `test_dead_heat_is_valid`, Task 5 `test_fit_handles_dead_heat_outside_top3`.
2. **Betfair TAB number not matching the results' horse number** (reserves, renumbering): the join must raise on unmatched runners, never silently drop them. Test: Task 3 `test_attach_bsp_raises_on_unmatched`.
3. **Missing or invalid odds for one runner** (blank, <= 1): market probabilities must raise a clear error rather than produce NaN predictions. Tests: Task 4 `test_implied_probs_rejects_bad_odds`, Task 7 `test_lineup_with_missing_odds_raises`.
4. **Zero-variance feature in a race** (every runner on the same weight): standardisation must give zeros, not NaN. Test: Task 4 `test_zscore_constant_column_is_zero`.
5. **A year with market data missing for some runners**: the combined model and market baseline must skip that year, not train on NaN. Test: Task 6 `test_incomplete_market_year_is_excluded`.

---

### Task 1: Validation module and test scaffolding

**Goal:** A tested `validate_race` / `compare_to_verified` module, plus pytest, scipy and gitignore exceptions.

**Files:**

- Create: `src/validate.py`
- Create: `tests/__init__.py`, `tests/test_validate.py`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt` (append `scipy>=1.11.0`)
- Modify: `.gitignore` (add `!data/verified/*.csv`, `!data/market/*.csv`, `!predictions/*.csv` after the existing `!data/historical/*.json` line)

**Acceptance Criteria:**

- [ ] `validate_race` returns `[]` for a well-formed race including a dead heat (positions 1,2,3,...,16,16,18,...).
- [ ] `validate_race` reports: missing columns, not exactly one winner, duplicate horse numbers, barrier outside 1-24, weight outside 45-65, more than 24 runners, and positions that are not competition-ranked.
- [ ] `compare_to_verified` returns one row per mismatching field, and one row for each runner present on only one side.
- [ ] `python -m pytest tests/test_validate.py -v` passes.

**Verify:** `python -m pytest tests/test_validate.py -v` → all tests PASS

**Steps:**

- [ ] **Step 1: Install tooling and add requirements**

```bash
printf 'pytest>=8.0.0\n-r requirements.txt\n' > requirements-dev.txt
printf 'scipy>=1.11.0\n' >> requirements.txt
pip install numpy pandas scipy pytest
```

Add to `.gitignore` directly under `!data/historical/*.json`:

```
!data/verified/*.csv
!data/market/*.csv
!predictions/*.csv
```

- [ ] **Step 2: Write the failing tests** in `tests/test_validate.py`

```python
import pandas as pd

from src.validate import compare_to_verified, validate_race


def make_race(n=24):
    return pd.DataFrame({
        "year": 2025,
        "horse_number": range(1, n + 1),
        "horse_name": [f"HORSE {i}" for i in range(1, n + 1)],
        "barrier": range(1, n + 1),
        "weight": [52.0 + (i % 5) for i in range(n)],
        "finishing_position": range(1, n + 1),
    })


def test_clean_race_is_valid():
    assert validate_race(make_race()) == []


def test_dead_heat_is_valid():
    df = make_race()
    df.loc[df.horse_number == 17, "finishing_position"] = 16   # tie for 16th
    df.loc[df.horse_number == 18, "finishing_position"] = 18   # nobody on 17th
    assert validate_race(df) == []


def test_two_winners_is_invalid():
    df = make_race()
    df.loc[1, "finishing_position"] = 1
    assert any("exactly one winner" in p for p in validate_race(df))


def test_skipped_position_is_invalid():
    df = make_race()
    df.loc[df.horse_number == 5, "finishing_position"] = 9   # 1,2,3,4,9,6,7,8,9,...
    assert any("competition" in p for p in validate_race(df))


def test_bad_barrier_weight_and_duplicates():
    df = make_race()
    df.loc[0, "barrier"] = 30
    df.loc[1, "weight"] = 80.0
    df.loc[2, "horse_number"] = 1
    problems = " | ".join(validate_race(df))
    assert "barrier" in problems and "weight" in problems and "duplicate horse_number" in problems


def test_more_than_24_runners_is_invalid():
    assert any("more than 24" in p for p in validate_race(make_race(25)))


def test_missing_columns_reported():
    assert any("missing columns" in p for p in validate_race(make_race().drop(columns=["barrier"])))


def test_compare_flags_field_differences_and_unmatched():
    verified = make_race(3)
    existing = verified.copy()
    existing.loc[0, "barrier"] = 7
    existing.loc[1, "horse_name"] = "SOMETHING ELSE"
    existing = pd.concat([existing, existing.iloc[[2]].assign(horse_number=9)], ignore_index=True)
    diff = compare_to_verified(existing, verified)
    fields = set(diff["field"])
    assert {"barrier", "horse_name", "only_in_existing"} <= fields


def test_compare_ignores_country_suffix_and_case():
    verified = make_race(3)
    existing = verified.copy()
    existing["horse_name"] = existing["horse_name"].str.lower() + " (IRE)"
    assert compare_to_verified(existing, verified).empty
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_validate.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'src.validate'`

- [ ] **Step 4: Write the implementation** `src/validate.py`

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_validate.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/validate.py tests/__init__.py tests/test_validate.py requirements.txt requirements-dev.txt .gitignore
git commit -m "feat: add race validation and verified-data comparison"
```

---

### Task 2: Build the verified results dataset

**Goal:** `data/verified/melbourne_cup_{year}.csv` for every Cup year whose runners, barriers, weights and placings can be checked against Wikipedia, plus a report of what disagreed with the old CSVs.

**Files:**

- Move: `data/backtest/melbourne_cup_2024.csv`, `data/backtest/melbourne_cup_2025.csv` → `data/verified/` (`git mv`; keep columns, drop the sparse `odds` column or leave blank)
- Create: `data/verified/melbourne_cup_{2010..2023}.csv` (only years that can be fully verified)
- Create: `scripts/compare_historical.py`
- Create: `docs/verification_report.md`
- Modify: `backtest.py` (read `data/verified` instead of `data/backtest`)

**Acceptance Criteria:**

- [ ] Every file in `data/verified/` passes `validate_race` with no problems.
- [ ] Every verified file has the columns `year,horse_number,horse_name,barrier,weight,finishing_position` and the field size matches the Wikipedia table for that year (scratched horses excluded, dead heats kept).
- [ ] Years where Wikipedia lacks barrier or weight for any runner are NOT added; they are listed under "Dropped years" in `docs/verification_report.md` with the reason.
- [ ] `docs/verification_report.md` states, per year 2010-2023, how many fields of the old `data/historical` CSV disagreed with the verified data (or "not verified").
- [ ] `python scripts/compare_historical.py` runs cleanly and prints the same per-year counts as the report.

**Verify:** `python scripts/compare_historical.py` → prints one line per verified year (`2024: 0 mismatches`, ...) and `ALL VERIFIED FILES VALID`

**Steps:**

- [ ] **Step 1: Move the two already-verified files**

```bash
mkdir -p data/verified
git mv data/backtest/melbourne_cup_2024.csv data/verified/melbourne_cup_2024.csv
git mv data/backtest/melbourne_cup_2025.csv data/verified/melbourne_cup_2025.csv
sed -i '' 's#data/backtest/\*.csv#data/verified/*.csv#' backtest.py
```

- [ ] **Step 2: Collect each year 2010-2023** from `https://en.wikipedia.org/wiki/{year}_Melbourne_Cup`

For each year, open the page's results/field table and write `data/verified/melbourne_cup_{year}.csv` with this exact header and one row per _finisher_ (exclude scratchings; keep dead heats using competition ranking):

```
year,horse_number,horse_name,barrier,weight,trainer,jockey,finishing_position
```

Name format: uppercase, with the country suffix as printed on Wikipedia, e.g. `WITHOUT A FIGHT (IRE)`. If the page has no barrier or weight column for a year, do not create the file; note the year and reason for the report (Step 4). Do not copy values from the old `data/historical` CSVs, since the point is an independent check.

- [ ] **Step 3: Write the comparison script** `scripts/compare_historical.py`

```python
"""Compare data/historical (old, partly synthetic) with data/verified and validate verified files."""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate import compare_to_verified, validate_race


def main() -> int:
    bad = 0
    for path in sorted(glob.glob("data/verified/melbourne_cup_*.csv")):
        verified = pd.read_csv(path)
        year = int(verified["year"].iloc[0])
        problems = validate_race(verified)
        if problems:
            bad += 1
            print(f"{year}: INVALID {problems}")
            continue
        old_path = f"data/historical/melbourne_cup_{year}.csv"
        if not os.path.exists(old_path):
            print(f"{year}: no old CSV to compare")
            continue
        diff = compare_to_verified(pd.read_csv(old_path), verified)
        print(f"{year}: {len(diff)} mismatches")
    if bad == 0:
        print("ALL VERIFIED FILES VALID")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it and write the report**

Run: `python scripts/compare_historical.py`
Expected: one line per year and `ALL VERIFIED FILES VALID`. Fix any INVALID file by re-checking it against Wikipedia (never by editing values to make the validator pass).

Write `docs/verification_report.md` with: a table `year | verified | mismatches vs old CSV | note`; a "Dropped years" section (year and reason); a one-paragraph conclusion on whether the old CSVs look real or synthetic.

- [ ] **Step 5: Commit**

```bash
git add data/verified scripts/compare_historical.py docs/verification_report.md backtest.py
git commit -m "data: add Wikipedia-verified Cup results and comparison report"
```

---

### Task 3: Betfair Cup odds loader

**Goal:** A tested loader that extracts the Melbourne Cup race (Flemington, 3200m, November) from Betfair's free ANZ thoroughbred CSVs and attaches BSP to the verified results.

**Files:**

- Create: `src/betfair.py`
- Create: `scripts/fetch_cup_bsp.py`
- Create: `tests/test_betfair.py`
- Create: `data/market/cup_bsp.csv` (output, committed)

**Acceptance Criteria:**

- [ ] `load_cup_bsp` returns columns `year,horse_number,horse_name,bsp` with one race per year, and raises if a year has none or more than one candidate race date.
- [ ] `attach_bsp` merges BSP onto verified results by `(year, horse_number)` and raises `ValueError` listing any verified runner with no BSP match.
- [ ] `python scripts/fetch_cup_bsp.py` writes `data/market/cup_bsp.csv` covering every Cup year 2020-2025, 20-24 runners each; the 2025 row for horse 14 (Half Yours) has `bsp` of about 10.18.
- [ ] `python -m pytest tests/test_betfair.py -v` passes.

**Verify:** `python -m pytest tests/test_betfair.py -v && python scripts/fetch_cup_bsp.py` → tests PASS, then prints `2020..2025` with runner counts

**Steps:**

- [ ] **Step 1: Write the failing tests** `tests/test_betfair.py`

```python
import pandas as pd
import pytest

from src.betfair import attach_bsp, load_cup_bsp

COLS = ["LOCAL_MEETING_DATE", "TRACK", "RACE_NO", "WIN_MARKET_NAME", "DISTANCE",
        "TAB_NUMBER", "SELECTION_NAME", "WIN_BSP"]


def row(date, track, dist, num, name, bsp, market="R7 3200m Grp1"):
    return [date, track, 7, market, dist, num, name, bsp]


@pytest.fixture
def betfair_csv(tmp_path):
    rows = [
        row("2025-11-04", "Flemington", 3200.0, 14, "Half Yours", 10.18),
        row("2025-11-04", "Flemington", 3200.0, 1, "Al Riffa", 8.2),
        row("2025-11-04", "Flemington", 1200.0, 3, "Sprinter", 5.0, "R3 1200m"),   # other race
        row("2025-03-01", "Flemington", 3200.0, 2, "March Staying", 4.0),         # not November
        row("2025-11-04", "Randwick", 3200.0, 5, "Wrong Track", 6.0),
    ]
    path = tmp_path / "ANZ_Thoroughbreds_2025_11.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return path


def test_load_cup_bsp_filters_to_the_cup(betfair_csv):
    out = load_cup_bsp([betfair_csv])
    assert list(out.columns) == ["year", "horse_number", "horse_name", "bsp"]
    assert set(out["horse_number"]) == {14, 1}
    assert set(out["year"]) == {2025}


def test_load_cup_bsp_raises_when_two_race_dates(tmp_path):
    rows = [row("2025-11-04", "Flemington", 3200.0, 1, "A", 5.0),
            row("2025-11-11", "Flemington", 3200.0, 1, "B", 5.0)]
    path = tmp_path / "x.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than one"):
        load_cup_bsp([path])


def test_attach_bsp_merges_by_number():
    results = pd.DataFrame({"year": [2025, 2025], "horse_number": [14, 1],
                            "horse_name": ["HALF YOURS", "AL RIFFA"]})
    bsp = pd.DataFrame({"year": [2025, 2025], "horse_number": [14, 1],
                        "horse_name": ["Half Yours", "Al Riffa"], "bsp": [10.18, 8.2]})
    out = attach_bsp(results, bsp)
    assert out.set_index("horse_number")["bsp"].to_dict() == {14: 10.18, 1: 8.2}


def test_attach_bsp_raises_on_unmatched():
    results = pd.DataFrame({"year": [2025, 2025], "horse_number": [14, 99],
                            "horse_name": ["HALF YOURS", "GHOST"]})
    bsp = pd.DataFrame({"year": [2025], "horse_number": [14],
                        "horse_name": ["Half Yours"], "bsp": [10.18]})
    with pytest.raises(ValueError, match="GHOST"):
        attach_bsp(results, bsp)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_betfair.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'src.betfair'`

- [ ] **Step 3: Write the implementation** `src/betfair.py`

```python
"""Extract Melbourne Cup runners and Betfair Starting Price from Betfair's free ANZ CSVs."""

from typing import Iterable

import pandas as pd

USECOLS = ["LOCAL_MEETING_DATE", "TRACK", "WIN_MARKET_NAME", "DISTANCE",
           "TAB_NUMBER", "SELECTION_NAME", "WIN_BSP"]


def load_cup_bsp(paths: Iterable) -> pd.DataFrame:
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
    dates = cup.groupby("year")["LOCAL_MEETING_DATE"].nunique()
    if (dates > 1).any():
        raise ValueError(f"more than one candidate Cup date in years {list(dates[dates > 1].index)}")
    out = cup.rename(columns={"TAB_NUMBER": "horse_number", "SELECTION_NAME": "horse_name",
                              "WIN_BSP": "bsp"})
    out = out[["year", "horse_number", "horse_name", "bsp"]].dropna(subset=["bsp"])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_betfair.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the fetch script** `scripts/fetch_cup_bsp.py`

```python
"""Download Betfair's free yearly ANZ thoroughbred zips and keep only the Melbourne Cup rows."""

import glob
import io
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betfair import load_cup_bsp

BASE = "https://betfair-datascientists.github.io/data/assets"
RAW = "data/raw/betfair"
YEARS = range(2020, 2026)


def fetch_november(year: int) -> str:
    """Download the yearly zip once and extract only the November CSV."""
    os.makedirs(RAW, exist_ok=True)
    target = f"{RAW}/ANZ_Thoroughbreds_{year}_11.csv"
    if os.path.exists(target):
        return target
    with urllib.request.urlopen(f"{BASE}/ANZ_Thoroughbreds_{year}.zip") as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    names = [n for n in archive.namelist() if n.endswith(f"{year}_11.csv")]
    if len(names) != 1:
        raise RuntimeError(f"{year}: expected one November file, found {names}")
    with open(target, "wb") as fh:
        fh.write(archive.read(names[0]))
    return target


def main() -> None:
    paths = [fetch_november(y) for y in YEARS]
    cup = load_cup_bsp(paths)
    os.makedirs("data/market", exist_ok=True)
    cup.to_csv("data/market/cup_bsp.csv", index=False)
    print(cup.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it and sanity-check the output**

Run: `python scripts/fetch_cup_bsp.py`
Expected: years 2020-2025 each with 20-24 runners. Then:

```bash
python - <<'E'
import pandas as pd
b = pd.read_csv("data/market/cup_bsp.csv")
print(b[(b.year == 2025) & (b.horse_number == 14)])
E
```

Expected: one row, `bsp` ≈ 10.18. If the 2020 zip has a different layout (columns or file names), fix `fetch_november`/`USECOLS` for that year and note it in the commit message; do not hand-edit the CSV.

- [ ] **Step 7: Commit**

```bash
git add src/betfair.py scripts/fetch_cup_bsp.py tests/test_betfair.py
git add -f data/market/cup_bsp.csv
git commit -m "feat: load Melbourne Cup BSP from Betfair ANZ files"
```

---

### Task 4: Market probabilities, within-race features and metrics

**Goal:** Pure, tested helpers: normalised market probabilities, within-race z-scores, per-race log-loss scoring and a race-level paired bootstrap.

**Files:**

- Create: `src/market.py`, `src/features.py`, `src/metrics.py`
- Create: `tests/test_market_features_metrics.py`

**Acceptance Criteria:**

- [ ] `implied_probs([2,4,4,8])` sums to 1 and equals `[4/9, 2/9, 2/9, 1/9]`; odds <= 1, NaN or inf raise `ValueError`.
- [ ] `add_race_features` adds `w_z`, `b_z` per year (mean 0), a constant-weight race gives all-zero `w_z`, and `log_mkt` is only set for years where every runner has `bsp`.
- [ ] `winner_log_loss` of a uniform 24-runner field equals `ln 24` (about 3.178).
- [ ] `top3_log_loss` is `-(log p1 + log(p2/(1-p1)) + log(p3/(1-p1-p2)))` for the actual 1-2-3.
- [ ] `paired_bootstrap` returns (mean, lo, hi) with `lo <= mean <= hi`, deterministic for a fixed seed.

**Verify:** `python -m pytest tests/test_market_features_metrics.py -v` → all PASS

**Steps:**

- [ ] **Step 1: Write the failing tests** `tests/test_market_features_metrics.py`

```python
import numpy as np
import pandas as pd
import pytest

from src.features import add_race_features
from src.market import implied_probs
from src.metrics import paired_bootstrap, top3_log_loss, winner_log_loss


def test_implied_probs_removes_overround():
    p = implied_probs([2, 4, 4, 8])
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx([4 / 9, 2 / 9, 2 / 9, 1 / 9])


@pytest.mark.parametrize("bad", [[2, 1.0], [2, 0.5], [2, float("nan")], [2, float("inf")]])
def test_implied_probs_rejects_bad_odds(bad):
    with pytest.raises(ValueError):
        implied_probs(bad)


def test_zscore_constant_column_is_zero():
    df = pd.DataFrame({"year": [2025] * 3, "weight": [54.0] * 3, "barrier": [1, 2, 3]})
    out = add_race_features(df)
    assert (out["w_z"] == 0).all()
    assert out["b_z"].mean() == pytest.approx(0.0)


def test_log_mkt_only_for_complete_years():
    df = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "weight": [50, 52, 50, 52], "barrier": [1, 2, 1, 2],
        "bsp": [2.0, 4.0, 3.0, np.nan],
    })
    out = add_race_features(df)
    assert out.loc[out.year == 2024, "log_mkt"].notna().all()
    assert out.loc[out.year == 2025, "log_mkt"].isna().all()


def test_winner_log_loss_uniform_is_ln_24():
    p = np.full(24, 1 / 24)
    positions = np.arange(1, 25)
    assert winner_log_loss(p, positions) == pytest.approx(np.log(24))


def test_top3_log_loss_matches_formula():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    positions = np.array([2, 1, 3, 4])        # winner idx1, second idx0, third idx2
    expected = -(np.log(0.3) + np.log(0.5 / 0.7) + np.log(0.15 / 0.2))
    assert top3_log_loss(p, positions) == pytest.approx(expected)


def test_paired_bootstrap_is_deterministic_and_ordered():
    d = [0.1, -0.2, 0.05, 0.3, -0.1]
    a = paired_bootstrap(d, n_boot=2000, seed=1)
    assert a == paired_bootstrap(d, n_boot=2000, seed=1)
    mean, lo, hi = a
    assert lo <= mean <= hi
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_market_features_metrics.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'src.features'`

- [ ] **Step 3: Write `src/market.py`**

```python
"""Convert betting odds into race-normalised probabilities."""

import numpy as np


def implied_probs(odds) -> np.ndarray:
    """Win probabilities from decimal odds, with the overround removed."""
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("odds must be finite decimal odds greater than 1")
    raw = 1.0 / odds
    return raw / raw.sum()
```

- [ ] **Step 4: Write `src/features.py`**

```python
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
```

- [ ] **Step 5: Write `src/metrics.py`**

```python
"""Per-race scoring rules and a race-level paired bootstrap."""

from typing import Sequence, Tuple

import numpy as np


def winner_log_loss(p: np.ndarray, positions: np.ndarray) -> float:
    """-log P(actual winner). Uniform over 24 runners is ln 24, about 3.18."""
    return float(-np.log(p[int(np.argmin(positions))]))


def top3_log_loss(p: np.ndarray, positions: np.ndarray) -> float:
    """-log PL likelihood of the actual 1-2-3 finishers, in order."""
    first, second, third = np.argsort(positions, kind="stable")[:3]
    ll = np.log(p[first])
    ll += np.log(p[second] / (1.0 - p[first]))
    ll += np.log(p[third] / (1.0 - p[first] - p[second]))
    return float(-ll)


def paired_bootstrap(diffs: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean and 95% CI of per-race differences, resampling races (never rows)."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_market_features_metrics.py -v`
Expected: 10 passed

- [ ] **Step 7: Commit**

```bash
git add src/market.py src/features.py src/metrics.py tests/test_market_features_metrics.py
git commit -m "feat: add market probabilities, race features and scoring metrics"
```

---

### Task 5: Plackett-Luce top-3 model

**Goal:** A regularised Plackett-Luce (exploded conditional logit) fitted on the first three finishers of each race, with exact top-3 probabilities.

**Files:**

- Create: `src/plackett_luce.py`
- Create: `tests/test_plackett_luce.py`

**Acceptance Criteria:**

- [ ] On 300 synthetic races with known coefficients `[1.5, -0.5]`, `fit` recovers them to within 0.15 each.
- [ ] `top3_probs(softmax(scores))` sums to 3 (tolerance 1e-9) for a 5-runner and a 24-runner field, and each value is between its win probability and 1.
- [ ] A 3-runner field gives top-3 probability 1 for every runner.
- [ ] `fit` with zero features returns an empty array; a race where two horses tie outside the top 3 does not raise.

**Verify:** `python -m pytest tests/test_plackett_luce.py -v` → all PASS

**Steps:**

- [ ] **Step 1: Write the failing tests** `tests/test_plackett_luce.py`

```python
import numpy as np
import pytest

from src.plackett_luce import fit, softmax, top3_probs


def simulate(n_races=300, n=12, true=(1.5, -0.5), seed=1):
    rng = np.random.default_rng(seed)
    true = np.array(true)
    races = []
    for _ in range(n_races):
        X = rng.normal(size=(n, len(true)))
        gumbel = -np.log(-np.log(rng.random(n)))
        positions = np.argsort(np.argsort(-(X @ true + gumbel))) + 1
        races.append((X, positions.astype(float)))
    return races


def test_fit_recovers_known_coefficients():
    beta = fit(simulate(), l2=0.1)
    assert beta == pytest.approx([1.5, -0.5], abs=0.15)


@pytest.mark.parametrize("n", [5, 24])
def test_top3_probs_sum_to_three_and_dominate_win_probs(n):
    p = softmax(np.linspace(0, 3, n))
    t = top3_probs(p)
    assert t.sum() == pytest.approx(3.0, abs=1e-9)
    assert (t >= p - 1e-12).all() and (t <= 1.0 + 1e-12).all()


def test_three_runner_field_everyone_places():
    assert top3_probs(np.array([0.5, 0.3, 0.2])) == pytest.approx([1, 1, 1])


def test_fit_with_no_features_returns_empty():
    races = [(np.zeros((6, 0)), np.arange(1.0, 7.0))]
    assert fit(races).shape == (0,)


def test_fit_handles_dead_heat_outside_top3():
    X = np.random.default_rng(0).normal(size=(8, 2))
    positions = np.array([1, 2, 3, 4, 5, 6, 6, 8], dtype=float)
    beta = fit([(X, positions)] * 5)
    assert np.isfinite(beta).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_plackett_luce.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'src.plackett_luce'`

- [ ] **Step 3: Write the implementation** `src/plackett_luce.py`

```python
"""Plackett-Luce (exploded conditional logit) truncated at third place."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

DEPTH = 3


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def race_nll(beta: np.ndarray, X: np.ndarray, positions: np.ndarray, depth: int = DEPTH) -> float:
    """Negative log-likelihood of the first `depth` finishers in one race."""
    scores = X @ beta
    order = np.argsort(positions, kind="stable")[:depth]
    remaining = np.ones(len(scores), dtype=bool)
    nll = 0.0
    for idx in order:
        s = scores[remaining]
        m = s.max()
        log_z = m + np.log(np.exp(s - m).sum())
        nll -= scores[idx] - log_z
        remaining[idx] = False
    return float(nll)


def fit(races: List[Tuple[np.ndarray, np.ndarray]], l2: float = 1.0) -> np.ndarray:
    """Fit coefficients on (X, positions) races with an L2 penalty."""
    n_features = races[0][0].shape[1]
    if n_features == 0:
        return np.zeros(0)

    def objective(beta):
        return sum(race_nll(beta, X, pos) for X, pos in races) + 0.5 * l2 * float(beta @ beta)

    return minimize(objective, np.zeros(n_features), method="L-BFGS-B").x


def top3_probs(p: np.ndarray) -> np.ndarray:
    """P(horse finishes in the top three) by exact enumeration of the PL model."""
    n = len(p)
    out = p.copy()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            out[i] += p[j] * p[i] / (1.0 - p[j])
            for k in range(n):
                if k == i or k == j:
                    continue
                out[i] += p[j] * p[k] / (1.0 - p[j]) * p[i] / max(1.0 - p[j] - p[k], 1e-12)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_plackett_luce.py -v`
Expected: 6 passed (the parametrised test counts twice)

- [ ] **Step 5: Commit**

```bash
git add src/plackett_luce.py tests/test_plackett_luce.py
git commit -m "feat: add truncated Plackett-Luce model with exact top-3 probabilities"
```

---

### Task 6: Leave-one-year-out evaluation vs the market

**Goal:** One command that scores uniform, market-only, fundamental (weight + barrier) and combined models per Cup year and reports each against the market with a race-level bootstrap.

**Files:**

- Create: `src/evaluate.py`
- Create: `tests/test_evaluate.py`
- Delete: `backtest.py` (superseded; keeps one source of truth)
- Create: `docs/evaluation_report.md` (generated output, committed)

**Acceptance Criteria:**

- [ ] `logo_pl` trains each year's model without that year (a test proves the held-out race is absent from training).
- [ ] A year where any runner lacks `bsp` is excluded from the market, combined and market-years tables (`test_incomplete_market_year_is_excluded`).
- [ ] `python -m src.evaluate` prints two tables ("All verified years" with uniform + fundamental; "Market years" with uniform, market, fundamental, combined) and writes the same text to `docs/evaluation_report.md`.
- [ ] The report states, in words, whether `combined` beat `market` on mean top-3 log loss and whether the bootstrap CI excludes 0.

**Verify:** `python -m pytest tests/test_evaluate.py -v && python -m src.evaluate` → tests PASS; two tables printed; `docs/evaluation_report.md` written

**Steps:**

- [ ] **Step 1: Write the failing tests** `tests/test_evaluate.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: Write the implementation** `src/evaluate.py`

````python
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
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the evaluation on the real verified data**

Run: `python -m src.evaluate`
Expected: two tables and a "Verdict:" line; `docs/evaluation_report.md` created. If it raises `no BSP match for ...`, a verified `horse_number` disagrees with Betfair's TAB number for that runner: fix the verified file by re-checking Wikipedia, not by dropping the runner.

- [ ] **Step 6: Remove the superseded script and commit**

```bash
git rm backtest.py
git add src/evaluate.py tests/test_evaluate.py docs/evaluation_report.md
git commit -m "feat: add leave-one-year-out evaluation against the Betfair market"
```

---

### Task 7: Freeze the model and log the 2026 prediction

**Goal:** A deterministic `predict_cup` that turns the post-draw 2026 lineup into win and top-3 probabilities, using a rule fixed in code for choosing between the market-only and combined model, committed and tagged before the race.

**Files:**

- Create: `src/predict_cup.py`
- Create: `tests/test_predict_cup.py`
- Create: `predictions/2026_melbourne_cup.csv` (output, committed with `git add -f`)
- Input: `data/processed/2026_lineup_provisional.csv` (local, gitignored; update with barriers, final weights, jockeys and live decimal win odds after the barrier draw)

**Acceptance Criteria:**

- [ ] `predict_lineup` returns one row per runner with `horse_number,horse_name,p_win,p_top3,model`; `p_win` sums to 1 and `p_top3` sums to 3 (1e-9).
- [ ] Rule: use `combined` only if its mean top-3 log loss is lower than the market's in `build_report`'s market-year LOGO; otherwise `market`. The chosen `model` is recorded in every output row.
- [ ] A lineup with a missing or <= 1 odds value for any runner raises `ValueError` naming the horse.
- [ ] After the barrier draw, the real file has `horse_number`, `barrier`, `weight`, `odds` for all runners (no blanks) and `predictions/2026_melbourne_cup.csv` is committed and tagged `freeze-2026-cup` before the race starts (Tue 3 Nov 2026, 3:00pm AEDT).

**Verify:** `python -m pytest tests/test_predict_cup.py -v && python -m src.predict_cup data/processed/2026_lineup_provisional.csv` → tests PASS; prints the ranked table and writes `predictions/2026_melbourne_cup.csv`

**Steps:**

- [ ] **Step 1: Write the failing tests** `tests/test_predict_cup.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predict_cup.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'src.predict_cup'`

- [ ] **Step 3: Write the implementation** `src/predict_cup.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predict_cup.py -v`
Expected: 4 passed

- [ ] **Step 5: Update the lineup after the barrier draw (Sat 31 Oct)**

Edit `data/processed/2026_lineup_provisional.csv`: fill `horse_number` (saddlecloth), `barrier`, final `weight`, jockeys, and set `odds` to live Ladbrokes/TAB decimal win odds (or Betfair price) for every runner; remove scratched horses. Leave `age`, `wins`, `places`, `recent_race_times` blank (unused by this model). Run the test of readiness:

```bash
python -c "import pandas as pd; d=pd.read_csv('data/processed/2026_lineup_provisional.csv'); print(d[['horse_number','barrier','weight','odds']].isna().sum())"
```

Expected: all four counts 0. If the draw is not out yet, stop here and resume after it.

- [ ] **Step 6: Generate, commit and tag the frozen prediction**

Run: `python -m src.predict_cup data/processed/2026_lineup_provisional.csv`
Expected: a ranked table of every runner with `p_win`, `p_top3`, `model`.

```bash
git add src/predict_cup.py tests/test_predict_cup.py
git add -f predictions/2026_melbourne_cup.csv
git commit -m "feat: freeze 2026 Melbourne Cup prediction"
git tag freeze-2026-cup
```

The tag must exist before the 3:00pm AEDT start on Tuesday 3 November. After the race, add `data/verified/melbourne_cup_2026.csv` and append 2026 BSP to `data/market/cup_bsp.csv`, re-run `python -m src.evaluate`, and compare the frozen prediction with the result. Never edit the frozen file.

---

### Task 8: Mark the legacy pipeline and document the new one

**Goal:** README states what is current, what is legacy and why, with the commands to reproduce the evaluation.

**Files:**

- Modify: `README.md` (replace the intro "Features" list and the "Model Details" section; keep installation)
- Modify: `src/train_models.py`, `src/predict.py`, `src/advanced_models.py`, `src/predict_advanced.py`, `src/cli.py` (add a one-line module-docstring note only; no logic changes)

**Acceptance Criteria:**

- [ ] README's first section lists the current pipeline: verified data, Plackett-Luce, Betfair market baseline, leave-one-year-out evaluation, frozen predictions; and the exact commands `python -m pytest`, `python scripts/fetch_cup_bsp.py`, `python -m src.evaluate`, `python -m src.predict_cup <lineup.csv>`.
- [ ] README no longer claims an XGBoost/LSTM ensemble or "0.6/0.4" weights as the current model; a "Legacy" section names the old modules and says they were trained partly on synthetic data and tuned on 2025 results.
- [ ] Each legacy module's docstring opens with `"""LEGACY:`; `python -m pytest` still passes.

**Verify:** `grep -l '^"""LEGACY:' src/*.py | wc -l` → `5`, and `python -m pytest -q` → all tests pass

**Steps:**

- [ ] **Step 1: Replace the README intro** (top of file through the end of "Features") with:

```markdown
# Melbourne Cup Top-3 Prediction

Predicts each runner's chance of finishing in the top three of the Melbourne Cup.

## Current pipeline

- **Data:** `data/verified/` holds Wikipedia-checked results (barrier, weight, finishing order). `data/market/cup_bsp.csv` holds Betfair Starting Prices for the Cups since 2020. The old `data/historical/` CSVs are kept only for the comparison in `docs/verification_report.md`.
- **Model:** a regularised Plackett-Luce (conditional logit) fitted on the first three finishers of each race, with exact top-3 probabilities.
- **Benchmark:** the Betfair market's normalised implied probabilities.
- **Evaluation:** leave-one-year-out, per-race log loss, race-level bootstrap. See `docs/evaluation_report.md`. With this few races the honest claim is "competitive with the market", not "beats the market".
- **Frozen predictions:** `predictions/` holds predictions committed and tagged before a race.

### Commands

    python -m pytest                              # all tests
    python scripts/fetch_cup_bsp.py               # download Cup BSP from Betfair
    python scripts/compare_historical.py          # check verified files, compare with old CSVs
    python -m src.evaluate                        # leave-one-year-out report
    python -m src.predict_cup data/processed/2026_lineup_provisional.csv

## Legacy (do not use for decisions)

`src/train_models.py`, `src/predict.py`, `src/advanced_models.py`, `src/predict_advanced.py` and `src/cli.py` are the original XGBoost / LSTM / logistic pipeline. They were trained on partly synthetic data, scored finishing position instead of each race's field, and were tuned toward the 2025 result. The 2025 top-3 hit rate was 0 of 3. They are kept for reference only.
```

Also delete the README's "Model Details" section (XGBoost/LSTM/ensemble weights) since the "Legacy" paragraph replaces it.

- [ ] **Step 2: Add the LEGACY note** as the first line of each legacy module's docstring, e.g. in `src/train_models.py`:

```python
"""LEGACY: superseded by src/plackett_luce.py + src/evaluate.py (see README).
```

(Keep the rest of each existing docstring after that line. If a file has no docstring, add a one-line `"""LEGACY: ..."""`.)

- [ ] **Step 3: Verify and commit**

Run: `grep -l "^\"\"\"LEGACY:" src/*.py | wc -l` → `5` (this matches the literal `"""LEGACY:` opening line), then `python -m pytest -q` → all pass.

```bash
git add README.md src/train_models.py src/predict.py src/advanced_models.py src/predict_advanced.py src/cli.py
git commit -m "docs: mark legacy pipeline and document the new one"
```

---

## Follow-up plans (out of scope here, from the alternatives doc)

- **Step 4:** train on all ANZ races of 2000m+ from the Betfair monthly/yearly CSVs; hold the Cups out as the test set. Needs its own plan (data volume, form features, joins).
- **Step 5:** Benter two-stage (fundamental model out-of-fold, then a 2-parameter logit on model and market), replacing this plan's joint regularised logit.
- **Step 6:** OpenSkill ratings, a ranker score as a feature, UK/IRE BSP signals for internationals.
- **Step 7:** Bayesian hierarchical Plackett-Luce with pooled jockey and trainer effects.

## Self-review notes

- **Spec coverage:** step 1 → Tasks 1-2; step 2 → Tasks 4 and 6 (LOGO, log loss, normalised market baseline, race-level bootstrap; MLflow deliberately skipped — results go to `docs/evaluation_report.md`); step 3 → Tasks 5 and 7 (the doc's "keep XGBoost + out-of-fold" is replaced by a Plackett-Luce model because the current code has no working XGBoost/LSTM path to reuse and PL gives the top-3 probabilities directly, as the doc itself recommends in its table); "freeze before 3 November" → Task 7; LSTM and the 0.6/0.4 blend → retired in Task 8; Pandera replaced by plain pandas checks in `src/validate.py` (YAGNI).
- **Known deviation:** the combined model uses a joint regularised logit on `[w_z, b_z, log_mkt]`, not Benter's two-stage. Two-stage is Follow-up Step 5.
- **Expected outcome:** with 16 verified races (at most 6 with market data) the evaluation cannot show an edge over the market; Task 6's report states this in its verdict line.
