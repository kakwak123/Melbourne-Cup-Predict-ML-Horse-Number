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
    cup = load_cup_bsp(paths, expected_years=YEARS)
    os.makedirs("data/market", exist_ok=True)
    cup.to_csv("data/market/cup_bsp.csv", index=False)
    print(cup.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
