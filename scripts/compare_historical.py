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
