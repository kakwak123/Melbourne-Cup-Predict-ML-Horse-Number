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
