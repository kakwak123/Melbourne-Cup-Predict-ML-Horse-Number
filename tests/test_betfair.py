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
