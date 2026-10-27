# Verification report: Melbourne Cup results

Verified files in `data/verified/` were built from the field table on each year's Wikipedia page
(`https://en.wikipedia.org/wiki/{year}_Melbourne_Cup`, wikitext). No value was copied from
`data/historical`. Scratched horses are excluded. Runners without a numeric placing on Wikipedia
(DNF, FF) are given the last position (equal to the number of runners), since each year has exactly one
and the numbered placings fill 1..n-1. Reproduce the counts with `python scripts/compare_historical.py`.

| year | verified | mismatches vs old CSV | note |
|------|----------|-----------------------|------|
| 2010 | yes | 23 | 22 barriers differ; 1 runner missing from old CSV |
| 2011 | yes | 10 | barriers differ |
| 2012 | yes | 1 | one weight (53.5 vs 54.0) |
| 2013 | yes | 1 | one runner (Verema) missing from old CSV |
| 2014 | yes | 20 | barriers differ |
| 2015 | yes | 3 | Red Cadeaux missing from old CSV; two name spellings |
| 2016 | not verified | not verified | dropped, see below |
| 2017 | yes | 4 | four barriers differ |
| 2018 | not verified | not verified | dropped, see below |
| 2019 | yes | 1 | one name differs (Il Paradiso) |
| 2020 | yes | 7 | 6 barriers differ; 1 runner missing from old CSV |
| 2021 | yes | 9 | finishing positions differ |
| 2022 | yes | 12 | 11 barriers differ; 1 runner missing from old CSV |
| 2023 | yes | 85 | barriers, names, weights and positions broadly differ; 1 runner only in old CSV |
| 2024 | yes | no old CSV | from `data/backtest` |
| 2025 | yes | no old CSV | from `data/backtest` |

## Dropped years

- 2016: Wikipedia's results table has no barrier column.
- 2018: Wikipedia's field table has no weight column.

## Conclusion

The old CSVs are mostly close to the real fields for 2012, 2013, 2015, 2017 and 2019 (a handful of
disagreements each), which suggests they were derived from real data with errors, mainly in barriers and
missing runners. 2010, 2011, 2014, 2020, 2021 and 2022 have systematic barrier or position disagreements, and
2023 disagrees on almost every field (85 mismatches), so it looks synthetic or badly corrupted. The old
CSVs should not be trusted for training or evaluation; use `data/verified` instead.
