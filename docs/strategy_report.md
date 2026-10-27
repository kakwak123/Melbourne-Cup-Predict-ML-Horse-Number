```
Melbourne Cup strategy replay, 6 Cups 2020-2025, $100 per Cup per strategy
Prices: race-normalised Betfair BSP with a margin. Trifecta dividends are simulated from the
market's Plackett-Luce probabilities (real tote dividends are not in the data).
Six Cups is a tiny sample: results show what a shape of bet looks like, not an edge.

Winners' market ranks (1 = favourite): [4, 4, 7, 9, 11, 23]
Top-3 finishers by market rank bucket (18 in total):
  market ranks 1-3: 5
  market ranks 4-6: 2
  market ranks 7-10: 4
  market ranks 11-16: 4
  market ranks 17-24: 3

== exchange {'win_m': 1.02, 'place_m': 1.02, 'takeout': 0.1}
                                        strategy  staked  returned  years_paid  cups  best_year  worst_year  roi_pct
                              favourite win $100   600.0       0.0           0     6        0.0         0.0   -100.0
                            favourite place $100   600.0     191.8           1     6      191.8         0.0    -68.0
                longshot spread $10 x ranks 8-17   600.0     544.4           2     6      298.9         0.0     -9.3
                          trifecta box ranks 1-5   600.0       0.0           0     6        0.0         0.0   -100.0
                          trifecta box ranks 4-9   600.0       0.0           0     6        0.0         0.0   -100.0
mixed: $40 longshots, $30 place 4-6, $30 tri 2-6   600.0     292.5           4     6      119.6         0.0    -51.2
Return per Cup ($):
year                                               2020   2021   2022  2023  2024  2025
strategy                                                                               
favourite place $100                                0.0  192.0    0.0   0.0   0.0   0.0
favourite win $100                                  0.0    0.0    0.0   0.0   0.0   0.0
longshot spread $10 x ranks 8-17                  299.0    0.0  246.0   0.0   0.0   0.0
mixed: $40 longshots, $30 place 4-6, $30 tri 2-6  120.0    0.0   98.0  37.0   0.0  38.0
trifecta box ranks 1-5                              0.0    0.0    0.0   0.0   0.0   0.0
trifecta box ranks 4-9                              0.0    0.0    0.0   0.0   0.0   0.0

== bookmaker {'win_m': 1.15, 'place_m': 1.25, 'takeout': 0.25}
                                        strategy  staked  returned  years_paid  cups  best_year  worst_year  roi_pct
                              favourite win $100   600.0       0.0           0     6        0.0         0.0   -100.0
                            favourite place $100   600.0     156.5           1     6      156.5         0.0    -73.9
                longshot spread $10 x ranks 8-17   600.0     482.9           2     6      265.1         0.0    -19.5
                          trifecta box ranks 1-5   600.0       0.0           0     6        0.0         0.0   -100.0
                          trifecta box ranks 4-9   600.0       0.0           0     6        0.0         0.0   -100.0
mixed: $40 longshots, $30 place 4-6, $30 tri 2-6   600.0     254.2           4     6      106.0         0.0    -57.6
Return per Cup ($):
year                                               2020   2021   2022  2023  2024  2025
strategy                                                                               
favourite place $100                                0.0  156.0    0.0   0.0   0.0   0.0
favourite win $100                                  0.0    0.0    0.0   0.0   0.0   0.0
longshot spread $10 x ranks 8-17                  265.0    0.0  218.0   0.0   0.0   0.0
mixed: $40 longshots, $30 place 4-6, $30 tri 2-6  106.0    0.0   87.0  30.0   0.0  31.0
trifecta box ranks 1-5                              0.0    0.0    0.0   0.0   0.0   0.0
trifecta box ranks 4-9                              0.0    0.0    0.0   0.0   0.0   0.0
```
