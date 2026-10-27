```
Melbourne Cup evaluation (leave-one-year-out; lower log loss is better)
Uniform winner log loss for a 24-runner field is ln 24 = 3.178.

All verified years (14 races), baseline = uniform
      model  races  win_ll  top3_ll  top3_ll_minus_uniform  ci_lo  ci_hi
    uniform     14   3.144    9.299                  0.000  0.000  0.000
fundamental     14   3.115    9.341                  0.042 -0.266  0.375

Market years [2020, 2021, 2022, 2023, 2024, 2025], baseline = market
      model  races  win_ll  top3_ll  top3_ll_minus_market  ci_lo  ci_hi
    uniform      6   3.135    9.270                -0.025 -1.521  1.538
     market      6   3.365    9.295                 0.000  0.000  0.000
fundamental      6   2.999    9.583                 0.288 -1.255  1.875
   combined      6   3.179    9.675                 0.380 -0.589  1.505

Verdict: combined did not beat the market on mean top-3 log loss (difference +0.380 nats, 95% CI [-0.589, +1.505]); the CI includes 0.
```
