# Fix the likelihood, then grow the data

The biggest gain won't come from swapping XGBoost for another library. It comes from two structural changes. First, treat each race as a choice among its runners and train with a within-race softmax likelihood (a conditional logit, or a Plackett-Luce model truncated at third place) instead of regressing finishing position and applying a softmax afterwards. Second, train on far more races than 14 Cups. Benter needed **500 to 1,000 races** just to develop and test a linear logit model, so 14 races is roughly 1.5 to 3% of that floor ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)). Betfair publishes **free per-runner price and result CSVs for every ANZ thoroughbred race since 2020**, so a training set of thousands of staying races costs nothing ([Betfair Data Listing](https://betfair-datascientists.github.io/data/dataListing/)). The Keras LSTM and the hand-set 0.6/0.4 blend should go. Tuning toward the 2025 winners is selection on the test set and should stop. The honest benchmark is the market's normalised implied probabilities. With 14 races you can't show you beat it. You can only show you're competitive with it.

## Fourteen Cups are a sample of fourteen, not 330

Start with the basic structure. Runners in a race are not independent rows. Exactly one horse wins, and each horse's chance depends on who else is in the field. A model that scores each row on its own never sees its competitors. The standard answer is the conditional (multinomial) logit. Each horse gets a linear score and its win probability is the softmax of scores within that race. Bolton and Chapman built this on 200 races in 1986 and Benter extended it in 1994. Probabilities sum to 1 by construction, and calibration is part of the training objective rather than a step bolted on later ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)).

Seen this way, the current pipeline has four problems. MSE on position treats 20th versus 21st as the same error as 1st versus 2nd, even though the first pair is noise and irrelevant to a top-3 task. The softmax temperature is never fitted, so the probabilities can be arbitrarily over- or under-confident. Summing the top three entries of a win-like distribution is not the probability of finishing top 3. That needs a Harville, Henery or Plackett-Luce calculation. And the LSTM has thousands of weights against 14 labelled races. For scale, a Conv1D network on more than 6,000 Hong Kong races reached about 92% training accuracy with "relatively poor" validation, even with dropout ([Towards Data Science](https://towardsdatascience.com/predicting-horse-racing-results-with-deep-learning-7942846287bf/)).

The GridSearch tuned toward 2025 is a separate issue. The 2025 result was Half Yours, Goodie Two Shoes, then Middle Earth ([Wikipedia](https://en.wikipedia.org/wiki/2025_Melbourne_Cup)). Picking features or weights because they would have found those horses makes 2025 a training race. Cawley and Talbot showed that overfitting during model selection often costs as much as the gap between learning algorithms ([JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). With 14 groups that risk is extreme. The 2026 Cup runs on 3 November. Freezing the pipeline before then gives you one genuinely out-of-sample race.

## Free Betfair files turn 14 races into thousands

More data will likely help more than any model change. Benter saw out-of-sample accuracy keep improving up to about 1,000 races ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)). Betfair's own thoroughbred model works the same way. It is one all-races model with a "Staying >2041 m" segment, not a Cup-only model ([Betfair hub tutorial](https://raw.githubusercontent.com/betfair-datascientists/betfair-datascientists.github.io/master/src/tutorials/hubPredictionsModel.md)). Train on every ANZ race of 2000 m or more, grouped by race, and hold out the Cups for testing.

| Source | What you get | Cost | Verdict |
|---|---|---|---|
| Betfair ANZ thoroughbred CSVs ([listing](https://betfair-datascientists.github.io/data/dataListing/)) | Per-runner matched prices, BSP, win and place result, overround at the off. 2020 to Sep 2026 | Free, no login | Adopt first |
| Betfair daily BSP files ([prices](https://promo.betfair.com/betfairsp/prices)) | BSP, morning WAP, volumes for AUS, UK, IRE, FR and more | Free | Adopt. Gives a market signal for internationals' European lead-ups |
| Betfair Kash model results ([listing](https://betfair-datascientists.github.io/data/dataListing/)) | Rated prices plus results, 2021 to 2026 | Free | Use as a benchmark |
| Punting Form ([tiers](https://docs.puntingform.com.au/docs/subscription-options)) | Form, ratings, jockey and trainer data from A$59/month. Sectionals and 10+ years of history in the Modeller tier ([Modeller](https://docs.puntingform.com.au/docs/modeller-membership)) | Paid. Higher tier prices not published | Best paid add-on. Buy history once, then cancel |
| Wikipedia per-year Cup pages ([2016 example](https://en.wikipedia.org/wiki/2016_Melbourne_Cup)) | Full fields and results | Free, CC BY-SA | Use to verify every Cup row |
| Racing and Sports ([about](https://new.racingandsports.com/about-us)) | Ratings across 13 jurisdictions, used by the VRC to rate internationals | B2B | No confirmed hobby route |
| Form King, FormFav, The Racing API ([comparison](https://puntersedge.online/compare/racing-data-apis-australia)) | AU or UK/IE form via API | A$10 to A$265/month or £499 one-off | Optional. Racing API helps with UK/IE form |
| Kaggle HK datasets ([gdaley](https://www.kaggle.com/datasets/gdaley/hkracing)) | Thousands of Hong Kong races | Free | Pipeline testing only. Poor transfer to a 3200 m handicap |
| Scraping racing.com, Racenet, Racing Australia | Fields and form | Free but blocked | Avoid |

Two legal points matter. Australian racing data is owned by the state principal racing authorities, and unauthorised use "could breach copyright law" ([The Straight](https://thestraight.com.au/bookies-dodge-racing-data-fees-bullet/)). Racing Australia restructured its data distribution in July 2025 partly to strengthen enforcement ([Past The Wire](https://pastthewire.com/racing-australia-announces-new-racing-materials/)). Keep scraped or licensed data out of any public repo.

Internationals need special handling. **23 of the 2026 nominations came from overseas**, mostly Ireland, England, Japan and the USA ([VRC](https://www.vrc.com.au/latest-news/lexus-melbourne-cup-rating-the-internationals/)). The free UK, IRE and FR BSP files give a market-implied ability signal for their lead-up runs. A simple `international` flag plus last-start BSP rank is a reasonable first pass.

The "possibly synthetic" rows need fixing before anything else. Join every (year, horse) to the Wikipedia field and result. Mock rows tend to show up as horses that never ran, wrong placings, two winners, or field sizes above 24. Don't rely on ML detectors here. Classifier-based synthetic-row detectors reach about 0.92 AUC within one table but collapse to near chance across tables ([arXiv 2503.01937](https://arxiv.org/pdf/2503.01937)). Drop any row you can't verify.

## The conditional logit family fits tiny data best

The table ranks alternatives by how well they suit this data size. "Expanded" means after the Betfair-based training set exists.

| Approach | Python tooling | Fit at 14 races | Fit when expanded | Verdict |
|---|---|---|---|---|
| Conditional logit | statsmodels `ConditionalLogit` ([docs](https://statsmodels.org/dev/generated/statsmodels.discrete.conditional_models.ConditionalLogit.html)), xlogit ([docs](https://xlogit.readthedocs.io/)) | Good with 3 to 6 features | Good | Adopt as the core |
| Exploded logit / top-3 Plackett-Luce | statsmodels on duplicated choice sets, or a short PyTorch/JAX loss | Good. Triples the choice sets | Good | Adopt |
| Benter two-stage market blend | 2-coefficient conditional logit on log model prob and log market prob | Good | Good | Adopt |
| Bayesian hierarchical PL | PyMC, NumPyro, cmdstanpy | Good. Pools jockey and trainer effects | Very good | Adopt in phase two |
| GBDT learning-to-rank | `LGBMRanker`, `XGBRanker`, `CatBoostRanker` | Weak. High variance | Reasonable | Use as one input feature to the logit |
| OpenSkill Plackett-Luce ratings | openskill 6.2.0 ([PyPI](https://pypi.org/project/openskill/)) | Useless. Most horses run once | Good feature generator | Adopt for features |
| choix | Per-item strengths only ([API](http://choix.lum.li/en/latest/api.html)) | No covariates | Jockey/trainer ratings only | Skip mostly |
| TabPFN | Small-data tabular model ([Nature 2025](https://pubmed.ncbi.nlm.nih.gov/39780007/)) | Not race-grouped. Unverified here | Same | Optional baseline |
| XGBoost regressor on position | Current | Wrong loss | Wrong loss | Retire or convert to ranker |
| Keras LSTM on recent times | Current | Far too many parameters | Only as a frozen feature extractor trained on thousands of horses | Drop |

The exploded logit deserves attention because it attacks the sample-size problem directly. You model the winner from the full field, then second from the rest, then third from what's left. Bolton and Chapman turned **200 races into 600 choice sets** this way. Standard errors fell about 28% going from depth 1 to 2 and about 17% more from 2 to 3 ([Bolton & Chapman 1986](https://www.Gwern.net/doc/statistics/decision/1986-bolton.pdf)). A Plackett-Luce model with covariates truncated at third place is the same thing. It also matches the top-3 objective exactly. There's no maintained Python package for covariate PL that I could confirm, so either duplicate each race three times for statsmodels or write the 10-line negative log-likelihood in PyTorch or JAX.

The market should enter through Benter's second stage, not as a raw regressor feature. Fit the fundamental model without odds, get out-of-fold probabilities, then fit a 2-parameter conditional logit on log model probability and log market probability. On 3,198 Hong Kong races Benter's results were these ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)).

| Estimate | Pseudo-R² |
|---|---|
| Public odds alone | 0.1218 |
| Fundamental model alone | 0.1245 |
| Combined | 0.1396 |

His key point is that a model's value is the information it adds to the market. Another model with almost identical stand-alone fit added almost nothing when combined. If your model coefficient comes out near zero, report the market-only model as the honest result.

Top-3 probabilities need a proper conversion. Harville's formula renormalises win probabilities successively, but it overrates favourites for 2nd and 3rd and underrates longshots ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)). Henery's normal running-time model fits better on large samples, though all models still show favourite-longshot bias for place betting ([Ali 1998](https://www.stat.berkeley.edu/~aldous/157/Papers/ali.pdf)). Benter fitted a two-parameter correction with γ = 0.81 and δ = 0.65. With 24 runners, exact enumeration of all 24 × 23 × 22 ≈ 12,000 top-3 orderings is trivial in numpy. Use a discounted Harville with literature values rather than fitting the discounts on 14 races. Better still, use the top-3 PL model, which produces these probabilities directly.

Rankers are still worth knowing. All three GBDT libraries take the race as a query group. CatBoost's `QuerySoftMax` is structurally a conditional logit inside a GBDT ([CatBoost docs](https://catboost.ai/docs/en/concepts/loss-functions-ranking.md)). LightGBM's default `min_data_in_leaf` is 20, which is far too high for 330 rows ([LightGBM params](https://lightgbm.readthedocs.io/en/latest/Parameters.html)). Use graded labels (3, 2, 1 for the placegetters, 0 otherwise), shallow trees and strong L2. Ranker scores are on an arbitrary scale. Feed them into the conditional logit rather than reading them as probabilities. A Japanese hobby app takes this route with LightGBM LambdaRank and Platt scaling ([Zenn](https://zenn.dev/mocha_lily/articles/504ab859882e9d)).

## Score per race, against the market, with the race as the unit

Use a strictly proper local scoring rule as the main metric. The log score needed fewer forecast-outcome pairs than Brier or RPS to identify the better forecaster in simulations ([Wheatcroft, arXiv 1908.08980](https://arxiv.org/pdf/1908.08980)). Brier and log loss mix calibration with resolution, so check calibration separately ([scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)).

| Metric | Role | Note |
|---|---|---|
| Winner log loss per race | Primary | Uniform baseline is ln 24 ≈ 3.18 nats |
| Top-3 log loss or PL likelihood of the 1-2-3 | Primary for this task | Needs per-race normalisation |
| Brier on placed/not placed | Secondary | Proper |
| Reliability diagram | Descriptive | 5 bins at most with about 42 placegetters |
| Top-3 hit rate | For reporting | Random picks average 0.375 hits per race |
| NDCG@3, Spearman | Diagnostic | Improper. Don't select models on them |
| ROI | Sanity check | One longshot dominates at 14 races |

Validation must split by race. Use `LeaveOneGroupOut` for the outer loop and `GroupKFold` for the inner tuning loop ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)). Non-nested scores are optimistic, and the sklearn docs note the bias "can be larger on smaller or less stable datasets" ([nested CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)). Add a walk-forward check on the last four or five Cups to catch drift. Watch the usual leaks. Final BSP is leakage if you predict days ahead. Jockey and trainer win rates must use only earlier years. Encoders and scalers must be fitted inside the fold. LSTM sequences must not include the target race.

Build the market baseline by converting odds to 1/odds and normalising out the overround. A Shin correction via `goto_conversion` handles favourite-longshot bias too ([goto_conversion](https://github.com/gotoConversion/goto_conversion)). Then test significance with a paired bootstrap over races of model log loss minus market log loss. Resample races, never rows.

Be realistic about power. Benter's meaningful edge over the public was Δρ² ≈ 0.018 on 3,198 races ([Benter 1994](https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf)). My rough estimate is that this translates to about 0.05 nats per race, while a 14-race confidence interval is around ±0.4 nats. That's roughly eight times too wide to detect a real edge. Top-3 hit rate is no better. Fifteen hits from 42 picks gives a 95% interval of about 22% to 52%. The Cup favourite has historically placed in about 46% of runnings ([Yahoo Sports AU](https://au.sports.yahoo.com/backing-the-melbourne-cup-favourite-are-the-odds-in-your-favour-25413527.html)), so you can't separate the two. Frame results as "competitive with the market" unless you test on hundreds of races. Even large efforts struggle here. One self-reported project on about 94,000 UK and Irish races found the market beat a 1,700-feature model on AUC ([prismix.dev](https://prismix.dev/news/a76839f0aa5e)).

## Small tools, used inside the CV loop

| Tool | Use here | Verdict |
|---|---|---|
| sklearn `Pipeline` + `ColumnTransformer` | Refit all preprocessing per fold | Adopt |
| sklearn `TargetEncoder` ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)) | Jockey, trainer, sire. Accepts `GroupKFold` as `cv` since 1.9 | Adopt until hierarchical effects replace it |
| Sigmoid or temperature calibration ([docs](https://scikit-learn.org/stable/modules/calibration.html)) | 1 to 2 parameters | Adopt. Isotonic needs about 1,000+ samples |
| `goto_conversion` | Odds to probabilities with Shin correction | Adopt |
| Pandera ([docs](https://pandera.readthedocs.io/)) | One winner per race, unique positions, odds > 1, barrier 1 to 24 | Adopt |
| MLflow local ([docs](https://mlflow.org/docs/latest/ml/tracking/)) | Log per-race log loss, market log loss, config and data hashes | Adopt |
| pytest | Probabilities sum to 1 or 3, no target-year leakage, no race in both train and test | Adopt |
| Optuna 5.0 ([docs](https://optuna.readthedocs.io/en/stable/)) | Seeded TPE, 50 trials max | Only after data expansion |
| SHAP ([docs](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)) | Compute per fold and report the spread | Optional. Unstable on 330 rows |
| cookiecutter-data-science v2 ([docs](https://cookiecutter-data-science.drivendata.org/)) | Project layout | Optional |
| DVC, Hydra, Great Expectations | Versioning and config | Overkill at this size |
| betfairlightweight, flumine ([PyPI](https://pypi.org/project/flumine/)) | Live prices and backtesting | Only if you start betting |

GridSearch versus Optuna matters less than the size of the search space. With 14 groups the CV noise is bigger than most hyperparameter effects, so a smarter optimiser mostly finds a better way to overfit the noise. Keep the grid to 6 to 12 configs.

No maintained, Melbourne Cup-specific predictor turned up on GitHub. The most useful references borrow in pieces.

| Reference | What to borrow |
|---|---|
| Betfair DS Topaz greyhound tutorial ([source](https://raw.githubusercontent.com/betfair-datascientists/betfair-datascientists.github.io/master/src/modelling/topazTutorial.md)) | Per-runner model, normalise within race, Brier score, compare to BSP |
| Betfair backtesting tutorial ([link](https://betfair-datascientists.github.io/tutorials/backtestingRatingsTutorial/)) | A sober benchmark. Every strategy lost at BSP, and the author says 10% POT at BSP points to a bug |
| Zenn keiba app ([link](https://zenn.dev/mocha_lily/articles/504ab859882e9d)) | LambdaRank plus Platt scaling, gating feature changes on tracked metrics |
| ethan-eplee/HorseRacePrediction ([repo](https://github.com/ethan-eplee/HorseRacePrediction)) | SHAP and Streamlit demo. Skip its F1-based evaluation |
| Macquarie quant scorecard ([Capital Brief](https://www.capitalbrief.com/briefing/macquaries-ai-each-way-bet-on-melbourne-cup-winner-ddc88077-48d4-409e-8c2a-31a474fdebf0/)) | An interpretable factor baseline. It picked the 2023 winner, Without A Fight |
| LLM tips ([SEN 2025](https://www.sen.com.au/news/2025/11/03/chatgpt-ai-melbourne-cup-tips-winner-half-yours)) | A warning. ChatGPT picked Half Yours but none of its other picks placed. These tips mostly echo the market |

## A seven-step migration path

Steps 1 to 3 work on the current data and fix the biggest errors cheaply. Steps 4 onward depend on the expanded dataset.

| Step | Change | Why | Effort |
|---|---|---|---|
| 1 | Verify every row against Wikipedia, add Pandera checks, tag `verified`, drop the rest. Stop tuning toward 2025 | Bad rows and test-set tuning invalidate every later result | Small |
| 2 | Build the harness: LOGO outer CV, per-race log loss, normalised market baseline, race-level bootstrap, MLflow logging | You can't compare alternatives without it | Small |
| 3 | Keep XGBoost but take out-of-fold scores, fit a 1 to 2 parameter conditional logit on them, compute top-3 via discounted Harville enumeration. Drop the LSTM and the 0.6/0.4 blend | Replaces the softmax-over-positions with fitted, per-race probabilities | Small |
| 4 | Expand training to ANZ races of 2000 m+ from the Betfair CSVs, joined to Punting Form history if budget allows. Hold Cups out as the test set | The largest likely gain | Medium |
| 5 | Make the core model a regularised top-3 Plackett-Luce (exploded conditional logit), then Benter's stage 2 with market odds | Correct likelihood, odds used properly | Medium |
| 6 | Add OpenSkill ratings, a `CatBoostRanker` or `LGBMRanker` score as an input, and UK/IRE BSP signals for internationals | Richer features without extra model capacity | Medium |
| 7 | Move to a Bayesian hierarchical PL in NumPyro or PyMC with pooled jockey and trainer effects | Shrinks noisy per-person rates and gives uncertainty on every probability | Larger |

Freeze steps 1 to 3 before 3 November and log the 2026 predictions. That one race is the only clean test the current data allows.

## Conclusion

The project's real constraint is measurement, not modelling. Fourteen races can't tell a good model from a lucky one, so effort spent tuning is mostly spent fitting noise. Expanding the training data buys a test set as much as a training set. Held-out big staying handicaps such as the Caulfield Cup and Sydney Cup give hundreds of races to judge on instead of fourteen.

The market should be treated as both the baseline and an input. A realistic target is calibrated top-3 probabilities that add measurable information to normalised Betfair prices across many staying races. Picking the Cup winner is the wrong success criterion, since even the bookmaker's favourite manages that only about one year in five.
