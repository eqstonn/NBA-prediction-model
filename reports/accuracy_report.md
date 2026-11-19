# Accuracy report: the live model vs Vegas and simple baselines

Generated 2026-11-19 by `python code/report.py`. Live model as in `params.py`: exact engine, fitted σ in the update and the simulation, team paces, home-court boost, rest (back-to-backs), 10,000-draw simulation.

## Summary

- **Training seasons, live-procedure backtest (2024-25, 2025-26; 2,638 games):** model 66.5% winners, Brier 0.2135; Vegas closing spread 68.9%, Brier 0.1994. Model − Vegas: Brier +0.0140 ± 0.0019 (+7.3 SE), accuracy -2.4 pts ± 0.8 (-3.2 SE). Vegas is better by 2+ SE on winners picked, Brier, log loss, margin MAE, total MAE.
- **The live procedure vs the training-pass backtest** (secondary, 2,636 common games): Brier -0.0120 ± 0.0015 (-8.1 SE), accuracy +4.0 pts ± 0.8 (+5.0 SE) (live − training pass).
- **Calibration:** live-procedure backtest underconfident (average confidence 62.5% vs 66.5% of winners picked); 2026-27 (210 games) calibrated within noise (average confidence 62.5% vs 67.6% of winners picked).
- **2026-27 with closing lines: 0 games so far.** Line collection started on 2026-11-19, after the 210 games logged so far; this comparison grows as `update.py --update` / `lines.py` save lines and the games are played.
- **2026-27, all 210 logged games (no Vegas yet):** model 67.6% winners, Brier 0.2110; home team 51.4%; better record 65.7%.

## Resume bullets

- Built a Bayesian NBA game predictor in Python: Four Factors offense/defense ratings updated game by game with conjugate Normal (Kalman) updates, and a 10,000-draw Monte Carlo game simulation with team pace, home-court and rest effects fitted by one-step-ahead likelihood.
- Backtested the live procedure (preseason ratings trained only on earlier seasons, then updated after every game) on 2,638 games of 2024-25 and 2025-26: the model picked 66.5% of winners vs 68.9% for Vegas closing lines (Brier 0.2135 vs 0.1994), closing 71% of the Brier gap between always picking the home team and Vegas.
- On the same games the model beat always picking the home team on both winners picked (66.5% vs 55.0%) and Brier (0.2135 vs 0.2475), and beat picking the team with the better record on Brier (0.2135 vs 0.2265) and was level on winners picked (66.5% vs 65.2%).
- Runs live as a daily pipeline (NBA.com data, crash-safe updates, every prediction logged before tip-off): 67.6% of winners and Brier 0.2110 over the first 210 games of 2026-27 (±3.5 pts), vs 51.4% for the home team and 65.7% for the better record.

Every number above is from this report; the training-season numbers are backtests (see the caveats).

## Method

- **Model, main training backtest (`backtest.py`): the live in-season procedure.** For each season, preseason priors come from the live training recipe (flat start, every game of the 3 years before Oct 3 of that season, as `priors_pre_2026-27_pace-home-rest.json` was trained before 2026-10-03), then the season reset (N = 20: stds 0.1/√20), then each game is predicted with `predict.predict_game` (the prediction log's seed) and applied with drift (M = 30), as `update.py --update` does. 2024-25: priors trained on 2,639 games (2022-02-03 to 2024-06-17, cutoff 2024-10-03), then 1,316 games played; 2025-26: priors trained on 3,954 games (2022-10-18 to 2025-06-22, cutoff 2025-10-03), then 1,322 games played. The Kaggle file has no game types for 2021-22, so 2024-25's priors rest on 2022-23 and 2023-24 only. Playoff and play-in games are included.
- **Model, secondary training backtest:** `train_calibration.py`'s one-step-ahead pass over the 3-year window before 2026-10-03 (2023-10-24 to 2026-06-13): ratings start flat, every game updates them with the TRAINING update (tau = 0, no season reset), and each game after the first season (2,636 games) is simulated from the ratings as they stood. Its ratings never forget, so by 2025-26 a game from 2023-24 weighs as much as last night's.
- **Model, 2026-27:** the logged predictions, each made before the game (`log_pace-home-rest.csv` (210)).
- **Vegas (spread):** P(home wins) = Φ(−spread_home / SD), SD = **13.83 points**, fitted on the 3,956 training games with a line (the RMS of actual home margin + spread; the mean residual is -0.09, i.e. lines are unbiased to within a fraction of a point). No 2026-27 game is used. Margin projection = −spread, total projection = the posted total.
- **Vegas (moneyline, no vig):** the median across books of each book's vig-free home probability (`lines.py`). The Kaggle file's moneylines stop in mid 2022-23, so this exists only for 2026-27 games.
- **Home team:** always the home team, P = 0.549 (home win rate over the 3,956 training games). **Better record:** the team with the better win% in that season before that day (0 games = .500), P = 0.649 (how often that pick won in training, 3,798 games); equal records fall back to the home team.
- **Metrics:** winner accuracy (a 50/50 pick counts half), Brier = mean (P(home) − home won)², log loss (P clipped to [0.0001, 0.9999]), margin / total MAE vs the actual result, calibration in confidence bins. All predictors in a table are scored on the same games. Paired differences are per-game differences, mean ± SE (SD/√n); |z| < 2 is within noise.
- **Data:** Kaggle "NBA Betting Data | October 2007 to June 2026" (`nba_2008-2026.csv`, SHA-256 `729eb2ead85c14af…`), matched to 3,956 of 3,956 training games (100.0%). 2026-27 lines: `data/lines_2026-27.csv`, 4 snapshots of 4 games; the closing line is a game's last snapshot before tip-off.

## Training seasons, main: live-procedure backtest (2,638 games)

| Predictor | Games | Winner accuracy | Brier | Log loss | Margin MAE | Total MAE | Avg confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model | 2638 | 66.5% | 0.2135 | 0.6161 | 11.37 | 15.08 | 62.5% |
| Vegas (spread) | 2638 | 68.9% | 0.1994 | 0.5823 | 10.80 | 14.48 | 68.5% |
| Home team | 2638 | 55.0% | 0.2475 | 0.6881 | – | – | 54.9% |
| Better record | 2638 | 65.2% | 0.2265 | 0.6455 | – | – | 64.5% |

**Paired differences vs Vegas (spread)** (negative Brier / log loss / MAE = better than Vegas):

| vs Vegas (spread) | Accuracy | Brier | Log loss | Margin MAE | Total MAE |
|---|---:|---:|---:|---:|---:|
| Model | -2.4 pts ± 0.8 (-3.2 SE) | +0.0140 ± 0.0019 (+7.3 SE) | +0.0338 ± 0.0043 (+7.9 SE) | +0.57 ± 0.08 (+7.6 SE) | +0.60 ± 0.10 (+6.0 SE) |
| Home team | -13.9 pts ± 1.2 (-11.6 SE) | +0.0481 ± 0.0032 (+14.8 SE) | +0.1058 ± 0.0072 (+14.6 SE) | – | – |
| Better record | -3.7 pts ± 0.8 (-4.4 SE) | +0.0271 ± 0.0026 (+10.3 SE) | +0.0632 ± 0.0059 (+10.8 SE) | – | – |

**Model vs the baselines** (positive accuracy / negative Brier and log loss = the model is better):

| Model − baseline | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Home team | +11.4 pts ± 1.2 (+9.3 SE) | -0.0340 ± 0.0025 (-13.7 SE) | -0.0720 ± 0.0053 (-13.5 SE) |
| Better record | +1.3 pts ± 0.8 (+1.5 SE) | -0.0131 ± 0.0022 (-6.0 SE) | -0.0294 ± 0.0047 (-6.3 SE) |

**Disagreements:** the model and Vegas picked different winners in 408 games (15.5%); the model was right in 172 (42.2%), Vegas in 236. (0 pick'em lines had no Vegas pick.) On average the model's margin is 3.34 pts and its total 4.33 pts from the Vegas numbers.

**By season** (same games, split):

| Season | Games | Model acc. | Vegas acc. | Model Brier | Vegas Brier | Brier diff (model − Vegas) |
|---|---:|---:|---:|---:|---:|---:|
| 2024-25 | 1316 | 66.7% | 68.8% | 0.2124 | 0.2012 | +0.0112 ± 0.0025 (+4.6 SE) |
| 2025-26 | 1322 | 66.2% | 69.0% | 0.2145 | 0.1977 | +0.0169 ± 0.0029 (+5.7 SE) |

**Calibration**

| Predictor | Confidence bin | Games | Avg confidence | Accuracy | Accuracy − confidence (pts) |
|---|---:|---:|---:|---:|---:|
| Model | 50–55% | 614 | 52.4% | 52.8% | +0.4 |
| Model | 55–60% | 605 | 57.4% | 59.3% | +1.9 |
| Model | 60–65% | 472 | 62.4% | 70.6% | +8.2 |
| Model | 65–70% | 396 | 67.4% | 72.2% | +4.8 |
| Model | 70–75% | 294 | 72.3% | 78.2% | +5.9 |
| Model | 75–80% | 157 | 77.2% | 82.8% | +5.6 |
| Model | 80–90% | 99 | 83.7% | 90.9% | +7.2 |
| Model | 90–100% | 1 | 92.2% | 100.0% | +7.8 |
| Vegas (spread) | 50–55% | 265 | 53.8% | 49.8% | -4.0 |
| Vegas (spread) | 55–60% | 503 | 58.4% | 57.7% | -0.7 |
| Vegas (spread) | 60–65% | 276 | 62.7% | 60.1% | -2.6 |
| Vegas (spread) | 65–70% | 471 | 66.9% | 67.1% | +0.2 |
| Vegas (spread) | 70–75% | 357 | 72.0% | 76.8% | +4.8 |
| Vegas (spread) | 75–80% | 372 | 77.4% | 77.2% | -0.2 |
| Vegas (spread) | 80–90% | 353 | 84.6% | 88.1% | +3.5 |
| Vegas (spread) | 90–100% | 41 | 92.2% | 100.0% | +7.8 |

![Calibration chart](calibration.png)

## Training seasons, secondary: training-pass backtest (2,636 games)

The same seasons predicted from the training pass (no season reset, no drift), the setup of `train_calibration.py`. Kept for comparison; it is not how the model runs live.

| Predictor | Games | Winner accuracy | Brier | Log loss | Margin MAE | Total MAE | Avg confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model | 2636 | 62.5% | 0.2255 | 0.6416 | 11.91 | 15.71 | 62.5% |
| Vegas (spread) | 2636 | 68.9% | 0.1994 | 0.5823 | 10.80 | 14.48 | 68.5% |
| Home team | 2636 | 55.0% | 0.2475 | 0.6882 | – | – | 54.9% |
| Better record | 2636 | 65.2% | 0.2266 | 0.6455 | – | – | 64.5% |

**Paired differences vs Vegas (spread)** (negative Brier / log loss / MAE = better than Vegas):

| vs Vegas (spread) | Accuracy | Brier | Log loss | Margin MAE | Total MAE |
|---|---:|---:|---:|---:|---:|
| Model | -6.4 pts ± 0.9 (-7.0 SE) | +0.0261 ± 0.0025 (+10.6 SE) | +0.0594 ± 0.0055 (+10.8 SE) | +1.11 ± 0.10 (+11.3 SE) | +1.23 ± 0.14 (+8.7 SE) |
| Home team | -13.9 pts ± 1.2 (-11.6 SE) | +0.0481 ± 0.0032 (+14.8 SE) | +0.1059 ± 0.0072 (+14.6 SE) | – | – |
| Better record | -3.7 pts ± 0.8 (-4.4 SE) | +0.0271 ± 0.0026 (+10.3 SE) | +0.0632 ± 0.0059 (+10.8 SE) | – | – |

**Model vs the baselines** (positive accuracy / negative Brier and log loss = the model is better):

| Model − baseline | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Home team | +7.5 pts ± 1.2 (+6.2 SE) | -0.0220 ± 0.0026 (-8.5 SE) | -0.0466 ± 0.0056 (-8.4 SE) |
| Better record | -2.7 pts ± 1.0 (-2.7 SE) | -0.0010 ± 0.0027 (-0.4 SE) | -0.0039 ± 0.0057 (-0.7 SE) |

**Disagreements:** the model and Vegas picked different winners in 603 games (22.9%); the model was right in 217 (36.0%), Vegas in 386. (0 pick'em lines had no Vegas pick.) On average the model's margin is 4.42 pts and its total 6.47 pts from the Vegas numbers.

**By season** (same games, split):

| Season | Games | Model acc. | Vegas acc. | Model Brier | Vegas Brier | Brier diff (model − Vegas) |
|---|---:|---:|---:|---:|---:|---:|
| 2024-25 | 1314 | 63.8% | 68.8% | 0.2198 | 0.2012 | +0.0186 ± 0.0030 (+6.3 SE) |
| 2025-26 | 1322 | 61.1% | 69.0% | 0.2312 | 0.1977 | +0.0336 ± 0.0039 (+8.5 SE) |

**Live procedure − training pass** on the 2,636 games both scored (negative Brier / log loss / MAE and positive accuracy = the live procedure is better):

|  | Accuracy | Brier | Log loss | Margin MAE |
|---|---:|---:|---:|---:|
| Live − training pass | +4.0 pts ± 0.8 (+5.0 SE) | -0.0120 ± 0.0015 (-8.1 SE) | -0.0255 ± 0.0032 (-8.1 SE) | -0.54 ± 0.06 (-8.6 SE) |

## 2026-27 season

### With Vegas closing lines (0 games)

No 2026-27 game has both a prediction and a closing line yet. Lines are saved from 2026-11-19 on (each `update.py --update` and `python code/lines.py` run); the games enter the prediction log when `--update` applies them, so this section fills in over the season. Re-run `python code/report.py` to update it.

### All logged games, model vs simple baselines (210 games, no Vegas)

| Predictor | Games | Winner accuracy | Brier | Log loss | Margin MAE | Total MAE | Avg confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model | 210 | 67.6% | 0.2110 | 0.6109 | 11.96 | 14.82 | 62.5% |
| Home team | 210 | 51.4% | 0.2510 | 0.6952 | – | – | 54.9% |
| Better record | 210 | 65.7% | 0.2224 | 0.6367 | – | – | 63.1% |

| Model − baseline | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Home team | +16.2 pts ± 4.3 (+3.7 SE) | -0.0400 ± 0.0087 (-4.6 SE) | -0.0842 ± 0.0188 (-4.5 SE) |
| Better record | +1.9 pts ± 3.5 (+0.5 SE) | -0.0114 ± 0.0090 (-1.3 SE) | -0.0258 ± 0.0191 (-1.4 SE) |

| Predictor | Confidence bin | Games | Avg confidence | Accuracy | Accuracy − confidence (pts) |
|---|---:|---:|---:|---:|---:|
| Model | 50–55% | 46 | 52.4% | 52.2% | -0.2 |
| Model | 55–60% | 53 | 57.4% | 69.8% | +12.4 |
| Model | 60–65% | 33 | 62.5% | 60.6% | -1.9 |
| Model | 65–70% | 37 | 67.3% | 70.3% | +3.0 |
| Model | 70–75% | 20 | 72.8% | 80.0% | +7.2 |
| Model | 75–80% | 12 | 77.3% | 91.7% | +14.4 |
| Model | 80–90% | 9 | 82.5% | 88.9% | +6.4 |

With 210 games, one standard error of the accuracy is about 3.5 points.

## Caveats

- **The training seasons are a backtest, not a forecast.** Each prediction uses only earlier games for the team ratings, but the model's constants (game noise σ, home boost, rest effect, pace noise, points formula) were fitted on the 2023-24 to 2025-26 window, which flatters the model a little. The Vegas SD (one number) was fitted on it too. 2026-27 is the only true out-of-sample test.
- **The season reset and drift (N = 20, M = 30) are the live settings,** used as they are; no other values were tried on these backtests.
- **The live-procedure backtest differs from the live run only in its data:** the games come from the Kaggle CSV (the same NBA.com four factors and possessions), and playoff / play-in games are included.
- **Kaggle lines:** the dataset gives one spread and total per game and doesn't say exactly when they were taken; treat them as closing or near-closing lines.
- **2026-27 "closing" lines** are the last snapshot before tip-off, which can be hours before the true close when lines were only fetched in the morning (median shown above).
- **The 2026-27 season so far is small.** Its first 210 games' accuracy was about 2 SE above what their own margin errors predict (CLAUDE.md, Phase 7), so expect it to drift down.
