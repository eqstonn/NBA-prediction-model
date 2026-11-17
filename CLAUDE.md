# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Bayesian NBA game predictor. Each team has independent **offensive** and **defensive** bell curves (mean + std) for Dean Oliver's Four Factors: eFG%, TOV%, ORB%, FTR. These are learned game-by-game with PyMC and stored in `priors.json`. Predictions are 10,000-draw Monte Carlo sims that turn the factors into points.

- **Strength of schedule:** a game's expected factor = `(team offense + opponent defense) / 2`. The observed value updates *both* curves, so a strong game against a good defense moves the rating more than the same game against a bad one.
- **Prior → posterior:** after each game, the posterior mean/std is written back and becomes that team's prior for its next game.

### Current state and goal

- `priors.json` was trained on all games **before the current (2026-27) season** started, using the Kaggle "NBA Dataset: Box Scores and Stats (1947 - Today)" CSV.
- **Next goal:** take games played this season, update the priors from each one (online learning, no retraining from scratch), and keep predicting upcoming games. The README's v2.0 "Automated Daily Updater" (fetch box scores with `nba_api`) is this feature.
- **Other v2.0 roadmap items:** dynamic pace distributions (replace the fixed 100 possessions), time decay so recent games weigh more, home-court adjustment to the factors, and garbage-time filtering (15+ point margins in the 4th quarter).

## Commands

The user is on **Windows** with a venv in `venv/`. Run everything from the **repo root**.

```powershell
.\venv\Scripts\Activate.ps1          # activate (PowerShell)
pip install -r requirements.txt

python code/build_site.py                                # predictions site for today + next 6 days -> site/index.html
python code/build_site.py --start 2026-11-16 --days 7    # custom window
python code/build_site.py --priors other_priors.json     # or set $env:NBA_PRIORS
python code/simulation.py                                # quick CLI sim of 3 hard-coded matchups
python code/dynamic_model.py                             # TRAINING: updates priors.json in place (slow, see below)
python code/state_manager.py                             # DESTRUCTIVE: resets priors.json to flat league-average defaults
python code/update.py --start-season                     # new season: priors.json <- frozen snapshot, std 0.1/sqrt(20); clears applied_games.json
python code/update.py --start-season --n 10              # same, prior worth N=10 games (std 0.1/sqrt(10))

python -m pytest tests                                   # tests (they use a temp folder, never the real priors.json)
```

No linter or build step. The repo is on GitHub (`eqstonn/NBA-prediction-model`). **The user commits and pushes themselves:** make changes, leave them uncommitted, and hand over a commit message. `.gitignore` ignores `stats/`, `site/`, `venv/`, `__pycache__/` and `.pytest_cache/`.

## Architecture

All code is in `code/` as flat scripts that import each other by bare module name (`from data_prep import ...`). Run them as `python code/<file>.py`, not with `-m`.

**Training path:** `data_prep.py` → `dynamic_model.py` → `priors.json`
- `data_prep.py` reads `stats/TeamStatisticsExtended.csv` (Kaggle; not in the repo). It drops rows without `gameId`/`gameType`, keeps the **last 3 years relative to `pd.Timestamp.now()`**, sorts chronologically, and selects the camelCase Four Factor columns (`effectiveFieldGoalPercentage`, `teamTurnoverPercentage`, `offensiveReboundPercentage`, `freeThrowAttemptRate`). Factors are fractions (e.g. eFG ≈ 0.54), not percents.
- `dynamic_model.py` has one row per team per game, so each game is processed twice, once from each side. For each row it builds a small PyMC model:
  - team offense priors + opponent defense priors
  - observed factors with fixed σ=0.05
  - `pm.sample(1000)`
  
  It then writes the posterior mean/std back into the dict. It checkpoints to `priors.json` every 50 rows.
- `state_manager.py` creates the initial `priors.json` with the same default curves for every team.

**In-season path:** `priors_pre_2026-27.json` (frozen) → `update.py` → `priors.json` + `applied_games.json`
- `update.start_season(n)` reads only the frozen snapshot. For the 30 NBA teams (`schedule.NBA_TEAMS`) it keeps every mean and sets every std to `0.1/sqrt(n)`, so last season counts as `n` games of evidence. The 7 non-NBA teams are copied unchanged. It writes `priors.json` and resets `applied_games.json` to `{}`, which maps `game_id` → info for games already applied.
- `update.write_json` writes atomically and raises `PermissionError` if pointed at the frozen snapshot. Route all writes through it.

**Prediction path:** `priors.json` → `simulation.py` → `predict.py` → `build_site.py` (+ `schedule.py`)
- `simulation.simulate_factors(team_a, team_b, priors)` does the following:
  - draws each side's offense and defense factors, blends them `(o + d)/2`
  - adds game noise (σ 0.05 for eFG/ORB, 0.02 for TOV/FTR)
  - converts to points with `pace=100`, `ft_pct=0.77`, `possessions = pace·(1−TOV)·(1+ORB·(1−eFG))`, `points = possessions·(2·eFG + FTR·ft_pct)`

  No home-court effect yet.
- `predict.py` is the **only** interface between the website and the model. `predict_game(home_id, away_id, model)` returns `{home_win_pct, away_win_pct, home_score, away_score}`, or `None` if a team isn't in the priors. If the model's internals change, keep this return shape and only edit this file. Home team = `team_a`.
- `schedule.fetch_games(start, days)` tries NBA.com's public schedule JSON and falls back to ESPN's scoreboard. Both are normalized to one game dict shape (documented at the top of `schedule.py`). ESPN abbreviations are mapped back to NBA team IDs. NBA.com currently returns 403 to our request, so in practice ESPN is used.
- `build_site.py` writes a static `site/index.html` (self-contained HTML/CSS/JS) and `site/predictions.json`. Finished games are graded against the model's pick.

### `priors.json` shape

Keys are **NBA.com team IDs as strings** (e.g. `"1610612738"` = BOS):
```
{ "<teamId>": { "offense": {"eFG": {"mean", "std"}, "TOV": ..., "ORB": ..., "FTR": ...},
                "defense": { ...same four... } } }
```
It has 37 keys: the 30 NBA teams plus a few non-NBA teams from the Kaggle data. "Defense" factors are what the team *allows*.

## Gotchas

- **`priors_pre_2026-27.json` is a frozen, read-only snapshot** of `priors.json` from before any 2026-27 games were applied (git tag `pretrained-2025-26`). No code may ever write to it. Read it to reset or compare. To restart in-season updates, copy it over `priors.json`, never the other way.
- **Updates are not idempotent.** Feeding the same game twice moves the priors twice. Any in-season updater must record which games it has already applied. Note that ESPN `game_id`s are ESPN event IDs, not NBA `gameId`s.
- **Running `dynamic_model.py` as-is would double-count.** Its 3-year window is relative to today and overlaps the games already in `priors.json`. In-season updates must only feed games after the training cutoff.
- **Posterior stds have collapsed.** Each update narrows std and nothing re-inflates it between games, so trained stds are ~0.005. New games will move the means only slightly unless variance is added back (the time-decay item on the roadmap).
- `dynamic_model.py` sets `PYTENSOR_FLAGS="cxx="` (no C++ compiler on this machine) *after* `import pymc`, so it may not take effect. Sampling is slow (one MCMC run per row).
- `dynamic_model.py` and `state_manager.py` use CWD-relative paths (`priors.json`, `stats/...`). `predict.py` resolves `priors.json` from `REPO_ROOT`.
- **Windows compatibility:** don't use `strftime` codes like `%-d` or `%-I` (Linux/macOS only). Always pass `encoding="utf-8"` when reading or writing files, because the default here is cp1252.
- `build_site.py` uses `zoneinfo` for US Eastern time. On Windows this needs the `tzdata` package, which is listed in `requirements.txt`.
