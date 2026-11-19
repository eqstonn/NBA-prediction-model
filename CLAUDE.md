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
python code/dynamic_model.py --out priors_pre_2026-27_exact.json --before 2026-10-03   # TRAINING into a NEW file (3 years before the cutoff, ~1 s)
python code/dynamic_model.py --out x.json --before 2026-10-03 --all-history --engine pymc   # every game before the cutoff, sampled (hours)
python code/state_manager.py                             # DESTRUCTIVE: resets priors.json to flat league-average defaults
python code/update.py --start-season                     # new season: priors.json <- frozen snapshot, std 0.1/sqrt(20); clears applied_games.json
python code/update.py --start-season --n 10              # same, prior worth N=10 games (std 0.1/sqrt(10))
python code/update.py --update                           # apply every finished, not-yet-applied 2026-27 game (saves after each game)
python code/update.py --update --through 2026-11-01 --limit 20   # stop at a date / after N games
python code/update.py --trial 5                          # DRY RUN: first 5 games of 2026-27 in memory; prints mean changes + timing, writes nothing
python code/update.py --trial 5 --m 20                   # same with per-game drift tau = 0.1/20 (default M=30)
python code/update.py --replay --from priors_pre_2026-27.json --out priors_2026-27_base.json --log log_base.csv
                                                         # rebuild the season into another file; never touches priors.json/applied_games.json
python code/compare.py log_base.csv log_other.csv        # side-by-side accuracy of 2+ prediction logs

python -m pytest tests                                   # tests (~1 min; real PyMC fits + a fake NBA.com; never touch the real priors.json)
```

No linter or build step. The repo is on GitHub (`eqstonn/NBA-prediction-model`). **Never commit, push or open PRs. The user makes all commits and pushes themselves**, even when a task spec says "commit": make changes, leave them uncommitted, and hand over a commit message plus the exact `git add` commands. `.gitignore` ignores `stats/`, `site/`, `venv/`, `__pycache__/` and `.pytest_cache/`.

## Architecture

All code is in `code/` as flat scripts that import each other by bare module name (`from data_prep import ...`). Run them as `python code/<file>.py`, not with `-m`.

**Training path:** `data_prep.py` → `dynamic_model.py` → a new priors file
- `data_prep.py` reads the Kaggle `TeamStatisticsExtended.csv`, which isn't in the repo. Training looks for it at `stats/TeamStatisticsExtended.csv` or `stats/NBA-DataSet/TeamStatisticsExtended.csv` (the unzipped dataset, currently there), or takes `--csv`. `clean_data_frame(df, before, years)` drops rows without `gameId`/`gameType`, keeps games before `before` and within `years` of it (`years=None`: all history), and sorts chronologically. Called without `before`, it keeps the original behavior: the last 3 years up to today. It then selects the camelCase Four Factor columns (`effectiveFieldGoalPercentage`, `teamTurnoverPercentage`, `offensiveReboundPercentage`, `freeThrowAttemptRate`). Factors are fractions (e.g. eFG ≈ 0.54), not percents.
- `dynamic_model.py --out FILE --before DATE [--years 3 | --all-history] [--engine exact|pymc]` starts every team in the window from `state_manager.default_priors` (flat league averages, std 0.05). Each row (one per team per game, so each game twice) goes through `bayes_update.update_game(priors, row, tau=0)`, and the result is written to `--out`. It refuses to write `priors_pre_2026-27.json`, `priors.json`, `applied_games.json` or `priors_2026-27_base.json`. **Use `--before 2026-10-03`**, the first day of the 2026-27 preseason. The Kaggle file gives a `gameType` only to *this* season's preseason games (Oct 3–16, 2026; older preseasons have none and are dropped), so a later cutoff would train on 2026-27 preseason games and add a 38th team ("Lions", `50015`). With 2026-10-03 and the default 3 years, training gets exactly the original's 3,963 games (2023-10-24 to 2026-06-13: regular season, playoffs, play-in, the 2025 Cup final, and 2026 All-Star games, which only touch the 7 non-NBA All-Star team keys) and its 37 teams. The old in-place mode (update `priors.json`, window relative to today) is gone, because it double-counted this season.
- **`priors_pre_2026-27_exact.json`** is that 3-year exact retrain. It differs from the PyMC-trained frozen file by a median 0.0026 (max 0.025) on NBA means, and the frozen stds are ~8% smaller. Same window, same game types, no restart double-counting (all three were checked by sweeping the alternatives). The cause is MCMC error compounding over ~250 chained fits per curve with tau = 0.
- `bayes_update.py` holds the shared per-row update. `update_game(priors, game, tau, random_seed, engine)` takes `game = {"team", "opponent", "eFG", "TOV", "ORB", "FTR"}`, the team's offensive line. Per factor the model is offense `N(m_o, s_o²)`, defense `N(m_d, s_d²)`, observed `N((o+d)/2, 0.05²)`. It writes back each curve's posterior mean/std. With `tau > 0`, the curves in this fit are first widened to `sqrt(std² + tau²)`.
  - **`engine="exact"` (default everywhere):** the closed-form conjugate (Kalman) update in `_fit_exact`. With `V = (s_o² + s_d²)/4 + 0.05²` and `surprise = y − (m_o + m_d)/2`: `m_o' = m_o + (s_o²/2)/V · surprise`, `s_o' = sqrt(s_o² − (s_o²/2)²/V)`, and the same for defense. Instant, exact, no seeds needed.
  - **`engine="pymc"`:** the original PyMC model in `_fit_pymc` (NUTS, `pm.sample(1000)`). Same answer plus sampling noise: over the 2026-27 replay, median 0.0012 and max 0.0050 from exact, with no bias. Kept for model extensions that aren't Normal/linear (bounded priors, Student-t, learned σ, hierarchical).
  - pymc is imported lazily (`_pymc()`), only on the first PyMC fit; `PYTENSOR_FLAGS="cxx="` is set at the top of the module before that. **Import pymc only through this module.**
- `--engine exact|pymc` is accepted by `dynamic_model.py` and by `update.py` (`--update`, `--replay`, `--trial`). `applied_games.json` entries record the engine that applied them. The first 210 games have no `engine` field; they were all applied with PyMC.
- `state_manager.default_priors(team_ids)` returns the flat starting curves; `initialize_priors` writes them to `priors.json` (destructive).

**In-season path:** `priors_pre_2026-27.json` (frozen) → `update.py` → `priors.json` + `applied_games.json`
- `update.start_season(n)` reads only the frozen snapshot. For the 30 NBA teams (`schedule.NBA_TEAMS`) it keeps every mean and sets every std to `0.1/sqrt(n)`, so last season counts as `n` games of evidence. The 7 non-NBA teams are copied unchanged. It writes `priors.json` and resets `applied_games.json` to `{}`, which maps `game_id` → info for games already applied.
- `update.write_json` writes atomically and raises `PermissionError` if pointed at the frozen snapshot. Route all writes through it.
- `boxscores.py` is the 2026-27 data source (NBA.com via nba_api, not Kaggle). Its only network functions are `_download_game_log`, `_download_schedule` and `_download_box_score`; tests replace all three (`tests/fake_nba.py`).
  - **Endpoints:** every request goes through nba_api to `https://stats.nba.com/stats/<endpoint>`: `leaguegamelog`, `scheduleleaguev2` and `boxscorefourfactorsv3`.
  - **Game list = game log ∪ schedule.** The game log is unreliable: on 2026-11-17/18 it listed 205, 204, 209 games and kept leaving out the final DEN @ IND game (`0022600189`, Nov 7, Mexico City). So `fetch_finished_games` also takes the `ScheduleLeagueV2` games with `gameStatus == 3`. `combine_game_lists` merges the two by gameId (the game log's details win) and prints a `WARNING:` for each game missing from either list or differing in date, teams or score. `run_update` returns these as `"disagreements"`.
  - **Which games:** from the game log, only the season types `"Regular Season"` and `"IST"` (NBA Cup) are requested, nothing else. Schedule games get their season type from the gameId and the NBA Cup label/subtype. `select_games` keeps game IDs starting `002` (regular season, which includes Cup group games, quarterfinals and semifinals) or `006` (Cup final, listed only under IST), played on or after `OPENING_NIGHT = 2026-10-20`, sorted by (date, gameId). Preseason is `001`/`"Pre Season"`, play-in `005`, playoffs `004`.
  - **Four factors:** from NBA.com's `BoxScoreFourFactorsV3`, the source of the Kaggle columns. **Don't recompute factors from basic box scores**: NBA.com's ORB%/TOV% include team rebounds and its own possession count, so simple formulas are off by up to ~0.05. `get_game_factors` returns `(factors, None)` or `(None, reason)`. Complete box scores are cached in `stats/boxscores/<gameId>.json` (gitignored); incomplete ones are never cached.
  - Every request is throttled (0.6 s) and retried 4 times with backoff of 2, 4 and 8 s.
- `update.apply_game(priors, game, m)` makes two `update_game` calls per game (home offense vs away defense, then away vs home) with `tau = 0.1/m` (M=30). So each of the two teams' 8 curves widens once per game. Stds then settle near ~0.019 instead of collapsing.
- `update.run_update(through, limit, m)` (`--update`) loads `applied_games.json`, skips games already in it, and applies the rest oldest first.
  - **Skipped games:** a game whose box score can't be downloaded or is incomplete is skipped and **not recorded**, so the next run retries it. It is then applied after later-dated games, slightly out of order.
  - **Crash-safe saving:** each game is saved with three atomic writes: `applied_games.json` gets the entry with `"pending": <fingerprint of new priors>`, then `priors.json`, then the pending mark is removed. `load_applied()` resolves a leftover pending entry by fingerprint (kept if `priors.json` has the game, dropped otherwise). A fit error aborts the run with nothing half-saved.
  - **`applied_games.json` entries:** `{gameId: {"date", "matchup", "score", "nba_cup"}}`. Its key order is the order games were actually applied.
  - **Seeded fits (pymc engine):** PyMC fits use `random_seed = update.fit_seed(gameId)` (the gameId as an int), so runs are reproducible. The first 210 games (through 2026-11-17) were applied before this, without seeds.
- **Prediction log** (`predlog.py`): before a game is applied, `log_prediction` predicts it from the current priors with `predict.predict_game(..., seed=predlog.PREDICT_SEED)` and appends a row to `prediction_log.csv` (`--log` overrides). Columns: `date, game_id, home, away, predicted_winner, win_pct, proj_home, proj_away, actual_home, actual_away`; `win_pct` is the predicted winner's probability in percent. `append_rows` skips game_ids already in the file, so reruns never duplicate rows. Each run ends by printing the log's winner accuracy, spread MAE and Brier score (`predlog.metrics`). The log only covers games applied after this feature was added; `log_base.csv` from a replay covers the whole season.
- **Replay** (`update.replay`, `--replay --from --out --log`): reset from `--from` (N via `--n`), then every game in `applied_games.json` sorted by (date, gameId), each predicted + logged before it is applied, from the box-score cache (no network if cached). It writes only `--out` (at the end) and `--log` (rebuilt from scratch), and refuses outputs that are `priors.json`, `applied_games.json`, `prediction_log.csv` or the `--from` file. Use it to test model changes on the same season, then `compare.py` the logs.
- **`compare.py`** scores 2+ logs on the games present in every log and marks the best value per row with `*`.
- Cost: with the exact engine, fitting is instant (the whole 210-game replay takes ~1 s), so the time goes to the simulation for the prediction log and ~0.9 s per box-score download. With `--engine pymc`, it's about 3–5 s per game on AC power (2 fits, mostly PyTensor recompiling the model each time); the 210-game PyMC replay took 69 min on battery. **On battery the laptop throttles badly**, so run PyMC jobs plugged in.

**Prediction path:** `priors.json` → `simulation.py` → `predict.py` → `build_site.py` (+ `schedule.py`)
- `simulation.simulate_factors(team_a, team_b, priors, seed=None)` does the following (`seed=None` = fresh randomness, as the website uses; an int = reproducible, as the prediction log uses):
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
- **Never train on 2026-27 games, including the preseason.** Always pass `--before 2026-10-03` to `dynamic_model.py`; the in-season updater applies this season's games.
- **Model comparisons so far** (210 games, through 2026-11-17, `compare.py`): `log_base.csv` (PyMC training + PyMC season) gets 69.0% of winners, spread MAE 12.08, Brier 0.2085. `log_exact.csv` (exact training + exact season) gets 71.0%, 12.02, 0.2065. The gain comes from the clean retraining, not the season engine (same start, engine only: Brier −0.0005). It is 1.5 standard errors, so not yet significant.
- **Trained stds collapsed** to ~0.005 because training never re-inflates them (tau=0). The season reset (`0.1/sqrt(N)`) plus per-game tau is what fixes this for 2026-27.
- **Per-game mean shift ≈ 0.093 × (observed − expected)** at season-start stds. A game that misses expectation by more than ~0.11 (common for ORB/FTR) moves a mean by more than 0.01, so a fixed cap like "< 0.01" fails on real games. `test_per_game_mean_shifts_damped` checks `shift ≤ 0.1 × |observed − expected|` instead.
- **Smart App Control is on** on this machine. The first import of scipy/pandas DLLs after an install can be blocked ("An Application Control policy has blocked this file") or take ~35 s while Windows checks them. Re-running usually works.
- NBA.com's CDN schedule JSON returns 403, but `stats.nba.com` through `nba_api` works.
- `state_manager.initialize_priors` uses a CWD-relative `priors.json`; `dynamic_model.py`, `update.py` and `predict.py` resolve paths from `REPO_ROOT`.
- **Windows compatibility:** don't use `strftime` codes like `%-d` or `%-I` (Linux/macOS only). Always pass `encoding="utf-8"` when reading or writing files, because the default here is cp1252.
- `build_site.py` uses `zoneinfo` for US Eastern time. On Windows this needs the `tzdata` package, which is listed in `requirements.txt`.
