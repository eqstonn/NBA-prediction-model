"""
In-season updates to priors.json.

    python code/update.py --start-season          # reset priors for the new season (N=20) from the live preseason file
    python code/update.py --start-season --n 10   # looser priors: worth 10 games of evidence
    python code/update.py --start-season --from priors_pre_2026-27.json --pace off --rest off
                                                  # reset from the frozen snapshot (no team paces)
    python code/update.py --update                # apply every finished, not-yet-applied game
    python code/update.py --update --through 2026-11-01 --limit 20   # stop at a date / after 20 games
    python code/update.py --update --lines off    # same, without saving betting lines (lines.py)
    python code/update.py --replay --from priors_pre_2026-27.json --out priors_x.json --log log_x.csv
                                                  # rebuild the season into another file (model experiments)
    python code/update.py --trial 5               # dry run: first 5 games of the season, writes nothing
    python code/update.py --trial 5 --m 20        # same with more drift per game (tau = 0.1/20)

Season reset
------------
priors_pre_2026-27_pace-home-rest.json is the live model's preseason priors (exact training
with team paces, the home boost and the rest effect: the defaults). priors_pre_2026-27_pace-home.json
(the previous live model's) and priors_pre_2026-27.json (the frozen snapshot of the original
training) are kept too. All three are only ever READ here (write_json refuses them). A reset
rebuilds priors.json from the first one (or --from):

  * NBA teams: every offense/defense factor keeps its mean, but its std is set to
    0.1 / sqrt(N). Training collapsed the stds to ~0.005, which would make this
    season's games barely move the ratings. Re-inflating them treats last season's
    rating as if it were worth only N games of evidence.
  * Non-NBA teams (the extra Kaggle teams): copied unchanged.

It also clears applied_games.json, the record of which games have already been fed
into priors.json ({game_id: {...}}), since the reset priors contain none of them.

Season games
------------
apply_game runs one finished game (boxscores.py shape) through the same Bayesian
update as training (bayes_update.update_game), once per side. Before each fit the
curves being fit are widened to sqrt(std^2 + tau^2), tau = 0.1/M (M=30), so each of
the two teams' 8 curves drifts once per game and the stds don't collapse again.

The update run
--------------
--update takes every finished 2026-27 game from boxscores.py (regular season + NBA
Cup, from opening night; NBA.com's game log and season schedule combined, with a
warning wherever they disagree), skips the ones already in applied_games.json, and applies
the rest oldest first. A game is applied only when both teams' four factors are
present; one that can't be downloaded or is incomplete is skipped and NOT recorded,
so a later run tries it again.

After every game both files are saved, crash-safely (write-ahead):
  1. applied_games.json gets the game marked "pending" with a fingerprint of the new priors
  2. priors.json is written (atomic rename)
  3. the "pending" mark is removed
On the next start a leftover pending entry is resolved by comparing fingerprints: if
priors.json already has the game it is kept, otherwise it is dropped and re-applied.
So a crash at any point never loses a finished game or applies one twice.

Prediction log and replay
-------------------------
Before a game is applied it is predicted from the current priors (predict.predict_game,
fixed simulation seed) and appended to prediction_log.csv (see predlog.py); a game
already in the log is never added twice. Each run ends with the log's winner accuracy,
spread MAE and Brier score.

Engines: --engine exact (default) computes each update with the closed-form formula in
bayes_update.py; --engine pymc samples it with PyMC (fixed seed per game, fit_seed).
Noise (params.py): --sigma-update fitted|legacy (default fitted: model_params.json) for the
update and --sigma-sim fitted|legacy (default fitted; legacy = 0.05/0.02/0.05/0.02) for the
simulation behind each logged prediction. Both legacy with every switch off = the pre-Phase-6
model (reproduces log_exact.csv). applied_games.json records the engine and both sigma modes.

Live defaults (params.py): exact engine, both sigmas fitted, --pace on, --sim-corr off,
--home on, --rest on. The previous live model (pace-home) is --rest off; the Phase 6 one is
--sigma-sim legacy --pace off --home off --rest off.

Simulation switches (Phase 7; see simulation.py): --sim-corr on draws the game
factors jointly with the fitted correlation; --pace on uses the teams' pace curves for the
possessions and the corrected points formula. --pace on also UPDATES each team's pace
after every game (bayes_update.update_pace, tau = 2 sigma_pace / M, possessions from NBA.com's
advanced box score), and the season reset sets pace stds to 2 sigma_pace / sqrt(N). It needs
priors with team paces (dynamic_model.py --pace on). applied_games.json records both switches.

Home court (Phase 8, default on): --home on adds model_params.json["home"] to the expected
value of each line in the update and in the simulation, except at neutral sites (flag from
NBA.com's schedule, see boxscores.py). Use priors trained with --home on
(priors_pre_2026-27_pace-home.json). applied_games.json records "home" and "neutral".

Rest (Phase 9, default on): --rest on adds model_params.json["rest"] (fit_rest.py) to the
expected value of each line in the update and in the simulation, from both teams' rest before
the game (back-to-back / 1 day / 2+, from NBA.com's schedule, see rest.py). Use priors trained
with --rest on (priors_pre_2026-27_pace-home-rest.json). applied_games.json records "rest".

Win-probability calibration (Phase 10b, --calib on|off, default off): each logged win % is
Phi(a * Phi^-1(P)) of the simulated P, a = model_params.json["calib"]["temperature"] (fit_calib.py).
It only changes the win %, never the pick or the scores, and never the update, so a replay with
--calib on ends in the same priors. applied_games.json records "calib" for games applied from now on.

Rating reactivity (Phase 10c, --react on|off, default off): the season reset N and the drift M
come from model_params.json["react"] (fit_react.py) instead of N=20, M=30, so the ratings follow
in-season change faster. Unlike --calib it changes the update, so its priors differ from the live
ones (a replay: priors_2026-27_react.json / log_react.csv). --n / --m can't be combined with it.
With --calib on as well, the win % uses the temperature refitted for these settings. applied_games.json
records "react" for games applied from now on.

Betting lines (Phase 10a, --lines on|off, default on; not part of the model): --update also saves
one snapshot of every upcoming game's Vegas lines from The Odds API into data/lines_2026-27.csv
(lines.py: ONE API request per run, appended, never overwritten). Without ODDS_API_KEY, or if the
request fails, it prints a warning and the update goes on as usual.

--replay rebuilds the season into another file: season reset from --from, then every
game in applied_games.json oldest first, each predicted + logged into --log before it is
applied. It writes only --out and --log. Compare logs with code/compare.py.
"""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import time
import warnings
from datetime import date
from pathlib import Path

# bayes_update sets PYTENSOR_FLAGS before it imports pymc
from bayes_update import DEFAULT_ENGINE, ENGINES, FACTORS, update_game, update_pace
import boxscores
import lines as betting_lines
import params
import predlog
import rest as rest_days
from predict import predict_game
from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27.json"  # READ-ONLY, never write
START_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27_pace-home-rest.json"   # live preseason priors, READ-ONLY
PACE_HOME_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27_pace-home.json"   # previous live preseason priors, READ-ONLY
READ_ONLY_PATHS = (FROZEN_PRIORS_PATH, START_PRIORS_PATH, PACE_HOME_PRIORS_PATH)
PRIORS_PATH = REPO_ROOT / "priors.json"
APPLIED_GAMES_PATH = REPO_ROOT / "applied_games.json"
LOG_PATH = REPO_ROOT / "prediction_log.csv"                 # --update's prediction log

SIDES = ("offense", "defense")
STD_SCALE = 0.1
DEFAULT_N = 20
DEFAULT_M = 30
SEASON_GAMES = 1230  # 30 teams x 82 games / 2


def season_settings(react: str = params.DEFAULT_REACT_MODE, n=None, m=None) -> tuple:
    """
    (N, M) of the in-season procedure: react "off" = n / m as given, else DEFAULT_N / DEFAULT_M (20, 30);
    react "on" = model_params.json["react"] (params.react_settings). With react on, n / m must be None:
    the switch sets them (ValueError otherwise).
    """
    if params.is_on(react, "react"):
        if n is not None or m is not None:
            raise ValueError("--react on sets N and M (model_params.json[\"react\"]); don't pass --n / --m with it")
        s = params.react_settings()
        return s["n"], s["m"]
    return (DEFAULT_N if n is None else n), (DEFAULT_M if m is None else m)


def calib_temperature(calib: str, react: str):
    """The calibration temperature for these settings: None when calib is off (validates both switches)."""
    with_react = params.is_on(react, "react")
    if not params.is_on(calib, "calib"):
        return None
    return params.calib_temperature("on" if with_react else "off")


def season_std(n: int) -> float:
    """Prior std worth `n` games of evidence: 0.1 / sqrt(n)."""
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    return STD_SCALE / math.sqrt(n)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    """Write JSON atomically (temp file + rename). Refuses to touch the read-only preseason files."""
    path = Path(path)
    if any(path.resolve() == p.resolve() for p in READ_ONLY_PATHS):
        raise PermissionError(f"{path.name} is a read-only preseason priors file and must never be written")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        file.flush()
        os.fsync(file.fileno())
    # OneDrive can hold a file for a moment while it syncs; retry the rename briefly
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def reset_priors(frozen: dict, n: int = DEFAULT_N) -> dict:
    """
    New priors from the frozen ones: NBA stds reset to season_std(n), everything else copied.
    A team's pace curve, if it has one, keeps its mean and gets std params.pace_season_std(n).
    """
    std = season_std(n)
    missing = [team_id for team_id in NBA_TEAMS if team_id not in frozen]
    if missing:
        raise KeyError(f"NBA teams missing from frozen priors: {missing}")

    priors = copy.deepcopy(frozen)
    pace_std = params.pace_season_std(n) if any("pace" in priors[t] for t in NBA_TEAMS) else None
    for team_id in NBA_TEAMS:
        for side in SIDES:
            for factor in FACTORS:
                priors[team_id][side][factor]["std"] = std
        if "pace" in priors[team_id]:
            priors[team_id]["pace"]["std"] = pace_std
    return priors


def require_pace(priors: dict, what: str) -> None:
    """Raise ValueError unless every NBA team in `priors` has a pace curve (needed for --pace on)."""
    missing = [abbr for team_id, (abbr, _) in NBA_TEAMS.items() if "pace" not in priors.get(team_id, {})]
    if missing:
        raise ValueError(f"--pace on needs team paces, but {what} has none for {', '.join(missing[:5])}"
                         f"{'...' if len(missing) > 5 else ''}; train priors with dynamic_model.py --pace on")


def start_season(n=None, from_path=None, pace: str = params.DEFAULT_PACE_MODE,
                 react: str = params.DEFAULT_REACT_MODE) -> dict:
    """
    Rebuild priors.json from the preseason priors (default START_PRIORS_PATH, the live
    model's) and clear applied_games.json. pace="on" (the default) refuses a source without
    team paces, since --update with --pace on couldn't use the result. n: default
    DEFAULT_N, or model_params.json["react"]["n"] with react="on" (season_settings).
    """
    n, _ = season_settings(react, n=n)
    from_path = Path(from_path or START_PRIORS_PATH)
    source = load_json(from_path)
    if params.is_on(pace, "pace"):
        require_pace(source, from_path.name)
    priors = reset_priors(source, n)
    write_json(PRIORS_PATH, priors)
    write_json(APPLIED_GAMES_PATH, {})
    return priors


def season_tau(m: int) -> float:
    """Per-game drift added to each curve before its fit: 0.1 / M."""
    if m <= 0:
        raise ValueError(f"m must be a positive integer, got {m}")
    return STD_SCALE / m


def apply_game(priors: dict, game: dict, m: int = DEFAULT_M, random_seed=None, engine: str = DEFAULT_ENGINE,
               sigma=None, pace: str = params.DEFAULT_PACE_MODE, home: str = params.DEFAULT_HOME_MODE,
               rest: str = params.DEFAULT_REST_MODE) -> dict:
    """
    Update `priors` in place from one finished game (boxscores.py shape): home offense vs
    away defense, then away offense vs home defense, each widened by tau = 0.1/M first.
    sigma: {factor: game noise}, default the fitted values (params.update_sigma()).
    pace="on": then both teams' pace curves from game["possessions"], widened by
    params.pace_tau(m) first (always the exact formula).
    home="on": the expected value of each line includes the home boost
    (params.home_offsets), unless game["neutral"] is True.
    rest="on": and the rest effect (params.rest_offsets) from game["rest"] = {"home": ..., "away": ...}
    ("b2b", "1", "2+"; see add_rest).
    """
    with_pace = params.is_on(pace, "pace")
    if with_pace and "possessions" not in game:
        raise KeyError(f"game {game.get('game_id')}: pace is on but the game has no possessions")
    if params.is_on(home, "home") and "neutral" not in game:
        raise KeyError(f"game {game.get('game_id')}: home is on but the game has no neutral-site flag")
    with_rest = params.is_on(rest, "rest")
    if with_rest and "rest" not in game:
        raise KeyError(f"game {game.get('game_id')}: rest is on but the game has no rest days")
    offsets = dict(zip(("home", "away"), params.home_offsets(home, game.get("neutral", False))))
    if with_rest:
        shifts = dict(zip(("home", "away"), params.rest_offsets(rest, game["rest"]["home"], game["rest"]["away"])))
        offsets = {side: params.combine_offsets(offsets[side], shifts[side]) for side in offsets}
    tau = season_tau(m)
    for i, (side, other) in enumerate((("home", "away"), ("away", "home"))):
        row = {"team": game[side]["id"], "opponent": game[other]["id"],
               **{f: game[side][f] for f in FACTORS}}
        update_game(priors, row, tau=tau, random_seed=None if random_seed is None else random_seed + i, engine=engine,
                    sigma=sigma, offset=offsets[side])
    if with_pace:
        update_pace(priors, game["home"]["id"], game["away"]["id"], game["possessions"], tau=params.pace_tau(m))
    return priors


def _quiet_pymc():
    """Keep the output readable: PyMC logs every fit; PyTensor warns about loop fusion without a C compiler."""
    logging.getLogger("pymc").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Loop fusion failed")


# ---------------------------------------------------------------------------
# the update run
# ---------------------------------------------------------------------------

def priors_fingerprint(priors: dict) -> str:
    """SHA-256 of the priors' content (key order and whitespace/line endings don't matter)."""
    return hashlib.sha256(json.dumps(priors, sort_keys=True).encode("utf-8")).hexdigest()


def load_applied() -> dict:
    """applied_games.json, after resolving a game left "pending" by a crash mid-save."""
    if not APPLIED_GAMES_PATH.exists():
        raise FileNotFoundError(f"{APPLIED_GAMES_PATH.name} not found; run --start-season first")
    applied = load_json(APPLIED_GAMES_PATH)
    pending = [gid for gid, entry in applied.items() if "pending" in entry]
    if pending:
        current = priors_fingerprint(load_json(PRIORS_PATH))
        for gid in pending:
            if applied[gid]["pending"] == current:
                del applied[gid]["pending"]
                print(f"Recovered {gid}: priors.json already includes it; marked applied.")
            else:
                del applied[gid]
                print(f"Recovered {gid}: priors.json does not include it; it will be applied again.")
        write_json(APPLIED_GAMES_PATH, applied)
    return applied


def record_game(priors: dict, applied: dict, game: dict, engine: str = DEFAULT_ENGINE,
                sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE,
                sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE,
                sim_corr: str = params.DEFAULT_SIM_CORR_MODE, pace: str = params.DEFAULT_PACE_MODE,
                home_mode: str = params.DEFAULT_HOME_MODE, rest_mode: str = params.DEFAULT_REST_MODE,
                calib: str = params.DEFAULT_CALIB_MODE, react: str = params.DEFAULT_REACT_MODE) -> None:
    """Save priors.json and record the game in applied_games.json, crash-safely (see module docstring)."""
    home, away = game["home"], game["away"]
    entry = {"date": game["date"],
             "matchup": f"{away['abbr']} @ {home['abbr']}",
             "score": f"{away['pts']}-{home['pts']}",
             "nba_cup": "IST" in game.get("season_types", []),
             "neutral": bool(game.get("neutral", False)),
             "engine": engine,
             "sigma_update": sigma_update,
             "sigma_sim": sigma_sim,
             "sim_corr": sim_corr,
             "pace": pace,
             "home": home_mode,
             "rest": rest_mode,
             "calib": calib,
             "react": react,
             "pending": priors_fingerprint(priors)}
    applied[game["game_id"]] = entry
    write_json(APPLIED_GAMES_PATH, applied)   # 1. pending, with the fingerprint of the new priors
    write_json(PRIORS_PATH, priors)           # 2. the new priors
    del entry["pending"]
    write_json(APPLIED_GAMES_PATH, applied)   # 3. done


def fit_seed(game_id: str) -> int:
    """Fixed PyMC seed for a game (its gameId as a number), so updates and replays are reproducible."""
    return int(game_id)


def log_prediction(priors: dict, game: dict, log_path, sigma=None, sim_corr: str = params.DEFAULT_SIM_CORR_MODE,
                   pace: str = params.DEFAULT_PACE_MODE, home: str = params.DEFAULT_HOME_MODE,
                   rest: str = params.DEFAULT_REST_MODE, calib: str = params.DEFAULT_CALIB_MODE,
                   react: str = params.DEFAULT_REACT_MODE) -> None:
    """
    Predict `game` from `priors` (fixed simulation seed) and append it to the log unless it's already there.
    react: only picks the calibration temperature (with calib on); the priors already are what they are.
    """
    days = game.get("rest", {})
    prediction = predict_game(game["home"]["id"], game["away"]["id"], priors, seed=predlog.PREDICT_SEED, sigma=sigma,
                              sim_corr=sim_corr, pace=pace, home=home, neutral=game.get("neutral", False),
                              rest=rest, rest_home=days.get("home"), rest_away=days.get("away"), calib=calib,
                              calib_temperature=calib_temperature(calib, react))
    if prediction is None:
        raise KeyError(f"game {game['game_id']}: a team is missing from the priors")
    predlog.append_rows(log_path, [predlog.make_row(game, prediction)])


def add_rest(game: dict, table: dict) -> dict:
    """The game with "rest": {"home": category, "away": category} from a rest.rest_games table."""
    info = table[game["game_id"]]
    return {**game, "rest": {side: info[game[side]["id"]]["rest"] for side in ("home", "away")}}


def save_lines(mode: str = betting_lines.DEFAULT_LINES_MODE):
    """
    lines="on": one snapshot of upcoming games' betting lines (lines.fetch_lines). Never stops the
    update: without ODDS_API_KEY, or when the request fails, it prints a warning and returns None.
    """
    if not params.is_on(mode, "lines"):
        return None
    if betting_lines.api_key() is None:
        print(f"WARNING: --lines on but {betting_lines.KEY_ENV} is not set; betting lines not saved "
              f"(set it, or pass --lines off)\n")
        return None
    try:
        result = betting_lines.fetch_lines()
    except Exception as err:   # lines are a side job: the model update goes on
        print(f"WARNING: betting lines not saved: {betting_lines.redact(err)}\n")
        return None
    print()
    return result


def print_log_metrics(log_path) -> dict:
    m = predlog.metrics(predlog.read_log(log_path))
    print(f"Prediction log {Path(log_path).name}: {predlog.format_metrics(m)}")
    return m


def run_update(through=None, limit=None, m=None, log_path=None, engine: str = DEFAULT_ENGINE,
               sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE,
               sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE,
               sim_corr: str = params.DEFAULT_SIM_CORR_MODE, pace: str = params.DEFAULT_PACE_MODE,
               home: str = params.DEFAULT_HOME_MODE, rest: str = params.DEFAULT_REST_MODE,
               lines: str = betting_lines.DEFAULT_LINES_MODE, calib: str = params.DEFAULT_CALIB_MODE,
               react: str = params.DEFAULT_REACT_MODE) -> dict:
    """
    Apply every finished, not-yet-applied game (oldest first) to priors.json; saves after
    each game. Each game is first predicted from the current priors and logged to
    `log_path` (default prediction_log.csv). `through` = last date to include
    (YYYY-MM-DD), `limit` = max games to apply. pace="on" also needs each game's
    possessions (a game without them is skipped and not recorded, like missing factors).
    home="on" adds the home boost except at neutral sites (flag from NBA.com's schedule).
    rest="on" adds the rest effect (both teams' rest from NBA.com's schedule, rest.py).
    lines="on" also saves a snapshot of upcoming games' betting lines (save_lines; never fails the run).
    m: default DEFAULT_M, or model_params.json["react"]["m"] with react="on" (season_settings).
    Returns {"applied": [ids], "skipped": [(id, reason)], "already": count, "remaining": count,
             "disagreements": [game log vs schedule warnings], "metrics": {...} of the whole log,
             "lines": lines.fetch_lines' result, or None when not saved}.
    """
    params.is_on(lines, "lines")                  # validates the setting before any work
    _, m = season_settings(react, m=m)            # validates the setting (and the react section) before any work
    calib_temperature(calib, react)               # validates the setting (and the temperature) before any work
    log_path = Path(log_path or LOG_PATH)
    update_sigma, sim_sigma = params.update_sigma(sigma_update), params.sim_sigma(sigma_sim)
    params.sim_corr(sim_corr)                     # validates the setting (and the matrix) before any work
    params.home_offsets(home)                     # validates the setting (and the boost) before any work
    params.rest_offsets(rest, "2+", "2+")         # validates the setting (and the effects) before any work
    with_pace, with_rest = params.is_on(pace, "pace"), params.is_on(rest, "rest")
    _quiet_pymc()
    applied = load_applied()
    priors = load_json(PRIORS_PATH)
    if with_pace:
        require_pace(priors, PRIORS_PATH.name)

    print(f"Listing finished {boxscores.SEASON} games on NBA.com (game log + schedule)...")
    disagreements = []
    schedule = boxscores.select_games(boxscores.fetch_finished_games(disagreements=disagreements), through=through)
    todo = [g for g in schedule if g["game_id"] not in applied]
    already = len(schedule) - len(todo)
    # rest from the schedule plus every finished game, so a game the schedule leaves out still counts
    rest_table = (rest_days.rest_games([g["game_id"] for g in todo], extra_games=[rest_days.from_game(g) for g in schedule])
                  if with_rest and todo else {})
    print(f"{len(schedule)} games from {boxscores.OPENING_NIGHT} through {through or 'today'}: "
          f"{already} already applied, {len(todo)} to go"
          + (f" (stopping after {limit})" if limit is not None and limit < len(todo) else "")
          + f". M={m}, tau={season_tau(m):.5f}, engine={engine}, sigma update={sigma_update} sim={sigma_sim}, "
            f"sim_corr={sim_corr}, pace={pace}, home={home}, rest={rest}, calib={calib}, react={react}\n")
    # betting lines for upcoming games (the schedule was just refreshed above, for the matching)
    lines_result = save_lines(lines)

    done, skipped = [], []
    start = time.perf_counter()
    for game in todo:
        if limit is not None and len(done) >= limit:
            break
        label = f"{game['date']}  {game['away']['abbr']} {game['away']['pts']} @ " \
                f"{game['home']['abbr']} {game['home']['pts']}  ({game['game_id']})"
        factors, problem = boxscores.get_game_factors(game["game_id"])
        if problem:
            skipped.append((game["game_id"], problem))
            print(f"  SKIP {label}: {problem}; not recorded")
            continue

        if with_pace:
            possessions, problem = boxscores.get_game_possessions(game["game_id"])
            if problem:
                skipped.append((game["game_id"], problem))
                print(f"  SKIP {label}: {problem}; not recorded")
                continue

        t = time.perf_counter()
        game = boxscores.with_factors(game, factors)
        if with_pace:
            game = boxscores.with_possessions(game, possessions)
        if with_rest:
            game = add_rest(game, rest_table)
        # before applying: the prediction uses pre-game priors
        log_prediction(priors, game, log_path, sim_sigma, sim_corr=sim_corr, pace=pace, home=home, rest=rest,
                       calib=calib, react=react)
        new_priors = apply_game(copy.deepcopy(priors), game, m, random_seed=fit_seed(game["game_id"]), engine=engine,
                                sigma=update_sigma, pace=pace, home=home, rest=rest)
        record_game(new_priors, applied, game, engine, sigma_update, sigma_sim, sim_corr, pace, home, rest, calib,
                    react)
        priors = new_priors
        done.append(game["game_id"])
        print(f"  [{len(done):>4}] {label}  {time.perf_counter() - t:.1f} s")

    remaining = len(todo) - len(done) - len(skipped)
    print(f"\nApplied {len(done)} game(s) in {time.perf_counter() - start:.0f} s; skipped {len(skipped)}; "
          f"{already} were already applied; {remaining} left for a later run; "
          f"{len(disagreements)} game log/schedule disagreement(s).")
    log_metrics = print_log_metrics(log_path)
    return {"applied": done, "skipped": skipped, "already": already, "remaining": remaining,
            "disagreements": disagreements, "metrics": log_metrics, "lines": lines_result}


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _replay_game(game_id: str, entry: dict, factors: dict) -> dict:
    """A boxscores.py-shaped game from its applied_games.json entry and its cached four factors."""
    away_abbr, home_abbr = entry["matchup"].split(" @ ")
    away_pts, home_pts = (int(x) for x in entry["score"].split("-"))
    for side, abbr in (("home", home_abbr), ("away", away_abbr)):
        if NBA_TEAMS.get(factors[side]["id"], ("?",))[0] != abbr:
            raise ValueError(f"game {game_id}: box score {side} team {factors[side]['id']} is not {abbr}")
    home = {"abbr": home_abbr, "pts": home_pts, **factors["home"]}
    away = {"abbr": away_abbr, "pts": away_pts, **factors["away"]}
    return {"game_id": game_id, "date": entry["date"], "home": home, "away": away,
            "winner": home["id"] if home_pts > away_pts else away["id"]}


def replay(from_path, out_path, log_path, n=None, m=None, engine: str = DEFAULT_ENGINE,
           sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE, sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE,
           overwrite: bool = False, sim_corr: str = params.DEFAULT_SIM_CORR_MODE,
           pace: str = params.DEFAULT_PACE_MODE, home: str = params.DEFAULT_HOME_MODE,
           rest: str = params.DEFAULT_REST_MODE, calib: str = params.DEFAULT_CALIB_MODE,
           react: str = params.DEFAULT_REACT_MODE) -> dict:
    """
    Season reset from `from_path` (std 0.1/sqrt(n)), then every game in applied_games.json,
    oldest first (date, gameId), each predicted + logged before it is applied. Writes ONLY
    `out_path` (at the end) and `log_path` (rebuilt from scratch); reads applied_games.json
    and the box-score cache, never writes priors.json or applied_games.json. Existing
    --out/--log files are only replaced with overwrite=True (--overwrite).
    pace="on" needs team paces in `from_path` and each game's possessions (advanced box-score cache).
    home="on" takes each game's neutral-site flag from the cached NBA.com schedule
    (stats/schedules/, downloaded if missing); home="off" never reads it. rest="on" takes
    both teams' rest from the same cached schedule (rest.py).
    n / m: default DEFAULT_N / DEFAULT_M, or model_params.json["react"] with react="on" (season_settings).
    """
    n, m = season_settings(react, n, m)
    from_path, out_path, log_path = Path(from_path), Path(out_path), Path(log_path)
    protected = {p.resolve(): p.name for p in (PRIORS_PATH, APPLIED_GAMES_PATH, FROZEN_PRIORS_PATH, LOG_PATH,
                                               params.PARAMS_PATH, from_path)}
    for flag, path in (("--out", out_path), ("--log", log_path)):
        if path.resolve() in protected:
            raise ValueError(f"{flag} {path} would overwrite {protected[path.resolve()]}; pick another file")
        if path.exists() and not overwrite:
            raise ValueError(f"{flag} {path.name} already exists; pick a new file or pass --overwrite")
    if out_path.resolve() == log_path.resolve():
        raise ValueError("--out and --log must be different files")
    update_sigma, sim_sigma = params.update_sigma(sigma_update), params.sim_sigma(sigma_sim)
    params.sim_corr(sim_corr)                     # validates the setting (and the matrix) before any work
    params.home_offsets(home)                     # validates the setting (and the boost) before any work
    params.rest_offsets(rest, "2+", "2+")         # validates the setting (and the effects) before any work
    calib_temperature(calib, react)               # validates the setting (and the temperature) before any work
    with_pace, with_home = params.is_on(pace, "pace"), params.is_on(home, "home")
    with_rest = params.is_on(rest, "rest")
    start_priors = load_json(from_path)
    if with_pace:
        require_pace(start_priors, from_path.name)

    applied = load_json(APPLIED_GAMES_PATH)
    if any("pending" in entry for entry in applied.values()):
        raise RuntimeError(f"{APPLIED_GAMES_PATH.name} has a pending game; run --update once to resolve it first")
    order = sorted(applied, key=lambda gid: (applied[gid]["date"], gid))
    neutral = boxscores.neutral_games(order) if with_home else {}
    rest_table = rest_days.rest_games(order) if with_rest else {}

    _quiet_pymc()
    priors = reset_priors(start_priors, n)
    if log_path.exists():
        log_path.unlink()   # a replay rebuilds its log from scratch
    print(f"Replaying {len(order)} games from {applied[order[0]]['date']} to {applied[order[-1]]['date']}, "
          f"oldest first, from {from_path.name} (N={n}, M={m}, engine={engine}, "
          f"sigma update={sigma_update} sim={sigma_sim}, sim_corr={sim_corr}, pace={pace}, home={home}, "
          f"rest={rest}, calib={calib}, react={react}) "
          f"-> {out_path.name}, {log_path.name}"
          + (f"; neutral sites: {', '.join(g for g in order if neutral[g]) or 'none'}" if with_home else "")
          + (f"; {sum(any(i['rest'] == 'b2b' for i in rest_table[g].values()) for g in order)} games with a team "
             f"on a back-to-back" if with_rest else "") + "\n")

    start = time.perf_counter()
    for i, game_id in enumerate(order, 1):
        factors, problem = boxscores.get_game_factors(game_id)
        if problem:
            raise RuntimeError(f"can't replay {game_id}: {problem}")
        game = _replay_game(game_id, applied[game_id], factors)
        if with_pace:
            possessions, problem = boxscores.get_game_possessions(game_id)
            if problem:
                raise RuntimeError(f"can't replay {game_id}: {problem}")
            game = boxscores.with_possessions(game, possessions)
        if with_home:
            game["neutral"] = neutral[game_id]
        if with_rest:
            game = add_rest(game, rest_table)
        t = time.perf_counter()
        log_prediction(priors, game, log_path, sim_sigma, sim_corr=sim_corr, pace=pace, home=home, rest=rest,
                       calib=calib, react=react)
        apply_game(priors, game, m, random_seed=fit_seed(game_id), engine=engine, sigma=update_sigma, pace=pace,
                   home=home, rest=rest)
        print(f"  [{i:>4}/{len(order)}] {game['date']}  {applied[game_id]['matchup']}  ({game_id})  "
              f"{time.perf_counter() - t:.1f} s")

    write_json(out_path, priors)
    print(f"\nReplayed {len(order)} games in {time.perf_counter() - start:.0f} s -> {out_path.name}")
    return {"games": order, "metrics": print_log_metrics(log_path)}


def trial(n: int = 5, m=None, engine: str = DEFAULT_ENGINE,
          sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE, pace: str = params.DEFAULT_PACE_MODE,
          home: str = params.DEFAULT_HOME_MODE, rest: str = params.DEFAULT_REST_MODE,
          react: str = params.DEFAULT_REACT_MODE) -> None:
    """
    Dry run: apply the season's first n games to an in-memory copy of priors.json. Never
    writes the priors (box scores and schedules are cached as usual). pace="on" also gets
    each game's possessions (a game without them is left out); home="on" takes the
    neutral-site flags from NBA.com's schedule. m: default DEFAULT_M, or the react one (season_settings).
    """
    from boxscores import SEASON, fetch_games

    _, m = season_settings(react, m=m)

    params.home_offsets(home)                     # validates the setting (and the boost) before any work
    with_pace, with_home = params.is_on(pace, "pace"), params.is_on(home, "home")
    priors = load_json(PRIORS_PATH)
    if with_pace:
        require_pace(priors, PRIORS_PATH.name)

    _quiet_pymc()
    print(f"Fetching the first {n} {SEASON} games from NBA.com...")
    start = time.perf_counter()
    games = fetch_games(limit=n)
    if with_pace:
        kept = []
        for game in games:
            possessions, problem = boxscores.get_game_possessions(game["game_id"])
            if problem:
                print(f"  leaving out {game['game_id']}: {problem}")
                continue
            kept.append(boxscores.with_possessions(game, possessions))
        games = kept
    if with_home and games:
        neutral = boxscores.neutral_games([g["game_id"] for g in games])
        games = [{**g, "neutral": neutral[g["game_id"]]} for g in games]
    if params.is_on(rest, "rest") and games:
        table = rest_days.rest_games([g["game_id"] for g in games])
        games = [add_rest(g, table) for g in games]
    fetch_seconds = (time.perf_counter() - start) / max(len(games), 1)

    print(f"Applying them to an in-memory copy of {PRIORS_PATH.name} "
          f"(M={m}, tau={season_tau(m):.5f}, engine={engine}, pace={pace}, home={home}, rest={rest}). "
          f"Nothing is written.\n")

    seconds, max_shift = [], 0.0
    for game in games:
        home_team, away_team = game["home"], game["away"]   # (`home` is the home-court setting)
        before = copy.deepcopy({t["id"]: priors[t["id"]] for t in (home_team, away_team)})
        start = time.perf_counter()
        apply_game(priors, game, m, engine=engine, sigma=params.update_sigma(sigma_update), pace=pace, home=home,
                   rest=rest)
        seconds.append(time.perf_counter() - start)

        winner = home_team if game["winner"] == home_team["id"] else away_team
        print(f"{game['date']}  {away_team['abbr']} {away_team['pts']} @ {home_team['abbr']} {home_team['pts']}  "
              f"({winner['abbr']} won)  {seconds[-1]:.1f} s")
        for team in (away_team, home_team):
            for side in SIDES:
                cells = []
                for f in FACTORS:
                    old, new = before[team["id"]][side][f]["mean"], priors[team["id"]][side][f]["mean"]
                    max_shift = max(max_shift, abs(new - old))
                    obs = f" obs {team[f]:.3f}" if side == "offense" else ""
                    cells.append(f"{f} {old:.4f}->{new:.4f} ({new - old:+.4f}{obs})")
                print(f"   {team['abbr']:<3} {side[:3]}  " + "   ".join(cells))
        print()

    avg = sum(seconds) / max(len(seconds), 1)
    estimate = SEASON_GAMES * (avg + fetch_seconds)
    print(f"Seconds per game (fit): {avg:.1f} average  [{', '.join(f'{s:.1f}' for s in seconds)}]")
    print(f"Seconds per game (NBA.com download): {fetch_seconds:.1f}")
    print(f"Estimated full season ({SEASON_GAMES} games): {estimate / 60:.0f} min ({estimate / 3600:.1f} h)")
    print(f"Largest mean shift from one game: {max_shift:.4f}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="In-season updates to priors.json.")
    parser.add_argument("--start-season", action="store_true",
                        help=f"reset priors.json from --from (default {START_PRIORS_PATH.name}) and clear "
                             f"{APPLIED_GAMES_PATH.name}")
    parser.add_argument("--n", type=int, default=None,
                        help="games of evidence the preseason prior is worth; std = 0.1/sqrt(N) "
                             f"(default {DEFAULT_N}; --react on sets it)")
    parser.add_argument("--update", action="store_true",
                        help="apply every finished, not-yet-applied game to priors.json (saved after each game)")
    parser.add_argument("--through", metavar="YYYY-MM-DD",
                        help="with --update: only games played on or before this date")
    parser.add_argument("--limit", type=int, metavar="GAMES",
                        help="with --update: apply at most this many games")
    parser.add_argument("--trial", type=int, metavar="GAMES",
                        help="dry run: apply the season's first GAMES games in memory, print the "
                             "changes and timing, write nothing")
    parser.add_argument("--m", type=int, default=None,
                        help=f"per-game drift tau = 0.1/M added before each fit (default {DEFAULT_M}; "
                             "--react on sets it)")
    parser.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE,
                        help=f"how each game's update is computed: exact (closed form) or pymc (MCMC); "
                             f"default {DEFAULT_ENGINE}")
    parser.add_argument("--sigma-update", choices=params.SIGMA_MODES, default=params.DEFAULT_UPDATE_SIGMA_MODE,
                        help="game noise in the Bayesian update: fitted (model_params.json) or legacy (0.05); "
                             f"default {params.DEFAULT_UPDATE_SIGMA_MODE}")
    parser.add_argument("--sigma-sim", choices=params.SIGMA_MODES, default=params.DEFAULT_SIM_SIGMA_MODE,
                        help="game noise in the prediction simulation: fitted (model_params.json) or legacy "
                             f"(0.05/0.02/0.05/0.02); default {params.DEFAULT_SIM_SIGMA_MODE}")
    parser.add_argument("--sim-corr", choices=params.SWITCH_MODES, default=params.DEFAULT_SIM_CORR_MODE,
                        help="draw the simulation's game factors jointly with the fitted correlation "
                             f"(use with --sigma-sim fitted); default {params.DEFAULT_SIM_CORR_MODE}")
    parser.add_argument("--pace", choices=params.SWITCH_MODES, default=params.DEFAULT_PACE_MODE,
                        help="update team paces after each game and simulate possessions from them with the "
                             "corrected points formula; needs priors with team paces; "
                             f"default {params.DEFAULT_PACE_MODE}")
    parser.add_argument("--home", choices=params.SWITCH_MODES, default=params.DEFAULT_HOME_MODE,
                        help="add the fitted home-court boost to the expected value in the update and the "
                             f"simulation (none at neutral sites); default {params.DEFAULT_HOME_MODE}")
    parser.add_argument("--rest", choices=params.SWITCH_MODES, default=params.DEFAULT_REST_MODE,
                        help="add the fitted rest effect (back-to-backs, from NBA.com's schedule) to the expected "
                             f"value in the update and the simulation; default {params.DEFAULT_REST_MODE}")
    parser.add_argument("--calib", choices=params.SWITCH_MODES, default=params.DEFAULT_CALIB_MODE,
                        help="calibrate the logged win %% with the fitted temperature (Phase 10b, "
                             f"model_params.json[\"calib\"]); scores unchanged; default {params.DEFAULT_CALIB_MODE}")
    parser.add_argument("--react", choices=params.SWITCH_MODES, default=params.DEFAULT_REACT_MODE,
                        help="season reset N and drift M from model_params.json[\"react\"] (Phase 10c: ratings "
                             "that follow in-season change faster) instead of N=20, M=30; not with --n / --m; "
                             f"default {params.DEFAULT_REACT_MODE}")
    parser.add_argument("--lines", choices=params.SWITCH_MODES, default=None,
                        help="with --update: also save a snapshot of upcoming games' betting lines from The Odds "
                             f"API (key from {betting_lines.KEY_ENV}; one request; skipped with a warning without "
                             f"a key); default {betting_lines.DEFAULT_LINES_MODE}")
    parser.add_argument("--overwrite", action="store_true", help="with --replay: allow replacing existing --out/--log")
    parser.add_argument("--log", metavar="CSV",
                        help=f"prediction log (--update default: {LOG_PATH.name}; required with --replay)")
    parser.add_argument("--replay", action="store_true",
                        help="season reset from --from + every game in applied_games.json, oldest first, "
                             "into --out; never touches priors.json or applied_games.json")
    parser.add_argument("--from", dest="from_path", metavar="PRIORS",
                        help="with --replay (required) or --start-season (default "
                             f"{START_PRIORS_PATH.name}): preseason priors to reset from")
    parser.add_argument("--out", metavar="PRIORS",
                        help="with --replay: where to write the replayed priors")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.n is not None and args.n <= 0:
        parser.error("--n must be a positive integer")
    if args.m is not None and args.m <= 0:
        parser.error("--m must be a positive integer")
    try:
        n, _ = season_settings(args.react, args.n, args.m)   # validates; each command resolves N and M the same way
    except (ValueError, KeyError) as err:
        parser.error(str(err.args[0]))
    if args.trial is not None and args.trial <= 0:
        parser.error("--trial must be a positive integer")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.through is not None:
        try:
            args.through = date.fromisoformat(args.through).isoformat()
        except ValueError:
            parser.error("--through must be a date like 2026-11-01")
    if (args.through or args.limit) and not args.update:
        parser.error("--through and --limit only work with --update")
    if args.lines is not None and not args.update:
        parser.error("--lines only works with --update")
    args.lines = args.lines or betting_lines.DEFAULT_LINES_MODE
    if args.replay:
        if args.start_season or args.update or args.trial is not None:
            parser.error("--replay runs on its own (not with --start-season, --update or --trial)")
        if not (args.from_path and args.out and args.log):
            parser.error("--replay needs --from, --out and --log")
    elif args.out:
        parser.error("--out only works with --replay")
    elif args.from_path and not args.start_season:
        parser.error("--from only works with --replay or --start-season")
    elif args.overwrite:
        parser.error("--overwrite only works with --replay")
    elif args.log and not args.update:
        parser.error("--log only works with --update or --replay")
    if not (args.start_season or args.update or args.trial is not None or args.replay):
        parser.error("nothing to do: pass --start-season, --update, --replay or --trial")

    if args.replay:
        try:
            replay(args.from_path, args.out, args.log, n=args.n, m=args.m, engine=args.engine,
                   sigma_update=args.sigma_update, sigma_sim=args.sigma_sim, overwrite=args.overwrite,
                   sim_corr=args.sim_corr, pace=args.pace, home=args.home, rest=args.rest, calib=args.calib,
                   react=args.react)
        except ValueError as err:
            parser.error(str(err))
        return

    if args.start_season:
        source = Path(args.from_path or START_PRIORS_PATH)
        try:
            start_season(args.n, source, pace=args.pace, react=args.react)
        except ValueError as err:
            parser.error(str(err))
        print(f"Reset {len(NBA_TEAMS)} NBA teams to std = {season_std(n):.5f} (N={n}, react={args.react}) "
              f"from {source.name} -> {PRIORS_PATH.name}; cleared {APPLIED_GAMES_PATH.name}")
    if args.update:
        if args.pace == "on":
            try:
                require_pace(load_json(PRIORS_PATH), PRIORS_PATH.name)
            except ValueError as err:
                parser.error(str(err))
        run_update(through=args.through, limit=args.limit, m=args.m, log_path=args.log, engine=args.engine,
                   sigma_update=args.sigma_update, sigma_sim=args.sigma_sim, sim_corr=args.sim_corr, pace=args.pace,
                   home=args.home, rest=args.rest, lines=args.lines, calib=args.calib, react=args.react)
    if args.trial is not None:
        trial(args.trial, args.m, engine=args.engine, sigma_update=args.sigma_update, pace=args.pace,
              home=args.home, rest=args.rest, react=args.react)


if __name__ == "__main__":
    main()
