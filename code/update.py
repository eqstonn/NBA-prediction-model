"""
In-season updates to priors.json.

    python code/update.py --start-season          # reset priors for the new season (N=20)
    python code/update.py --start-season --n 10   # looser priors: worth 10 games of evidence
    python code/update.py --update                # apply every finished, not-yet-applied game
    python code/update.py --update --through 2026-11-01 --limit 20   # stop at a date / after 20 games
    python code/update.py --replay --from priors_pre_2026-27.json --out priors_x.json --log log_x.csv
                                                  # rebuild the season into another file (model experiments)
    python code/update.py --trial 5               # dry run: first 5 games of the season, writes nothing
    python code/update.py --trial 5 --m 20        # same with more drift per game (tau = 0.1/20)

Season reset
------------
priors_pre_2026-27.json is the frozen snapshot trained on every game before the
2026-27 season. It is only ever READ here. A reset rebuilds priors.json from it:

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
update and --sigma-sim fitted|legacy (default legacy: 0.05/0.02/0.05/0.02) for the simulation
behind each logged prediction. Both legacy = the pre-Phase-6 model (reproduces log_exact.csv).
applied_games.json records the engine and both sigma modes that applied each game.

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
from bayes_update import DEFAULT_ENGINE, ENGINES, FACTORS, update_game
import boxscores
import params
import predlog
from predict import predict_game
from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27.json"  # READ-ONLY, never write
PRIORS_PATH = REPO_ROOT / "priors.json"
APPLIED_GAMES_PATH = REPO_ROOT / "applied_games.json"
LOG_PATH = REPO_ROOT / "prediction_log.csv"                 # --update's prediction log

SIDES = ("offense", "defense")
STD_SCALE = 0.1
DEFAULT_N = 20
DEFAULT_M = 30
SEASON_GAMES = 1230  # 30 teams x 82 games / 2


def season_std(n: int) -> float:
    """Prior std worth `n` games of evidence: 0.1 / sqrt(n)."""
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    return STD_SCALE / math.sqrt(n)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    """Write JSON atomically (temp file + rename). Refuses to touch the frozen snapshot."""
    path = Path(path)
    if path.resolve() == FROZEN_PRIORS_PATH.resolve():
        raise PermissionError(f"{path.name} is the frozen preseason snapshot and must never be written")
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
    """New priors from the frozen ones: NBA stds reset to season_std(n), everything else copied."""
    std = season_std(n)
    missing = [team_id for team_id in NBA_TEAMS if team_id not in frozen]
    if missing:
        raise KeyError(f"NBA teams missing from frozen priors: {missing}")

    priors = copy.deepcopy(frozen)
    for team_id in NBA_TEAMS:
        for side in SIDES:
            for factor in FACTORS:
                priors[team_id][side][factor]["std"] = std
    return priors


def start_season(n: int = DEFAULT_N) -> dict:
    """Rebuild priors.json from the frozen snapshot and clear applied_games.json."""
    priors = reset_priors(load_json(FROZEN_PRIORS_PATH), n)
    write_json(PRIORS_PATH, priors)
    write_json(APPLIED_GAMES_PATH, {})
    return priors


def season_tau(m: int) -> float:
    """Per-game drift added to each curve before its fit: 0.1 / M."""
    if m <= 0:
        raise ValueError(f"m must be a positive integer, got {m}")
    return STD_SCALE / m


def apply_game(priors: dict, game: dict, m: int = DEFAULT_M, random_seed=None, engine: str = DEFAULT_ENGINE,
               sigma=None) -> dict:
    """
    Update `priors` in place from one finished game (boxscores.py shape): home offense vs
    away defense, then away offense vs home defense, each widened by tau = 0.1/M first.
    sigma: {factor: game noise}, default the fitted values (params.update_sigma()).
    """
    tau = season_tau(m)
    for i, (side, other) in enumerate((("home", "away"), ("away", "home"))):
        row = {"team": game[side]["id"], "opponent": game[other]["id"],
               **{f: game[side][f] for f in FACTORS}}
        update_game(priors, row, tau=tau, random_seed=None if random_seed is None else random_seed + i, engine=engine,
                    sigma=sigma)
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
                sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE) -> None:
    """Save priors.json and record the game in applied_games.json, crash-safely (see module docstring)."""
    home, away = game["home"], game["away"]
    entry = {"date": game["date"],
             "matchup": f"{away['abbr']} @ {home['abbr']}",
             "score": f"{away['pts']}-{home['pts']}",
             "nba_cup": "IST" in game.get("season_types", []),
             "engine": engine,
             "sigma_update": sigma_update,
             "sigma_sim": sigma_sim,
             "pending": priors_fingerprint(priors)}
    applied[game["game_id"]] = entry
    write_json(APPLIED_GAMES_PATH, applied)   # 1. pending, with the fingerprint of the new priors
    write_json(PRIORS_PATH, priors)           # 2. the new priors
    del entry["pending"]
    write_json(APPLIED_GAMES_PATH, applied)   # 3. done


def fit_seed(game_id: str) -> int:
    """Fixed PyMC seed for a game (its gameId as a number), so updates and replays are reproducible."""
    return int(game_id)


def log_prediction(priors: dict, game: dict, log_path, sigma=None) -> None:
    """Predict `game` from `priors` (fixed simulation seed) and append it to the log unless it's already there."""
    prediction = predict_game(game["home"]["id"], game["away"]["id"], priors, seed=predlog.PREDICT_SEED, sigma=sigma)
    if prediction is None:
        raise KeyError(f"game {game['game_id']}: a team is missing from the priors")
    predlog.append_rows(log_path, [predlog.make_row(game, prediction)])


def print_log_metrics(log_path) -> dict:
    m = predlog.metrics(predlog.read_log(log_path))
    print(f"Prediction log {Path(log_path).name}: {predlog.format_metrics(m)}")
    return m


def run_update(through=None, limit=None, m: int = DEFAULT_M, log_path=None, engine: str = DEFAULT_ENGINE,
               sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE,
               sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE) -> dict:
    """
    Apply every finished, not-yet-applied game (oldest first) to priors.json; saves after
    each game. Each game is first predicted from the current priors and logged to
    `log_path` (default prediction_log.csv). `through` = last date to include
    (YYYY-MM-DD), `limit` = max games to apply.
    Returns {"applied": [ids], "skipped": [(id, reason)], "already": count, "remaining": count,
             "disagreements": [game log vs schedule warnings], "metrics": {...} of the whole log}.
    """
    log_path = Path(log_path or LOG_PATH)
    update_sigma, sim_sigma = params.update_sigma(sigma_update), params.sim_sigma(sigma_sim)
    _quiet_pymc()
    applied = load_applied()
    priors = load_json(PRIORS_PATH)

    print(f"Listing finished {boxscores.SEASON} games on NBA.com (game log + schedule)...")
    disagreements = []
    schedule = boxscores.select_games(boxscores.fetch_finished_games(disagreements=disagreements), through=through)
    todo = [g for g in schedule if g["game_id"] not in applied]
    already = len(schedule) - len(todo)
    print(f"{len(schedule)} games from {boxscores.OPENING_NIGHT} through {through or 'today'}: "
          f"{already} already applied, {len(todo)} to go"
          + (f" (stopping after {limit})" if limit is not None and limit < len(todo) else "")
          + f". M={m}, tau={season_tau(m):.5f}, engine={engine}, sigma update={sigma_update} sim={sigma_sim}\n")

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

        t = time.perf_counter()
        game = boxscores.with_factors(game, factors)
        log_prediction(priors, game, log_path, sim_sigma)   # before applying: the prediction uses pre-game priors
        new_priors = apply_game(copy.deepcopy(priors), game, m, random_seed=fit_seed(game["game_id"]), engine=engine,
                                sigma=update_sigma)
        record_game(new_priors, applied, game, engine, sigma_update, sigma_sim)
        priors = new_priors
        done.append(game["game_id"])
        print(f"  [{len(done):>4}] {label}  {time.perf_counter() - t:.1f} s")

    remaining = len(todo) - len(done) - len(skipped)
    print(f"\nApplied {len(done)} game(s) in {time.perf_counter() - start:.0f} s; skipped {len(skipped)}; "
          f"{already} were already applied; {remaining} left for a later run; "
          f"{len(disagreements)} game log/schedule disagreement(s).")
    log_metrics = print_log_metrics(log_path)
    return {"applied": done, "skipped": skipped, "already": already, "remaining": remaining,
            "disagreements": disagreements, "metrics": log_metrics}


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


def replay(from_path, out_path, log_path, n: int = DEFAULT_N, m: int = DEFAULT_M, engine: str = DEFAULT_ENGINE,
           sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE, sigma_sim: str = params.DEFAULT_SIM_SIGMA_MODE,
           overwrite: bool = False) -> dict:
    """
    Season reset from `from_path` (std 0.1/sqrt(n)), then every game in applied_games.json,
    oldest first (date, gameId), each predicted + logged before it is applied. Writes ONLY
    `out_path` (at the end) and `log_path` (rebuilt from scratch); reads applied_games.json
    and the box-score cache, never writes priors.json or applied_games.json. Existing
    --out/--log files are only replaced with overwrite=True (--overwrite).
    """
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

    applied = load_json(APPLIED_GAMES_PATH)
    if any("pending" in entry for entry in applied.values()):
        raise RuntimeError(f"{APPLIED_GAMES_PATH.name} has a pending game; run --update once to resolve it first")
    order = sorted(applied, key=lambda gid: (applied[gid]["date"], gid))

    _quiet_pymc()
    priors = reset_priors(load_json(from_path), n)
    if log_path.exists():
        log_path.unlink()   # a replay rebuilds its log from scratch
    print(f"Replaying {len(order)} games from {applied[order[0]]['date']} to {applied[order[-1]]['date']}, "
          f"oldest first, from {from_path.name} (N={n}, M={m}, engine={engine}, "
          f"sigma update={sigma_update} sim={sigma_sim}) "
          f"-> {out_path.name}, {log_path.name}\n")

    start = time.perf_counter()
    for i, game_id in enumerate(order, 1):
        factors, problem = boxscores.get_game_factors(game_id)
        if problem:
            raise RuntimeError(f"can't replay {game_id}: {problem}")
        game = _replay_game(game_id, applied[game_id], factors)
        t = time.perf_counter()
        log_prediction(priors, game, log_path, sim_sigma)
        apply_game(priors, game, m, random_seed=fit_seed(game_id), engine=engine, sigma=update_sigma)
        print(f"  [{i:>4}/{len(order)}] {game['date']}  {applied[game_id]['matchup']}  ({game_id})  "
              f"{time.perf_counter() - t:.1f} s")

    write_json(out_path, priors)
    print(f"\nReplayed {len(order)} games in {time.perf_counter() - start:.0f} s -> {out_path.name}")
    return {"games": order, "metrics": print_log_metrics(log_path)}


def trial(n: int = 5, m: int = DEFAULT_M, engine: str = DEFAULT_ENGINE,
          sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE) -> None:
    """Dry run: apply the season's first n games to an in-memory copy of priors.json. Writes nothing."""
    from boxscores import SEASON, fetch_games

    _quiet_pymc()
    print(f"Fetching the first {n} {SEASON} games from NBA.com...")
    start = time.perf_counter()
    games = fetch_games(limit=n)
    fetch_seconds = (time.perf_counter() - start) / max(len(games), 1)

    priors = load_json(PRIORS_PATH)
    print(f"Applying them to an in-memory copy of {PRIORS_PATH.name} "
          f"(M={m}, tau={season_tau(m):.5f}, engine={engine}). Nothing is written.\n")

    seconds, max_shift = [], 0.0
    for game in games:
        home, away = game["home"], game["away"]
        before = copy.deepcopy({t["id"]: priors[t["id"]] for t in (home, away)})
        start = time.perf_counter()
        apply_game(priors, game, m, engine=engine, sigma=params.update_sigma(sigma_update))
        seconds.append(time.perf_counter() - start)

        winner = home if game["winner"] == home["id"] else away
        print(f"{game['date']}  {away['abbr']} {away['pts']} @ {home['abbr']} {home['pts']}  "
              f"({winner['abbr']} won)  {seconds[-1]:.1f} s")
        for team in (away, home):
            for side in SIDES:
                cells = []
                for f in FACTORS:
                    old, new = before[team["id"]][side][f]["mean"], priors[team["id"]][side][f]["mean"]
                    max_shift = max(max_shift, abs(new - old))
                    obs = f" obs {team[f]:.3f}" if side == "offense" else ""
                    cells.append(f"{f} {old:.4f}->{new:.4f} ({new - old:+.4f}{obs})")
                print(f"   {team['abbr']:<3} {side[:3]}  " + "   ".join(cells))
        print()

    avg = sum(seconds) / len(seconds)
    estimate = SEASON_GAMES * (avg + fetch_seconds)
    print(f"Seconds per game (fit): {avg:.1f} average  [{', '.join(f'{s:.1f}' for s in seconds)}]")
    print(f"Seconds per game (NBA.com download): {fetch_seconds:.1f}")
    print(f"Estimated full season ({SEASON_GAMES} games): {estimate / 60:.0f} min ({estimate / 3600:.1f} h)")
    print(f"Largest mean shift from one game: {max_shift:.4f}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="In-season updates to priors.json.")
    parser.add_argument("--start-season", action="store_true",
                        help=f"reset priors.json from {FROZEN_PRIORS_PATH.name} and clear "
                             f"{APPLIED_GAMES_PATH.name}")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="games of evidence the preseason prior is worth; std = 0.1/sqrt(N) "
                             f"(default {DEFAULT_N})")
    parser.add_argument("--update", action="store_true",
                        help="apply every finished, not-yet-applied game to priors.json (saved after each game)")
    parser.add_argument("--through", metavar="YYYY-MM-DD",
                        help="with --update: only games played on or before this date")
    parser.add_argument("--limit", type=int, metavar="GAMES",
                        help="with --update: apply at most this many games")
    parser.add_argument("--trial", type=int, metavar="GAMES",
                        help="dry run: apply the season's first GAMES games in memory, print the "
                             "changes and timing, write nothing")
    parser.add_argument("--m", type=int, default=DEFAULT_M,
                        help=f"per-game drift tau = 0.1/M added before each fit (default {DEFAULT_M})")
    parser.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE,
                        help=f"how each game's update is computed: exact (closed form) or pymc (MCMC); "
                             f"default {DEFAULT_ENGINE}")
    parser.add_argument("--sigma-update", choices=params.SIGMA_MODES, default=params.DEFAULT_UPDATE_SIGMA_MODE,
                        help="game noise in the Bayesian update: fitted (model_params.json) or legacy (0.05); "
                             f"default {params.DEFAULT_UPDATE_SIGMA_MODE}")
    parser.add_argument("--sigma-sim", choices=params.SIGMA_MODES, default=params.DEFAULT_SIM_SIGMA_MODE,
                        help="game noise in the prediction simulation: fitted (model_params.json) or legacy "
                             f"(0.05/0.02/0.05/0.02); default {params.DEFAULT_SIM_SIGMA_MODE}")
    parser.add_argument("--overwrite", action="store_true", help="with --replay: allow replacing existing --out/--log")
    parser.add_argument("--log", metavar="CSV",
                        help=f"prediction log (--update default: {LOG_PATH.name}; required with --replay)")
    parser.add_argument("--replay", action="store_true",
                        help="season reset from --from + every game in applied_games.json, oldest first, "
                             "into --out; never touches priors.json or applied_games.json")
    parser.add_argument("--from", dest="from_path", metavar="PRIORS",
                        help="with --replay: preseason priors to reset from (e.g. priors_pre_2026-27.json)")
    parser.add_argument("--out", metavar="PRIORS",
                        help="with --replay: where to write the replayed priors")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.n <= 0:
        parser.error("--n must be a positive integer")
    if args.m <= 0:
        parser.error("--m must be a positive integer")
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
    if args.replay:
        if args.start_season or args.update or args.trial is not None:
            parser.error("--replay runs on its own (not with --start-season, --update or --trial)")
        if not (args.from_path and args.out and args.log):
            parser.error("--replay needs --from, --out and --log")
    elif args.from_path or args.out:
        parser.error("--from and --out only work with --replay")
    elif args.overwrite:
        parser.error("--overwrite only works with --replay")
    elif args.log and not args.update:
        parser.error("--log only works with --update or --replay")
    if not (args.start_season or args.update or args.trial is not None or args.replay):
        parser.error("nothing to do: pass --start-season, --update, --replay or --trial")

    if args.replay:
        try:
            replay(args.from_path, args.out, args.log, n=args.n, m=args.m, engine=args.engine,
                   sigma_update=args.sigma_update, sigma_sim=args.sigma_sim, overwrite=args.overwrite)
        except ValueError as err:
            parser.error(str(err))
        return

    if args.start_season:
        start_season(args.n)
        print(f"Reset {len(NBA_TEAMS)} NBA teams to std = {season_std(args.n):.5f} (N={args.n}) "
              f"from {FROZEN_PRIORS_PATH.name} -> {PRIORS_PATH.name}; cleared {APPLIED_GAMES_PATH.name}")
    if args.update:
        run_update(through=args.through, limit=args.limit, m=args.m, log_path=args.log, engine=args.engine,
                   sigma_update=args.sigma_update, sigma_sim=args.sigma_sim)
    if args.trial is not None:
        trial(args.trial, args.m, engine=args.engine, sigma_update=args.sigma_update)


if __name__ == "__main__":
    main()
