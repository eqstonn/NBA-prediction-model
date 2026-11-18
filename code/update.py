"""
In-season updates to priors.json.

    python code/update.py --start-season          # reset priors for the new season (N=20)
    python code/update.py --start-season --n 10   # looser priors: worth 10 games of evidence
    python code/update.py --update                # apply every finished, not-yet-applied game
    python code/update.py --update --through 2026-11-01 --limit 20   # stop at a date / after 20 games
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
from bayes_update import FACTORS, update_game
import boxscores
from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27.json"  # READ-ONLY, never write
PRIORS_PATH = REPO_ROOT / "priors.json"
APPLIED_GAMES_PATH = REPO_ROOT / "applied_games.json"

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


def apply_game(priors: dict, game: dict, m: int = DEFAULT_M, random_seed=None) -> dict:
    """
    Update `priors` in place from one finished game (boxscores.py shape): home offense vs
    away defense, then away offense vs home defense, each widened by tau = 0.1/M first.
    """
    tau = season_tau(m)
    for i, (side, other) in enumerate((("home", "away"), ("away", "home"))):
        row = {"team": game[side]["id"], "opponent": game[other]["id"],
               **{f: game[side][f] for f in FACTORS}}
        update_game(priors, row, tau=tau, random_seed=None if random_seed is None else random_seed + i)
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


def record_game(priors: dict, applied: dict, game: dict) -> None:
    """Save priors.json and record the game in applied_games.json, crash-safely (see module docstring)."""
    home, away = game["home"], game["away"]
    entry = {"date": game["date"],
             "matchup": f"{away['abbr']} @ {home['abbr']}",
             "score": f"{away['pts']}-{home['pts']}",
             "nba_cup": "IST" in game.get("season_types", []),
             "pending": priors_fingerprint(priors)}
    applied[game["game_id"]] = entry
    write_json(APPLIED_GAMES_PATH, applied)   # 1. pending, with the fingerprint of the new priors
    write_json(PRIORS_PATH, priors)           # 2. the new priors
    del entry["pending"]
    write_json(APPLIED_GAMES_PATH, applied)   # 3. done


def run_update(through=None, limit=None, m: int = DEFAULT_M) -> dict:
    """
    Apply every finished, not-yet-applied game (oldest first) to priors.json; saves after
    each game. `through` = last date to include (YYYY-MM-DD), `limit` = max games to apply.
    Returns {"applied": [ids], "skipped": [(id, reason)], "already": count, "remaining": count,
             "disagreements": [game log vs schedule warnings]}.
    """
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
          + f". M={m}, tau={season_tau(m):.5f}\n")

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
        new_priors = apply_game(copy.deepcopy(priors), boxscores.with_factors(game, factors), m)
        record_game(new_priors, applied, game)
        priors = new_priors
        done.append(game["game_id"])
        print(f"  [{len(done):>4}] {label}  {time.perf_counter() - t:.1f} s")

    remaining = len(todo) - len(done) - len(skipped)
    print(f"\nApplied {len(done)} game(s) in {time.perf_counter() - start:.0f} s; skipped {len(skipped)}; "
          f"{already} were already applied; {remaining} left for a later run; "
          f"{len(disagreements)} game log/schedule disagreement(s).")
    return {"applied": done, "skipped": skipped, "already": already, "remaining": remaining,
            "disagreements": disagreements}


def trial(n: int = 5, m: int = DEFAULT_M) -> None:
    """Dry run: apply the season's first n games to an in-memory copy of priors.json. Writes nothing."""
    from boxscores import SEASON, fetch_games

    _quiet_pymc()
    print(f"Fetching the first {n} {SEASON} games from NBA.com...")
    start = time.perf_counter()
    games = fetch_games(limit=n)
    fetch_seconds = (time.perf_counter() - start) / max(len(games), 1)

    priors = load_json(PRIORS_PATH)
    print(f"Applying them to an in-memory copy of {PRIORS_PATH.name} "
          f"(M={m}, tau={season_tau(m):.5f}). Nothing is written.\n")

    seconds, max_shift = [], 0.0
    for game in games:
        home, away = game["home"], game["away"]
        before = copy.deepcopy({t["id"]: priors[t["id"]] for t in (home, away)})
        start = time.perf_counter()
        apply_game(priors, game, m)
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


def main(argv=None):
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
    if not (args.start_season or args.update or args.trial is not None):
        parser.error("nothing to do: pass --start-season, --update or --trial")

    if args.start_season:
        start_season(args.n)
        print(f"Reset {len(NBA_TEAMS)} NBA teams to std = {season_std(args.n):.5f} (N={args.n}) "
              f"from {FROZEN_PRIORS_PATH.name} -> {PRIORS_PATH.name}; cleared {APPLIED_GAMES_PATH.name}")
    if args.update:
        run_update(through=args.through, limit=args.limit, m=args.m)
    if args.trial is not None:
        trial(args.trial, args.m)


if __name__ == "__main__":
    main()
