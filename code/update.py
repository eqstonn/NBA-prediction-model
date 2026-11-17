"""
In-season updates to priors.json.

    python code/update.py --start-season          # reset priors for the new season (N=20)
    python code/update.py --start-season --n 10   # looser priors: worth 10 games of evidence

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
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path

from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_PRIORS_PATH = REPO_ROOT / "priors_pre_2026-27.json"  # READ-ONLY, never write
PRIORS_PATH = REPO_ROOT / "priors.json"
APPLIED_GAMES_PATH = REPO_ROOT / "applied_games.json"

SIDES = ("offense", "defense")
FACTORS = ("eFG", "TOV", "ORB", "FTR")
STD_SCALE = 0.1
DEFAULT_N = 20


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
    os.replace(tmp, path)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="In-season updates to priors.json.")
    parser.add_argument("--start-season", action="store_true",
                        help=f"reset priors.json from {FROZEN_PRIORS_PATH.name} and clear "
                             f"{APPLIED_GAMES_PATH.name}")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="games of evidence the preseason prior is worth; std = 0.1/sqrt(N) "
                             f"(default {DEFAULT_N})")
    args = parser.parse_args(argv)

    if args.n <= 0:
        parser.error("--n must be a positive integer")
    if not args.start_season:
        parser.error("nothing to do: pass --start-season")

    start_season(args.n)
    print(f"Reset {len(NBA_TEAMS)} NBA teams to std = {season_std(args.n):.5f} (N={args.n}) "
          f"from {FROZEN_PRIORS_PATH.name} -> {PRIORS_PATH.name}; cleared {APPLIED_GAMES_PATH.name}")


if __name__ == "__main__":
    main()
