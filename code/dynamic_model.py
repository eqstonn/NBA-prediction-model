"""
TRAINING: run the Kaggle games through the Bayesian update, one row at a time, into a
new priors file.

    python code/dynamic_model.py --out priors_pre_2026-27_exact.json --before 2026-10-03
    python code/dynamic_model.py --out x.json --before 2026-10-03 --years 5 --engine pymc
    python code/dynamic_model.py --out x.json --before 2026-10-03 --all-history

Use --before 2026-10-03 (the first day of the 2026-27 preseason): the Kaggle file labels
only this season's preseason games "Preseason" (older preseasons have no gameType and are
dropped), so a later cutoff would train on 2026-27 preseason games. 2026-10-03 with the
default 3 years reproduces the original training's games and its 37 teams.

It reads stats/TeamStatisticsExtended.csv or stats/NBA-DataSet/TeamStatisticsExtended.csv
(Kaggle, not in the repo; stats/ is gitignored), keeps games before
--before and within --years of it (default 3, like the original training), starts every
team from the flat league-average curves (state_manager.default_priors), applies every
row oldest first with tau=0 (one row = one team's offense vs the opponent's defense),
and writes --out. It never writes priors.json or the frozen priors_pre_2026-27.json.

--pace on (Phase 7) also trains each team's pace curve, priors[team]["pace"] = {"mean",
"std"} in possessions per game: every team starts at model_params.json["pace"]
(start_mean, start_std), then every game's possessions (the average of the two teams'
"possessions" columns) updates both teams' paces with bayes_update.update_pace (tau=0).
The factor training is unchanged, so the offense/defense curves are identical to --pace off.
"""
import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

# bayes_update sets PYTENSOR_FLAGS before pymc can be imported
from bayes_update import DEFAULT_ENGINE, ENGINES, update_game, update_pace
from data_prep import create_data_frame, clean_data_frame, get_four_factors
import params
from state_manager import default_priors

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_LOCATIONS = [REPO_ROOT / "stats" / "TeamStatisticsExtended.csv",
                 REPO_ROOT / "stats" / "NBA-DataSet" / "TeamStatisticsExtended.csv"]   # loose, or the unzipped Kaggle folder
DEFAULT_CSV = next((p for p in CSV_LOCATIONS if p.exists()), CSV_LOCATIONS[0])
# never written, even with --overwrite
PROTECTED = [REPO_ROOT / name for name in
             ("priors_pre_2026-27.json", "priors.json", "applied_games.json", "priors_2026-27_base.json",
              "model_params.json")]
TRAINING_CUTOFF = "2026-10-03"   # first day of the 2026-27 preseason (see the module docstring)
TRAINING_TAU = 0.0
CHECKPOINT_EVERY = 50   # rows, pymc engine only


def team_key(value) -> str:
    return str(int(value))


def row_to_game(row) -> dict:
    """One Kaggle row (one team's offense vs the opponent's defense) as an update_game input."""
    return {
        "team": team_key(row["teamId"]),
        "opponent": team_key(row["opponentTeamId"]),
        "eFG": row["effectiveFieldGoalPercentage"],
        "TOV": row["teamTurnoverPercentage"],
        "ORB": row["offensiveReboundPercentage"],
        "FTR": row["freeThrowAttemptRate"],
    }


def write_priors(path, priors):
    path = Path(path)
    if any(path.resolve() == p.resolve() for p in PROTECTED):
        raise PermissionError(f"refusing to overwrite {path.name}; train into a new file")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(priors, file, indent=4)
    os.replace(tmp, path)


def train(four_factors, priors: dict, engine: str = DEFAULT_ENGINE, checkpoint=None, sigma=None) -> dict:
    """Apply every row of `four_factors` (oldest first) to `priors` in place; returns `priors`."""
    for count, (_, row) in enumerate(four_factors.iterrows(), 1):
        update_game(priors, row_to_game(row), tau=TRAINING_TAU, engine=engine, sigma=sigma)
        if checkpoint is not None and count % CHECKPOINT_EVERY == 0:
            checkpoint(priors, count)
    return priors


def game_possessions(cleaned) -> list:
    """
    One entry per game with both teams' possessions, oldest first (order of first appearance):
    {"game_id", "team", "opponent", "possessions"}, possessions = the average of the two
    teams' counts (NBA.com's differ by ~1). Empty if the data has no "possessions" column.
    """
    if "possessions" not in cleaned.columns:
        return []
    rows = {}
    for game_id, team, opponent, poss in zip(cleaned["gameId"], cleaned["teamId"], cleaned["opponentTeamId"],
                                             cleaned["possessions"]):
        rows.setdefault(game_id, []).append((team_key(team), team_key(opponent), poss))
    games = []
    for game_id, pair in rows.items():
        if len(pair) != 2 or any(p != p or p is None for _, _, p in pair):   # p != p: NaN
            continue
        (team, opponent, p1), (_, _, p2) = pair
        games.append({"game_id": str(game_id), "team": team, "opponent": opponent, "possessions": (p1 + p2) / 2})
    return games


def load_training_rows(csv_path, before, years=3) -> dict:
    """
    The training rows (oldest first) and their teams: {"rows", "team_ids", "games", "first",
    "last", "possessions" (game_possessions), "cleaned" (every column of the window)}.
    """
    cleaned = clean_data_frame(create_data_frame(csv_path), before=before, years=years)
    four_factors = get_four_factors(cleaned)
    if four_factors.empty:
        raise ValueError(f"no games in the window before {before}")
    team_ids = sorted({team_key(t) for t in four_factors["teamId"]} | {team_key(t) for t in four_factors["opponentTeamId"]})
    return {"rows": four_factors, "team_ids": team_ids, "games": four_factors["gameId"].nunique(),
            "first": str(cleaned["gameDateTimeEst"].min())[:10], "last": str(cleaned["gameDateTimeEst"].max())[:10],
            "possessions": game_possessions(cleaned), "cleaned": cleaned}


def start_pace(priors: dict, team_ids, start_mean: float, start_std: float) -> dict:
    """Give every team the flat starting pace curve; returns `priors`."""
    for team in team_ids:
        priors[team]["pace"] = {"mean": start_mean, "std": start_std}
    return priors


def train_pace(possession_games: list, priors: dict, sigma: float) -> dict:
    """Apply every game's possessions (oldest first) to both teams' pace curves, tau=0; returns `priors`."""
    for g in possession_games:
        update_pace(priors, g["team"], g["opponent"], g["possessions"], tau=TRAINING_TAU, sigma=sigma)
    return priors


def check_output(path, overwrite=False):
    path = Path(path)
    if any(path.resolve() == p.resolve() for p in PROTECTED):
        raise PermissionError(f"refusing to overwrite {path.name}; train into a new file")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path.name} already exists; pick a new file or pass --overwrite")


def run_training(csv_path, out_path, before, years=3, engine: str = DEFAULT_ENGINE,
                 sigma_update: str = params.DEFAULT_UPDATE_SIGMA_MODE, overwrite: bool = False,
                 pace: str = params.DEFAULT_PACE_MODE) -> dict:
    """
    Train from flat defaults on the games before `before` (and within `years` of it) into
    `out_path`. pace="on" also trains every team's pace curve (see the module docstring).
    """
    out_path = Path(out_path)
    check_output(out_path, overwrite)
    sigma = params.update_sigma(sigma_update)
    with_pace = params.is_on(pace, "pace")
    pace_params = params.pace_params() if with_pace else None

    start = time.perf_counter()
    data = load_training_rows(csv_path, before, years)
    four_factors, team_ids, first, last = data["rows"], data["team_ids"], data["first"], data["last"]
    priors = default_priors(team_ids)
    loaded = time.perf_counter()

    def checkpoint(p, count):
        write_priors(out_path, p)
        print(f"Checkpoint: saved row {count}...")

    if with_pace and not data["possessions"]:
        raise KeyError(f"--pace on needs a 'possessions' column in {Path(csv_path).name}")
    train(four_factors, priors, engine, checkpoint if engine == "pymc" else None, sigma=sigma)
    if with_pace:
        start_pace(priors, team_ids, pace_params["start_mean"], pace_params["start_std"])
        train_pace(data["possessions"], priors, pace_params["sigma"])
    write_priors(out_path, priors)
    done = time.perf_counter()

    summary = {"rows": len(four_factors), "games": data["games"], "teams": len(team_ids),
               "first": first, "last": last, "load_seconds": loaded - start, "train_seconds": done - loaded,
               "pace_games": len(data["possessions"]) if with_pace else 0}
    print(f"Trained {summary['rows']} rows ({summary['games']} games, {summary['teams']} teams, "
          f"{first} to {last}) with the {engine} engine, {sigma_update} sigma "
          f"({', '.join(f'{f} {v:.4f}' for f, v in sigma.items())}) in {summary['train_seconds']:.1f} s "
          f"(+ {summary['load_seconds']:.1f} s loading the CSV) -> {out_path.name}")
    if with_pace:
        print(f"  + team paces from {summary['pace_games']} games (start {pace_params['start_mean']:.2f} "
              f"+- {pace_params['start_std']:.1f}, game noise {pace_params['sigma']:.2f} possessions)")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train priors on the Kaggle history into a new file.")
    parser.add_argument("--out", required=True, help="new priors file to write")
    parser.add_argument("--before", required=True, metavar="YYYY-MM-DD",
                        help=f"only games before this date (use {TRAINING_CUTOFF}, the start of the 2026-27 preseason)")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--all-history", action="store_true", help="use every game before --before")
    parser.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE, help=f"default {DEFAULT_ENGINE}")
    parser.add_argument("--sigma-update", choices=params.SIGMA_MODES, default=params.DEFAULT_UPDATE_SIGMA_MODE,
                        help="game noise in the update: fitted (model_params.json) or legacy (0.05 for every "
                             f"factor); default {params.DEFAULT_UPDATE_SIGMA_MODE}. (Training never simulates.)")
    parser.add_argument("--pace", choices=params.SWITCH_MODES, default=params.DEFAULT_PACE_MODE,
                        help="also train each team's pace (possessions per game); "
                             f"default {params.DEFAULT_PACE_MODE}")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error(f"--before must be a date like {TRAINING_CUTOFF}")
    if args.years <= 0:
        parser.error("--years must be a positive integer")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")
    try:
        run_training(args.csv, args.out, before, years=None if args.all_history else args.years, engine=args.engine,
                     sigma_update=args.sigma_update, overwrite=args.overwrite, pace=args.pace)
    except (PermissionError, FileExistsError, KeyError) as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
