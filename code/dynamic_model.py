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
"""
import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

# bayes_update sets PYTENSOR_FLAGS before pymc can be imported
from bayes_update import DEFAULT_ENGINE, ENGINES, update_game
from data_prep import create_data_frame, clean_data_frame, get_four_factors
from state_manager import default_priors

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_LOCATIONS = [REPO_ROOT / "stats" / "TeamStatisticsExtended.csv",
                 REPO_ROOT / "stats" / "NBA-DataSet" / "TeamStatisticsExtended.csv"]   # loose, or the unzipped Kaggle folder
DEFAULT_CSV = next((p for p in CSV_LOCATIONS if p.exists()), CSV_LOCATIONS[0])
PROTECTED = [REPO_ROOT / name for name in
             ("priors_pre_2026-27.json", "priors.json", "applied_games.json", "priors_2026-27_base.json")]
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


def train(four_factors, priors: dict, engine: str = DEFAULT_ENGINE, checkpoint=None) -> dict:
    """Apply every row of `four_factors` (oldest first) to `priors` in place; returns `priors`."""
    for count, (_, row) in enumerate(four_factors.iterrows(), 1):
        update_game(priors, row_to_game(row), tau=TRAINING_TAU, engine=engine)
        if checkpoint is not None and count % CHECKPOINT_EVERY == 0:
            checkpoint(priors, count)
    return priors


def run_training(csv_path, out_path, before, years=3, engine: str = DEFAULT_ENGINE) -> dict:
    """Train from flat defaults on the games before `before` (and within `years` of it) into `out_path`."""
    out_path = Path(out_path)
    if any(out_path.resolve() == p.resolve() for p in PROTECTED):
        raise PermissionError(f"refusing to overwrite {out_path.name}; train into a new file")

    start = time.perf_counter()
    cleaned = clean_data_frame(create_data_frame(csv_path), before=before, years=years)
    four_factors = get_four_factors(cleaned)
    if four_factors.empty:
        raise ValueError(f"no games in the window before {before}")
    first, last = str(cleaned["gameDateTimeEst"].min())[:10], str(cleaned["gameDateTimeEst"].max())[:10]
    team_ids = sorted({team_key(t) for t in four_factors["teamId"]} | {team_key(t) for t in four_factors["opponentTeamId"]})
    priors = default_priors(team_ids)
    loaded = time.perf_counter()

    def checkpoint(p, count):
        write_priors(out_path, p)
        print(f"Checkpoint: saved row {count}...")

    train(four_factors, priors, engine, checkpoint if engine == "pymc" else None)
    write_priors(out_path, priors)
    done = time.perf_counter()

    summary ={"rows": len(four_factors), "games": four_factors["gameId"].nunique(), "teams": len(team_ids),
               "first": first, "last": last, "load_seconds": loaded - start, "train_seconds": done - loaded}
    print(f"Trained {summary['rows']} rows ({summary['games']} games, {summary['teams']} teams, "
          f"{first} to {last}) with the {engine} engine in {summary['train_seconds']:.1f} s "
          f"(+ {summary['load_seconds']:.1f} s loading the CSV) -> {out_path.name}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train priors on the Kaggle history into a new file.")
    parser.add_argument("--out", required=True, help="new priors file to write")
    parser.add_argument("--before", required=True, metavar="YYYY-MM-DD",
                        help="only games before this date (e.g. 2026-27 opening night, 2026-10-20)")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--all-history", action="store_true", help="use every game before --before")
    parser.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE, help=f"default {DEFAULT_ENGINE}")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    args = parser.parse_args(argv)

    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error("--before must be a date like 2026-10-20")
    if args.years <= 0:
        parser.error("--years must be a positive integer")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")
    try:
        run_training(args.csv, args.out, before, years=None if args.all_history else args.years, engine=args.engine)
    except PermissionError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
