"""
Live-procedure backtest (Phase 10): the training seasons run exactly the way 2026-27 is run live.

    python code/backtest.py                  # 2024-25 and 2025-26: prints accuracy, Brier, margin MAE per season (~1 min)

For each season S starting in year Y (2024-25: Y = 2024):
  1. Preseason priors: the live training recipe (dynamic_model.train_priors with the live defaults:
     exact engine, fitted sigma, paces, home boost, rest), flat start, tau = 0, on the 3 years of
     games before Y-10-03, the same calendar cutoff as the live model's 2026-10-03. Nothing from
     season S itself is used.
  2. Season reset: update.reset_priors with N = 20 (NBA stds 0.1/sqrt(20), pace stds 2 sigma_pace/sqrt(20)).
  3. Every NBA-vs-NBA game of S, oldest first (date, gameId): predicted with predict.predict_game
     (live defaults, the prediction log's fixed seed, neutral-site flag, both teams' rest) from the
     ratings as they stand, THEN applied with update.apply_game (M = 30: drift 0.1/30 per game,
     paces, home, rest). This is what update.py --update does for each 2026-27 game.

Differences from the live run, all from the data: the season's games come from the Kaggle CSV
(four factors, possessions; the same numbers NBA.com serves), and playoff / play-in games are
included (the live run hasn't reached them yet). The Kaggle file has no game types for 2021-22,
so 2024-25's preseason priors come from 2022-23 and 2023-24 only (1 stray 2021-22 game aside);
2025-26's come from three full seasons, like 2026-27's. A game without possessions would be
skipped (neither predicted nor applied), as --update skips it; there are none.

The model's constants (sigma, home, rest, pace noise, points formula) were fitted on the 2023-24
to 2025-26 window, so this is still a backtest, not a forecast.

Reads the Kaggle CSV and the cached NBA.com schedules (stats/schedules/; the preseason trainings
also need 2021-22's and 2022-23's, downloaded once for their neutral-site / rest flags). Writes
nothing.
"""
import argparse
import time

import numpy as np

import boxscores
import dynamic_model
import params
import predlog
import rest as rest_days
import update
from boxscores import season_of
from predict import predict_game
from schedule import NBA_TEAMS

SEASONS = ("2024-25", "2025-26")
KAGGLE_FACTORS = {"eFG": "effectiveFieldGoalPercentage", "TOV": "teamTurnoverPercentage",
                  "ORB": "offensiveReboundPercentage", "FTR": "freeThrowAttemptRate"}


def preseason_cutoff(season: str) -> str:
    """
    The training cutoff for a season: Oct 3 of its first year, like the live model's 2026-10-03. Only
    seasons before 2026-27 can be backtested: 2026-27 and later are the test set (the live log).
    """
    cutoff = f"{season[:4]}-10-03"
    if cutoff >= dynamic_model.TRAINING_CUTOFF:
        raise ValueError(f"{season}: only seasons before 2026-27 can be backtested; 2026-27 games are the test set")
    return cutoff


def season_games(data: dict, season: str):
    """
    (games, skipped): `season`'s NBA-vs-NBA games from load_training_rows data, in update.apply_game
    shape ({"game_id", "date", "home": {"id", "abbr", "pts", eFG, TOV, ORB, FTR}, "away": ...,
    "possessions", "neutral", "rest": {"home", "away"}}), oldest first (date, gameId).
    skipped: gameIds left out because a side or the possessions are missing.
    """
    cleaned = data["cleaned"]
    possessions = {boxscores.game_key(g["game_id"]): g["possessions"] for g in data["possessions"]}
    by_id = {}
    for gid, when, team, opponent, home, score, *factors in zip(
            cleaned["gameId"], cleaned["gameDateTimeEst"], cleaned["teamId"], cleaned["opponentTeamId"],
            cleaned["home"], cleaned["teamScore"], *(cleaned[c] for c in KAGGLE_FACTORS.values())):
        key = boxscores.game_key(gid)
        team, opponent = str(int(team)), str(int(opponent))
        if season_of(key) != season or team not in NBA_TEAMS or opponent not in NBA_TEAMS:
            continue
        game = by_id.setdefault(key, {"game_id": key, "date": when.date().isoformat()})
        game["home" if int(home) == 1 else "away"] = {
            "id": team, "abbr": NBA_TEAMS[team][0], "pts": int(score),
            **{f: float(v) for f, v in zip(KAGGLE_FACTORS, factors)}}
    complete = [g for g in by_id.values() if "home" in g and "away" in g and g["game_id"] in possessions]
    skipped = sorted(set(by_id) - {g["game_id"] for g in complete})
    ids = [g["game_id"] for g in complete]
    neutral = boxscores.neutral_games(ids) if ids else {}
    table = rest_days.rest_games(ids) if ids else {}
    for g in complete:
        g["possessions"] = possessions[g["game_id"]]
        g["neutral"] = neutral[g["game_id"]]
        g["rest"] = {side: table[g["game_id"]][g[side]["id"]]["rest"] for side in ("home", "away")}
        g["winner"] = g["home"]["id"] if g["home"]["pts"] > g["away"]["pts"] else g["away"]["id"]
    return sorted(complete, key=lambda g: (g["date"], g["game_id"])), skipped


def preseason_priors(csv_path, season: str) -> dict:
    """The live training recipe on the 3 years before the season's cutoff: {"priors", "first", "last", "games"}."""
    data = dynamic_model.load_training_rows(csv_path, preseason_cutoff(season), 3)
    return {"priors": dynamic_model.train_priors(data)["priors"], "first": data["first"], "last": data["last"],
            "games": data["games"]}


def run_season(start_priors: dict, games: list, n: int = update.DEFAULT_N, m: int = update.DEFAULT_M,
               calib: str = params.DEFAULT_CALIB_MODE) -> list:
    """
    The live in-season procedure on `games`: reset (N), then each game predicted, then applied (M).
    Returns one row per game: {"game_id", "date", "season", "home_won", "margin", "total", "p_model",
    "proj_margin", "proj_total"} (the report's row shape). calib: the win-probability calibration
    (Phase 10b; it changes only p_model, never the ratings).
    """
    priors = update.reset_priors(start_priors, n)
    update_sigma, sim_sigma = params.update_sigma(), params.sim_sigma()
    rows = []
    for game in games:
        home, away = game["home"], game["away"]
        prediction = predict_game(home["id"], away["id"], priors, seed=predlog.PREDICT_SEED, sigma=sim_sigma,
                                  neutral=game["neutral"], rest_home=game["rest"]["home"],
                                  rest_away=game["rest"]["away"], calib=calib)
        if prediction is None:
            raise KeyError(f"game {game['game_id']}: a team is missing from the priors")
        margin = home["pts"] - away["pts"]
        rows.append({"game_id": game["game_id"], "date": game["date"], "season": season_of(game["game_id"]),
                     "home_won": margin > 0, "margin": margin, "total": home["pts"] + away["pts"],
                     "p_model": prediction["home_win_pct"] / 100,
                     "proj_margin": prediction["home_score"] - prediction["away_score"],
                     "proj_total": prediction["home_score"] + prediction["away_score"]})
        update.apply_game(priors, game, m, sigma=update_sigma)
    return rows


def run_backtest(csv_path=None, seasons=SEASONS, n: int = update.DEFAULT_N, m: int = update.DEFAULT_M,
                 data=None, calib: str = params.DEFAULT_CALIB_MODE) -> dict:
    """
    Every season in `seasons` with the live procedure. data: the scoring window's load_training_rows
    (default: the 3 years before 2026-10-03, which hold both seasons).
    Returns {"rows": [...], "seasons": {season: {"cutoff", "trained_on", "train_games", "games", "skipped"}}}.
    """
    csv_path = csv_path or dynamic_model.DEFAULT_CSV
    data = data or dynamic_model.load_training_rows(csv_path, dynamic_model.TRAINING_CUTOFF, 3)
    rows, info = [], {}
    for season in seasons:
        games, skipped = season_games(data, season)
        pre = preseason_priors(csv_path, season)
        rows += run_season(pre["priors"], games, n, m, calib=calib)
        info[season] = {"cutoff": preseason_cutoff(season), "trained_on": f"{pre['first']} to {pre['last']}",
                        "train_games": pre["games"], "games": len(games), "skipped": skipped}
    return {"rows": rows, "seasons": info, "n": n, "m": m, "calib": calib}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest the training seasons with the live in-season procedure.")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS), help=f"default {' '.join(SEASONS)}")
    parser.add_argument("--n", type=int, default=update.DEFAULT_N, help=f"season reset N (default {update.DEFAULT_N})")
    parser.add_argument("--m", type=int, default=update.DEFAULT_M, help=f"per-game drift M (default {update.DEFAULT_M})")
    parser.add_argument("--calib", choices=params.SWITCH_MODES, default=params.DEFAULT_CALIB_MODE,
                        help=f"calibrated win %% (Phase 10b); default {params.DEFAULT_CALIB_MODE}")
    args = parser.parse_args(argv)
    for season in args.seasons:
        try:
            preseason_cutoff(season)
        except ValueError as err:
            parser.error(str(err))
    start = time.perf_counter()
    result = run_backtest(args.csv, args.seasons, args.n, args.m, calib=args.calib)
    for season, s in result["seasons"].items():
        rows = [r for r in result["rows"] if r["season"] == season]
        p = np.array([r["p_model"] for r in rows])
        y = np.array([r["home_won"] for r in rows], dtype=float)
        hit = np.where(p == 0.5, 0.5, np.where(p > 0.5, y, 1 - y))
        mae = np.mean([abs(r["proj_margin"] - r["margin"]) for r in rows])
        print(f"{season}: {len(rows)} games (preseason priors from {s['train_games']} games, {s['trained_on']}; "
              f"skipped {len(s['skipped'])}): accuracy {hit.mean():.1%}, Brier {np.mean((p - y) ** 2):.4f}, "
              f"margin MAE {mae:.2f}")
    print(f"N={result['n']}, M={result['m']}, calib={result['calib']}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
