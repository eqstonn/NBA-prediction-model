"""
Historical closing lines for the training seasons (Phase 10a), matched to our training gameIds.

    python code/historical_lines.py                          # match rate of stats/betting/nba_2008-2026.csv
    python code/historical_lines.py --csv path/to/nba_2008-2026.csv
    python code/historical_lines.py --out data/lines_training.csv   # also save the matched lines

Dataset
-------
Kaggle "NBA Betting Data | October 2007 to June 2026" (user cviaxmiwnptr), file nba_2008-2026.csv
(~2.5 MB, one row per game). Download it from
https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
(Kaggle login needed) and put it at stats/betting/nba_2008-2026.csv (stats/ is gitignored).

Columns used: season (the season's END year: 2024 = 2023-24), date (US Eastern), regular,
playoffs, away, home (lowercase codes: "gs", "ny", "no", "sa", "utah", "wsh", ...), score_away,
score_home, whos_favored ("home"/"away"), spread (always >= 0: the favorite's margin), total,
moneyline_away, moneyline_home. For 2023-24 to 2025-26 every game has a spread and a total, but
the moneylines are EMPTY (they stop in mid 2022-23), so ml_home/ml_away come out blank.

Matching
--------
The training games are the Kaggle TeamStatisticsExtended rows that dynamic_model.py trains on
(--before 2026-10-03, 3 years; NBA vs NBA only, so the All-Star games drop out). Each is keyed
by (Eastern date, home team, away team); a betting row matches on the same key, or with home/away
swapped (neutral sites, marked flipped). A team never plays twice on one date, so the key is
unambiguous; the scores are compared only as a check and disagreements are reported, not
dropped. Checked against NBA.com on 2026-11-19 (3,956 training games, 100% matched): 5 scores
disagree, and in every one it is a data error, not a different game (0022400627 and 0022400988
have 0-0 in the Kaggle training rows; 0022500029, 0022500232 and 0022500801 have wrong scores
in the betting file). The 6 betting rows in the window without a training game are the 2023
and 2024 NBA Cup finals and the 4 regular-season games the Kaggle training CSV lacks (rest.py).
Output, one row per matched game, in lines.py's convention:

    game_id, date, home, away, spread_home (negative = home favored), total, ml_home, ml_away,
    flipped, source
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from data_prep import clean_data_frame, create_data_frame
from lines import match_game
from schedule import ABBR_TO_ID, NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BETTING_CSV = REPO_ROOT / "stats" / "betting" / "nba_2008-2026.csv"
TRAINING_CSVS = [REPO_ROOT / "stats" / "TeamStatisticsExtended.csv",
                 REPO_ROOT / "stats" / "NBA-DataSet" / "TeamStatisticsExtended.csv"]   # as dynamic_model.py
TRAINING_CUTOFF = "2026-10-03"
SOURCE = "kaggle:cviaxmiwnptr/nba_2008-2026"
OUT_FIELDS = ["game_id", "date", "home", "away", "spread_home", "total", "ml_home", "ml_away", "flipped", "source"]


def team_id(code):
    """NBA.com team id for the dataset's lowercase team codes ("gs", "utah", "bkn", ...), or None."""
    return ABBR_TO_ID.get(str(code).strip().upper())


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number   # NaN -> None


def read_betting_csv(path) -> list:
    """
    The dataset's rows in our convention: {"date", "season", "home_id", "away_id", "home_code",
    "away_code", "home_pts", "away_pts", "spread_home", "total", "ml_home", "ml_away"}.
    home_id/away_id are None for a code we don't know.
    """
    out = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        for r in csv.DictReader(file):
            spread = _number(r.get("spread"))
            favored = (r.get("whos_favored") or "").strip().lower()
            if spread is not None and favored in ("home", "away"):
                spread_home = -spread if favored == "home" else spread
                spread_home = spread_home + 0.0   # -0.0 (a pick'em) -> 0.0
            else:
                spread_home = None
            ml_home, ml_away = _number(r.get("moneyline_home")), _number(r.get("moneyline_away"))
            out.append({"date": r["date"][:10], "season": r.get("season"),
                        "home_id": team_id(r["home"]), "away_id": team_id(r["away"]),
                        "home_code": r["home"], "away_code": r["away"],
                        "home_pts": _number(r.get("score_home")), "away_pts": _number(r.get("score_away")),
                        "spread_home": spread_home, "total": _number(r.get("total")),
                        "ml_home": None if ml_home is None else int(ml_home),
                        "ml_away": None if ml_away is None else int(ml_away)})
    return out


def training_games(csv_path=None, before=TRAINING_CUTOFF, years=3) -> list:
    """
    The training window's NBA-vs-NBA games, one per gameId: {"game_id", "date", "home", "away",
    "home_pts", "away_pts", "game_type"}, from the Kaggle TeamStatisticsExtended rows (home == 1).
    """
    csv_path = Path(csv_path or next((p for p in TRAINING_CSVS if p.exists()), TRAINING_CSVS[0]))
    df = clean_data_frame(create_data_frame(csv_path), before=before, years=years)
    games = {}
    for gid, when, team, opponent, home, pts, opp_pts, game_type in zip(
            df["gameId"], df["gameDateTimeEst"], df["teamId"], df["opponentTeamId"], df["home"],
            df["teamScore"], df["opponentScore"], df["gameType"]):
        team, opponent = str(int(team)), str(int(opponent))
        if int(home) != 1 or team not in NBA_TEAMS or opponent not in NBA_TEAMS:
            continue
        game_id = str(int(gid)).zfill(10)
        games.setdefault(game_id, {"game_id": game_id, "date": pd.Timestamp(when).date().isoformat(),
                                   "home": team, "away": opponent, "home_pts": float(pts),
                                   "away_pts": float(opp_pts), "game_type": game_type})
    return sorted(games.values(), key=lambda g: (g["date"], g["game_id"]))


def match_lines(betting_rows: list, games: list):
    """
    (matched, report). matched: one OUT_FIELDS row per training game that has a betting line.
    report: {"games", "matched", "rate", "by_season": {season: (matched, games)}, "flipped",
             "missing": [games without a line], "unmatched_lines": [betting rows in the window
             with no training game], "score_mismatch": [(game, betting row)], "duplicates": [...],
             "unknown_codes": Counter}.
    """
    index = {(g["date"], g["home"], g["away"]): g for g in games}
    first, last = (games[0]["date"], games[-1]["date"]) if games else ("", "")
    matched, used = {}, set()
    unmatched, mismatch, duplicates, unknown = [], [], [], Counter()
    flipped_count = 0
    for row in betting_rows:
        if not first <= row["date"] <= last:
            continue
        if row["home_id"] is None or row["away_id"] is None:
            unknown.update(c for c, i in ((row["home_code"], row["home_id"]), (row["away_code"], row["away_id"]))
                           if i is None)
            unmatched.append(row)
            continue
        game, flipped = match_game(index, row["date"], row["home_id"], row["away_id"])
        if game is None:
            unmatched.append(row)
            continue
        if game["game_id"] in used:
            duplicates.append((game, row))
            continue
        used.add(game["game_id"])
        home_pts, away_pts = (row["away_pts"], row["home_pts"]) if flipped else (row["home_pts"], row["away_pts"])
        if (home_pts, away_pts) != (game["home_pts"], game["away_pts"]):
            mismatch.append((game, row))   # same date and teams, so the same game: reported, line kept
        flip = -1 if flipped else 1
        spread = None if row["spread_home"] is None else flip * row["spread_home"] + 0.0
        ml_home, ml_away = (row["ml_away"], row["ml_home"]) if flipped else (row["ml_home"], row["ml_away"])
        flipped_count += bool(flipped)
        matched[game["game_id"]] = {"game_id": game["game_id"], "date": game["date"],
                                    "home": NBA_TEAMS[game["home"]][0], "away": NBA_TEAMS[game["away"]][0],
                                    "spread_home": spread, "total": row["total"], "ml_home": ml_home,
                                    "ml_away": ml_away, "flipped": int(bool(flipped)), "source": SOURCE}

    by_season = {}
    for g in games:
        season = _season(g["game_id"])
        hit, total = by_season.get(season, (0, 0))
        by_season[season] = (hit + (g["game_id"] in matched), total + 1)
    rows = [matched[g["game_id"]] for g in games if g["game_id"] in matched]
    report = {"games": len(games), "matched": len(rows), "rate": len(rows) / len(games) if games else 0.0,
              "by_season": by_season, "flipped": flipped_count,
              "missing": [g for g in games if g["game_id"] not in matched],
              "unmatched_lines": unmatched, "score_mismatch": mismatch, "duplicates": duplicates,
              "unknown_codes": unknown}
    return rows, report


def _season(game_id: str) -> str:
    year = 2000 + int(game_id[3:5])
    return f"{year}-{str(year + 1)[2:]}"


def print_report(report: dict) -> None:
    print(f"Training games (NBA vs NBA): {report['games']}; with a betting line: {report['matched']} "
          f"({100 * report['rate']:.1f}%); matched with home/away swapped: {report['flipped']}")
    for season, (hit, total) in sorted(report["by_season"].items()):
        print(f"  {season}: {hit}/{total} ({100 * hit / total:.1f}%)")
    for label, items, show in (
            ("training games without a line", report["missing"],
             lambda g: f"{g['game_id']} {g['date']} {NBA_TEAMS[g['away']][0]} @ {NBA_TEAMS[g['home']][0]} "
                       f"({g['game_type']})"),
            ("betting rows in the window without a training game", report["unmatched_lines"],
             lambda r: f"{r['date']} {r['away_code']} @ {r['home_code']}"),
            ("score disagreements (same date and teams, line kept)", report["score_mismatch"],
             lambda gr: f"{gr[0]['game_id']} {gr[0]['date']}: training {gr[0]['away_pts']:.0f}-{gr[0]['home_pts']:.0f}, "
                        f"betting {gr[1]['away_pts']:.0f}-{gr[1]['home_pts']:.0f}"),
            ("duplicate betting rows (second one not used)", report["duplicates"],
             lambda gr: f"{gr[0]['game_id']} {gr[0]['date']}")):
        print(f"{label}: {len(items)}")
        for item in items[:15]:
            print(f"  {show(item)}")
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more")
    if report["unknown_codes"]:
        print(f"unknown team codes: {dict(report['unknown_codes'])}")


def write_matched(path, rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: "" if r.get(k) is None else r[k] for k in OUT_FIELDS} for r in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Match historical closing lines to the training games.")
    parser.add_argument("--csv", default=str(DEFAULT_BETTING_CSV),
                        help=f"the Kaggle betting CSV (default {DEFAULT_BETTING_CSV.relative_to(REPO_ROOT)})")
    parser.add_argument("--training-csv", help="Kaggle TeamStatisticsExtended.csv (default: as dynamic_model.py)")
    parser.add_argument("--out", help="also write the matched lines to this CSV")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out")
    args = parser.parse_args(argv)
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found; download nba_2008-2026.csv from Kaggle "
                     "(see the top of code/historical_lines.py) and put it there, or pass --csv")
    if args.out and Path(args.out).exists() and not args.overwrite:
        parser.error(f"{args.out} already exists; pass --overwrite to replace it")

    rows, report = match_lines(read_betting_csv(args.csv), training_games(args.training_csv))
    print_report(report)
    if args.out:
        write_matched(args.out, rows)
        print(f"Wrote {len(rows)} matched lines to {args.out}")


if __name__ == "__main__":
    main()
