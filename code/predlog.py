"""
Prediction log: one CSV row per game, predicted from the priors as they were just
before that game was applied.

    date,game_id,home,away,predicted_winner,win_pct,proj_home,proj_away,actual_home,actual_away
    2026-10-20,0022600001,DET,BOS,DET,57.31,112.4,110.9,121,113

win_pct is the predicted winner's win probability in percent. Rows are keyed by
game_id: appending a game that is already in the file does nothing.

Metrics
-------
  winner accuracy   share of games where predicted_winner won
  avg confidence    mean win_pct of the predicted winner; well calibrated when it matches the accuracy
  spread MAE        mean |(proj_home - proj_away) - (actual_home - actual_away)|, in points
  Brier score       mean (p - y)^2, p = win_pct / 100, y = 1 if predicted_winner won else 0
                    (identical to the home-win form; 0 is perfect, 0.25 is a coin flip)
  projected total   mean proj_home + proj_away;  actual total: mean actual_home + actual_away
  margin error SD   SD of (actual_home - actual_away) - (proj_home - proj_away): how far real
                    margins land from the projection. A well-calibrated simulation's own
                    margin SD (simulation.py "margin_sd") matches it.
"""
import statistics
import csv
import os
from pathlib import Path

COLUMNS = ["date", "game_id", "home", "away", "predicted_winner", "win_pct",
           "proj_home", "proj_away", "actual_home", "actual_away"]
PREDICT_SEED = 2026  # fixed seed for the Monte Carlo simulation behind every logged prediction


def make_row(game: dict, prediction: dict) -> dict:
    """A log row from a game (boxscores.py shape) and predict.predict_game's result."""
    home, away = game["home"], game["away"]
    home_favored = prediction["home_win_pct"] >= prediction["away_win_pct"]
    return {
        "date": game["date"],
        "game_id": game["game_id"],
        "home": home["abbr"],
        "away": away["abbr"],
        "predicted_winner": home["abbr"] if home_favored else away["abbr"],
        "win_pct": f"{prediction['home_win_pct'] if home_favored else prediction['away_win_pct']:.2f}",
        "proj_home": f"{prediction['home_score']:.1f}",
        "proj_away": f"{prediction['away_score']:.1f}",
        "actual_home": str(home["pts"]),
        "actual_away": str(away["pts"]),
    }


def read_log(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_rows(path, rows) -> int:
    """Append rows whose game_id isn't in the file yet; returns how many were written."""
    path = Path(path)
    seen = {r["game_id"] for r in read_log(path)}
    new = []
    for row in rows:
        if row["game_id"] not in seen:
            seen.add(row["game_id"])
            new.append(row)
    if not new:
        return 0
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new)
        file.flush()
        os.fsync(file.fileno())
    return len(new)


METRIC_KEYS = ("accuracy", "confidence", "spread_mae", "brier", "proj_total", "actual_total", "margin_sd")


def metrics(rows) -> dict:
    """{"games", *METRIC_KEYS} over the rows (None values when there are no rows)."""
    if not rows:
        return {"games": 0, **{k: None for k in METRIC_KEYS}}
    correct, spread_errors, brier = 0, [], []
    confidence = 0.0
    proj_totals, actual_totals, margin_errors = [], [], []
    for r in rows:
        actual_home, actual_away = int(r["actual_home"]), int(r["actual_away"])
        actual_winner = r["home"] if actual_home > actual_away else r["away"]
        hit = r["predicted_winner"] == actual_winner
        correct += hit
        projected = float(r["proj_home"]) - float(r["proj_away"])
        spread_errors.append(abs(projected - (actual_home - actual_away)))
        brier.append((float(r["win_pct"]) / 100 - hit) ** 2)
        confidence += float(r["win_pct"]) / 100
        proj_totals.append(float(r["proj_home"]) + float(r["proj_away"]))
        actual_totals.append(actual_home + actual_away)
        margin_errors.append((actual_home - actual_away) - projected)
    n = len(rows)
    return {"games": n, "accuracy": correct / n, "confidence": confidence / n,
            "spread_mae": sum(spread_errors) / n, "brier": sum(brier) / n,
            "proj_total": sum(proj_totals) / n, "actual_total": sum(actual_totals) / n,
            "margin_sd": statistics.pstdev(margin_errors)}


def format_metrics(m: dict) -> str:
    if not m["games"]:
        return "no games in the log yet"
    return (f"{m['games']} games: winner accuracy {m['accuracy']:.1%} (avg confidence {m['confidence']:.1%}), "
            f"spread MAE {m['spread_mae']:.2f} pts, Brier {m['brier']:.4f}, "
            f"total {m['proj_total']:.1f} projected vs {m['actual_total']:.1f} actual")
