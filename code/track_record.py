"""
The model's track record for the website (Phase 13). DISPLAY ONLY: reads the prediction logs, the saved
betting lines and reports/accuracy_report.md; writes nothing and never touches the model.

    build(today) -> {"season": ..., "last7": ..., "recent": [...], "backtest": ..., "vegas": ...}

  season    2026-27 so far, from the live prediction logs (LIVE_LOGS, the same ones report.py scores:
            log_pace-home-rest.csv = the first 210 games, then prediction_log.csv, which update.py --update
            appends to before applying each game; the first log wins on a duplicate): games, winners picked,
            Brier score, average confidence of the pick.
  last7     the same over games dated in the 7 days up to `today` (US Eastern).
  recent    the RECENT (10) latest games: date, matchup, pick + its win %, final score, right or wrong.
  backtest  the headline of the live-procedure backtest, parsed from reports/accuracy_report.md's main
            table (Model / Vegas (spread) / Home team: games, winners picked, Brier), so the site always
            shows what the report says; None if the report is missing or its table can't be read.
  vegas     model vs Vegas on this season's logged games that have a closing line (lines.closing_lines:
            the last snapshot before tip-off), scored like report.py's "Vegas (spread)": P(home) =
            Phi(-spread_home / SD), SD from the report (13.83, fitted on the training games). None until a
            logged game has a closing line.

Winners picked: a 50/50 pick counts half (as in report.py). Brier: mean (P(home wins) - home won)^2.
"""
import math
import re
from datetime import date, timedelta
from pathlib import Path

import lines
import predlog

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_LOGS = [REPO_ROOT / "log_pace-home-rest.csv", REPO_ROOT / "prediction_log.csv"]   # = report.LIVE_LOGS
REPORT_PATH = REPO_ROOT / "reports" / "accuracy_report.md"
RECENT = 10
DAYS = 7
DEFAULT_VEGAS_SD = 13.83   # the report's fitted SD, if the report can't be read
BACKTEST_HEADING = "## Training seasons, main"
BACKTEST_ROWS = ("Model", "Vegas (spread)", "Home team")


# ---------------------------------------------------------------------------
# the season's games
# ---------------------------------------------------------------------------

def log_row_to_game(r: dict) -> dict:
    """A prediction-log row as a game (the same as report.log_row_to_game, plus the pick and its win %)."""
    home_pts, away_pts = int(r["actual_home"]), int(r["actual_away"])
    p = float(r["win_pct"]) / 100
    p_home = p if r["predicted_winner"] == r["home"] else 1 - p
    return {"game_id": r["game_id"], "date": r["date"], "home": r["home"], "away": r["away"],
            "home_pts": home_pts, "away_pts": away_pts, "home_won": home_pts > away_pts, "p_model": p_home,
            "pick": r["predicted_winner"], "pick_pct": float(r["win_pct"])}


def season_games(log_paths=None) -> list:
    """Every logged game, oldest first (date, game_id); the first log wins on a duplicate game_id."""
    seen, games = set(), []
    for path in log_paths or LIVE_LOGS:
        for r in predlog.read_log(path):
            if r["game_id"] not in seen:
                seen.add(r["game_id"])
                games.append(log_row_to_game(r))
    return sorted(games, key=lambda g: (g["date"], g["game_id"]))


def hit(p_home: float, home_won: bool) -> float:
    return 0.5 if p_home == 0.5 else float((p_home > 0.5) == home_won)


def summarize(games: list, p_key: str = "p_model") -> dict:
    """{"games", "right", "accuracy", "brier", "confidence"} (None values without games)."""
    n = len(games)
    if not n:
        return {"games": 0, "right": 0, "accuracy": None, "brier": None, "confidence": None}
    hits = [hit(g[p_key], g["home_won"]) for g in games]
    return {"games": n, "right": sum(hits), "accuracy": sum(hits) / n,
            "brier": sum((g[p_key] - g["home_won"]) ** 2 for g in games) / n,
            "confidence": sum(max(g[p_key], 1 - g[p_key]) for g in games) / n}


# ---------------------------------------------------------------------------
# the accuracy report's headline
# ---------------------------------------------------------------------------

def _num(text):
    text = text.strip().rstrip("%").replace(",", "")
    return float(text)


def parse_backtest(text: str):
    """
    {"generated", "sd", "rows": {name: {"games", "accuracy", "brier"}}} from the report's markdown, or None.
    Reads the first table after the BACKTEST_HEADING heading (columns Predictor | Games | Winner accuracy | Brier ...).
    """
    start = text.find(BACKTEST_HEADING)
    if start < 0:
        return None
    rows = {}
    for line in text[start:].splitlines()[1:]:
        if line.startswith("## "):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in BACKTEST_ROWS:
            try:
                rows[cells[0]] = {"games": int(_num(cells[1])), "accuracy": _num(cells[2]) / 100,
                                  "brier": _num(cells[3])}
            except ValueError:
                return None
        elif rows and not line.strip():
            break
    if set(rows) != set(BACKTEST_ROWS):
        return None
    generated = re.search(r"Generated (\d{4}-\d{2}-\d{2})", text)
    sd = re.search(r"SD = \*\*([\d.]+) points\*\*", text)
    return {"generated": generated.group(1) if generated else None,
            "sd": float(sd.group(1)) if sd else None, "rows": rows}


def read_backtest(path=None):
    try:
        return parse_backtest(Path(path or REPORT_PATH).read_text(encoding="utf-8"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Vegas on this season's games
# ---------------------------------------------------------------------------

def spread_win_prob(spread_home: float, sd: float) -> float:
    """P(home wins) = Phi(-spread_home / sd) (report.spread_win_prob)."""
    return 0.5 * (1 + math.erf(-spread_home / sd / math.sqrt(2)))


def vegas_games(games: list, snapshots: list, sd: float) -> list:
    """The games with a closing line, each with "p_vegas" (from the closing spread)."""
    closing = lines.closing_lines(snapshots)
    out = []
    for g in games:
        c = closing.get(g["game_id"])
        if c is None or c.get("spread_home") in (None, ""):
            continue
        out.append({**g, "p_vegas": spread_win_prob(float(c["spread_home"]), sd)})
    return out


# ---------------------------------------------------------------------------
# everything for the page
# ---------------------------------------------------------------------------

def build(today: date, log_paths=None, lines_path=None, report_path=None, recent: int = RECENT, days: int = DAYS) -> dict:
    games = season_games(log_paths)
    since = (today - timedelta(days=days - 1)).isoformat()
    week = [g for g in games if since <= g["date"] <= today.isoformat()]
    backtest = read_backtest(report_path)
    sd = (backtest or {}).get("sd") or DEFAULT_VEGAS_SD
    with_lines = vegas_games(games, lines.read_lines(lines_path), sd)
    vegas = None
    if with_lines:
        vegas = {"games": len(with_lines), "model": summarize(with_lines), "vegas": summarize(with_lines, "p_vegas"),
                 "sd": sd}
    latest = list(reversed(games[-recent:])) if recent > 0 else []
    return {"season": summarize(games), "last7": summarize(week), "since": since, "today": today.isoformat(),
            "first": games[0]["date"] if games else None, "last": games[-1]["date"] if games else None,
            "recent": [{**g, "right": hit(g["p_model"], g["home_won"]) == 1} for g in latest],
            "backtest": backtest, "vegas": vegas}
