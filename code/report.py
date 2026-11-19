"""
Accuracy report (Phase 10): the live model vs Vegas and two simple baselines.

    python code/report.py                      # -> reports/accuracy_report.md + reports/calibration.png (~1 min)
    python code/report.py --betting-csv path/to/nba_2008-2026.csv

Reads only: the Kaggle training CSV, the Kaggle betting CSV (historical_lines.py), the prediction
logs (log_pace-home-rest.csv, prediction_log.csv) and data/lines_2026-27.csv. Writes only the two
report files. Never touches the model, priors or logs.

Predictors
----------
  Model         the live model (params.py defaults: exact engine, fitted sigma, pace, home, rest)
  Vegas         the closing spread as a win probability: home margin ~ N(-spread_home, SD^2), so
                P(home wins) = Phi(-spread_home / SD). SD = the RMS of (actual margin + spread_home)
                over the TRAINING games (fit_spread_sd); no 2026-27 game is used.
  Vegas ML      the no-vig moneyline probability (lines.py home_win_prob), where we have one: the
                training seasons' Kaggle file has no moneylines, so only 2026-27 games
  Home          always the home team, with P = the training seasons' home win rate
  Better record the team with the better win% in that season before that day (0 games = .500),
                with P = the training seasons' win rate of that pick; equal records -> Home

Samples
-------
  Training      2024-25 and 2025-26, two backtests, each game predicted only from earlier games.
                Vegas: Kaggle lines (historical_lines.py).
                MAIN: the live in-season procedure (backtest.py): preseason priors from the live
                training recipe on the 3 years before each season, season reset (N = 20), then
                each game predicted and applied with drift (M = 30), as update.py --update does.
                SECONDARY: train_calibration.run_pass's live variant, i.e. the TRAINING pass (tau = 0,
                no season reset) over the 3-year window, the first season as burn-in.
                Caveat: the model's constants (sigma, home, rest, pace, points formula) and the Vegas
                SD were fitted on these same seasons.
  2026-27       the live predictions (log_pace-home-rest.csv, then prediction_log.csv for later
                games), each made before the game. Vegas: closing lines from data/lines_2026-27.csv
                (lines.closing_lines), only for games that have one, so that comparison grows as
                lines are collected. Model vs Home / Better record uses every logged game.

Metrics (per predictor, on the same games)
-------
  accuracy      share of winners picked (P > 0.5 picks home, P < 0.5 away; P = 0.5 counts half)
  Brier         mean (P(home) - home won)^2
  log loss      mean -log P(actual winner), P clipped to [1e-4, 1 - 1e-4] (a 10,000-draw simulation
                can say 0% or 100%)
  margin MAE    mean |projected home margin - actual| (model: simulated scores; Vegas: -spread)
  total MAE     mean |projected total - actual| (model: simulated scores; Vegas: the total)
  calibration   confidence (P of the pick) vs accuracy in bins
Paired differences vs Vegas are means of the per-game differences with SE = SD / sqrt(n).
"""
import argparse
import hashlib
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import backtest
import dynamic_model
import historical_lines
import lines
import params
import predlog
import train_calibration
from boxscores import season_of

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "accuracy_report.md"
CHART_NAME = "calibration.png"
LIVE_LOGS = [REPO_ROOT / "log_pace-home-rest.csv", REPO_ROOT / "prediction_log.csv"]
EPS = 1e-4
BIN_EDGES = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)
LIVE_VARIANT = "live"


# ---------------------------------------------------------------------------
# spread -> probability
# ---------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fit_spread_sd(margins, spreads_home) -> dict:
    """
    SD of the normal model home margin ~ N(-spread_home, SD^2), by maximum likelihood with the
    mean fixed at the line: SD = sqrt(mean((margin + spread_home)^2)). Also the mean residual
    (bias: > 0 = home teams beat the spread on average) and n.
    """
    residuals = np.asarray(margins, dtype=float) + np.asarray(spreads_home, dtype=float)
    if residuals.size == 0:
        raise ValueError("no games to fit the spread SD on")
    return {"sd": float(np.sqrt(np.mean(residuals ** 2))), "bias": float(residuals.mean()), "n": int(residuals.size)}


def spread_win_prob(spread_home, sd: float):
    """P(home wins) from the home spread (negative = home favored): Phi(-spread_home / sd)."""
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    if np.ndim(spread_home) == 0:
        return normal_cdf(-float(spread_home) / sd)
    return np.array([normal_cdf(-float(s) / sd) for s in spread_home])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def per_game(p_home, home_won) -> dict:
    """Per-game arrays: hit (1 / 0, 0.5 for P = 0.5), brier, log loss, confidence (P of the pick)."""
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(home_won, dtype=float)
    if p.shape != y.shape:
        raise ValueError("p_home and home_won differ in length")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    hit = np.where(p == 0.5, 0.5, np.where(p > 0.5, y, 1 - y))
    clipped = np.clip(p, EPS, 1 - EPS)
    return {"hit": hit, "brier": (p - y) ** 2, "log_loss": -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)),
            "confidence": np.maximum(p, 1 - p)}


def evaluate(p_home, home_won, proj_margin=None, margin=None, proj_total=None, total=None) -> dict:
    """
    {"n", "accuracy", "brier", "log_loss", "confidence", "margin_mae", "total_mae", "each": per-game arrays}.
    margin_mae / total_mae are None unless both the projection and the actual values are given.
    """
    each = per_game(p_home, home_won)
    if proj_margin is not None and margin is not None:
        each["margin_err"] = np.abs(np.asarray(proj_margin, dtype=float) - np.asarray(margin, dtype=float))
    if proj_total is not None and total is not None:
        each["total_err"] = np.abs(np.asarray(proj_total, dtype=float) - np.asarray(total, dtype=float))
    n = len(each["hit"])

    def mean(key):
        return float(each[key].mean()) if key in each and n else None

    return {"n": n, "accuracy": mean("hit"), "brier": mean("brier"), "log_loss": mean("log_loss"),
            "confidence": mean("confidence"), "margin_mae": mean("margin_err"), "total_mae": mean("total_err"),
            "each": each}


def paired(a, b) -> dict:
    """Mean of the per-game differences a - b, its SE (SD / sqrt(n)) and z; NaNs when n < 2."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n < 2:
        return {"n": n, "diff": float(d.mean()) if n else float("nan"), "se": float("nan"), "z": float("nan")}
    se = float(d.std(ddof=1) / math.sqrt(n))
    return {"n": n, "diff": float(d.mean()), "se": se, "z": float(d.mean() / se) if se > 0 else float("nan")}


def calibration_bins(p_home, home_won, edges=BIN_EDGES) -> list:
    """[(lo, hi, n, mean confidence, accuracy)] of the pick's confidence; empty bins left out, the last bin includes 1."""
    each = per_game(p_home, home_won)
    conf, hit = each["confidence"], each["hit"]
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mask = (conf >= lo) & ((conf <= hi) if last else (conf < hi))
        if mask.any():
            out.append((lo, hi, int(mask.sum()), float(conf[mask].mean()), float(hit[mask].mean())))
    return out


def disagreements(p_model, p_vegas, home_won) -> dict:
    """Games where the two pick different winners (a P = 0.5 pick'em is no pick): count and who was right."""
    pm, pv = np.asarray(p_model, dtype=float), np.asarray(p_vegas, dtype=float)
    y = np.asarray(home_won, dtype=bool)
    differ = (pm != 0.5) & (pv != 0.5) & ((pm > 0.5) != (pv > 0.5))
    model_right = differ & ((pm > 0.5) == y)
    return {"games": int(differ.sum()), "model_right": int(model_right.sum()),
            "vegas_right": int((differ & ~model_right).sum()), "vegas_pickem": int((pv == 0.5).sum())}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def records_before(games: list) -> dict:
    """
    {game_id: ((home wins, home losses), (away wins, away losses))}: each team's record in the same
    season (boxscores.season_of) from games on EARLIER dates only. games: {"game_id", "date",
    "home", "away", "home_won"} (team keys of any kind).
    """
    by_day = defaultdict(list)
    for g in games:
        by_day[(season_of(g["game_id"]), g["date"])].append(g)
    record = defaultdict(lambda: [0, 0])   # (season, team) -> [wins, losses]
    out = {}
    for season, day in sorted(by_day):
        todays = by_day[(season, day)]
        for g in todays:
            out[g["game_id"]] = (tuple(record[(season, g["home"])]), tuple(record[(season, g["away"])]))
        for g in todays:
            winner, loser = (g["home"], g["away"]) if g["home_won"] else (g["away"], g["home"])
            record[(season, winner)][0] += 1
            record[(season, loser)][1] += 1
    return out


def win_pct(record) -> float:
    wins, losses = record
    return 0.5 if wins + losses == 0 else wins / (wins + losses)


def better_record_side(records) -> int:
    """+1 if the home team's win% is higher, -1 if the away team's, 0 if equal."""
    home, away = (win_pct(r) for r in records)
    return 0 if home == away else (1 if home > away else -1)


def fit_baselines(games: list) -> dict:
    """
    From training games ({"game_id", "date", "home", "away", "home_won"}): "home" = the home win
    rate; "record" = how often the better-record team won (games with different records only).
    """
    if not games:
        raise ValueError("no training games for the baselines")
    records = records_before(games)
    home_rate = float(np.mean([g["home_won"] for g in games]))
    sides = [(better_record_side(records[g["game_id"]]), g["home_won"]) for g in games]
    picked = [(side == 1) == won for side, won in sides if side != 0]
    return {"home": home_rate, "record": float(np.mean(picked)) if picked else 0.5, "n": len(games),
            "record_n": len(picked)}


def baseline_probs(games: list, fitted: dict, records: dict) -> dict:
    """{"home": P(home) per game, "record": P(home) per game} for `games`, with the fitted constants."""
    home = np.full(len(games), fitted["home"])
    record = []
    for g in games:
        side = better_record_side(records[g["game_id"]])
        record.append(fitted["home"] if side == 0 else (fitted["record"] if side == 1 else 1 - fitted["record"]))
    return {"home": home, "record": np.array(record)}


# ---------------------------------------------------------------------------
# the two samples
# ---------------------------------------------------------------------------

def check_live_variant():
    """train_calibration's "live" variant must be the params.py defaults (so the report scores the live model)."""
    v = train_calibration.VARIANTS[LIVE_VARIANT]
    expected = {"sigma": params.DEFAULT_SIM_SIGMA_MODE, "sim_corr": params.DEFAULT_SIM_CORR_MODE,
                "pace": params.DEFAULT_PACE_MODE}
    if v != expected:
        raise RuntimeError(f"train_calibration's {LIVE_VARIANT!r} variant {v} is not the live simulation {expected}")


def join_lines(model_rows: list, by_id: dict, line_by_id: dict) -> list:
    """Model rows ({"game_id", "home_won", "margin", "total", "p_model", "proj_margin", "proj_total"}) with
    their game's date, season, type and Vegas spread / total; games without both lines are left out."""
    out = []
    for r in model_rows:
        line, g = line_by_id.get(r["game_id"]), by_id.get(r["game_id"])
        if g is None or line is None or line["spread_home"] is None or line["total"] is None:
            continue
        out.append({**r, "date": g["date"], "season": season_of(r["game_id"]), "game_type": g["game_type"],
                    "spread_home": line["spread_home"], "vegas_total": line["total"], "p_ml": None})
    return out


def training_sample(training_csv, betting_csv) -> dict:
    """
    The training seasons: both backtests' predictions (main: the live procedure, backtest.py; secondary:
    the training pass, train_calibration.run_pass) joined to the Kaggle lines, plus baselines and the Vegas SD.
    """
    check_live_variant()
    data = dynamic_model.load_training_rows(training_csv, dynamic_model.TRAINING_CUTOFF, 3)
    games = historical_lines.games_from_frame(data["cleaned"])
    for g in games:
        g["home_won"] = g["home_pts"] > g["away_pts"]
    line_rows, match = historical_lines.match_lines(historical_lines.read_betting_csv(betting_csv), games)
    line_by_id = {r["game_id"]: r for r in line_rows}
    by_id = {g["game_id"]: g for g in games}

    with_spread = [g for g in games if line_by_id.get(g["game_id"], {}).get("spread_home") is not None]
    sd = fit_spread_sd([g["home_pts"] - g["away_pts"] for g in with_spread],
                       [line_by_id[g["game_id"]]["spread_home"] for g in with_spread])
    baselines = fit_baselines(games)
    records = records_before(games)

    live = backtest.run_backtest(training_csv, data=data)
    records_out = train_calibration.run_pass(data, params.DEFAULT_HOME_MODE, [LIVE_VARIANT], params.DEFAULT_REST_MODE,
                                             with_ids=True)[LIVE_VARIANT]
    pass_rows = [{"game_id": gid, "home_won": margin > 0, "margin": margin, "total": total, "p_model": p,
                  "proj_margin": proj_margin, "proj_total": proj_total}
                 for p, margin, proj_margin, proj_total, total, _, gid in records_out]
    return {"rows": join_lines(live["rows"], by_id, line_by_id), "pass_rows": join_lines(pass_rows, by_id, line_by_id),
            "live": live, "scored": len(records_out), "sd": sd, "baselines": baselines, "records": records,
            "match": match, "first": data["first"], "last": data["last"]}


def compare_procedures(live_rows: list, pass_rows: list) -> dict:
    """Paired live procedure − training pass on the games both scored: accuracy, Brier, log loss, margin MAE."""
    by_id = {r["game_id"]: r for r in pass_rows}
    common = [(r, by_id[r["game_id"]]) for r in live_rows if r["game_id"] in by_id]
    if not common:
        return {"n": 0}
    won = [a["home_won"] for a, _ in common]
    a = evaluate([x["p_model"] for x, _ in common], won, [x["proj_margin"] for x, _ in common], [x["margin"] for x, _ in common])
    b = evaluate([y["p_model"] for _, y in common], won, [y["proj_margin"] for _, y in common], [y["margin"] for _, y in common])
    return {"n": len(common), "live": a, "pass": b,
            **{k: paired(a["each"][key], b["each"][key]) for k, key in
               (("accuracy", "hit"), ("brier", "brier"), ("log_loss", "log_loss"), ("margin_mae", "margin_err"))}}


def log_row_to_game(r: dict) -> dict:
    """A prediction-log row as {"game_id", "date", "home", "away", "home_won", "p_model", margins, totals}."""
    home_pts, away_pts = int(r["actual_home"]), int(r["actual_away"])
    p = float(r["win_pct"]) / 100
    p_home = p if r["predicted_winner"] == r["home"] else 1 - p
    proj_home, proj_away = float(r["proj_home"]), float(r["proj_away"])
    return {"game_id": r["game_id"], "date": r["date"], "season": season_of(r["game_id"]),
            "home": r["home"], "away": r["away"], "home_won": home_pts > away_pts,
            "margin": home_pts - away_pts, "total": home_pts + away_pts, "p_model": p_home,
            "proj_margin": proj_home - proj_away, "proj_total": proj_home + proj_away}


def season_sample(log_paths, lines_path) -> dict:
    """2026-27: every logged prediction (first log wins on duplicates), plus closing lines where we have them."""
    seen, games, used_logs = set(), [], []
    for path in log_paths:
        rows = predlog.read_log(path)
        if rows:
            used_logs.append((Path(path).name, len(rows)))
        for r in rows:
            if r["game_id"] not in seen:
                seen.add(r["game_id"])
                games.append(log_row_to_game(r))
    games.sort(key=lambda g: (g["date"], g["game_id"]))
    records = records_before(games)

    snapshots = lines.read_lines(lines_path)
    closing = lines.closing_lines(snapshots)
    with_lines, hours = [], []
    for g in games:
        c = closing.get(g["game_id"])
        if c is None or c.get("spread_home") in (None, "") or c.get("total") in (None, ""):
            continue
        hours.append((lines.parse_utc(c["tipoff_utc"]) - lines.parse_utc(c["fetched_at"])).total_seconds() / 3600)
        with_lines.append({**g, "spread_home": float(c["spread_home"]), "vegas_total": float(c["total"]),
                           "p_ml": float(c["home_win_prob"]) if c.get("home_win_prob") not in (None, "") else None})
    return {"games": games, "rows": with_lines, "records": records, "logs": used_logs,
            "snapshots": len(snapshots), "snapshot_games": len({s["game_id"] for s in snapshots if s.get("game_id")}),
            "hours_before": float(np.median(hours)) if hours else None}


# ---------------------------------------------------------------------------
# scoring a sample
# ---------------------------------------------------------------------------

def score_games(games: list, sd: float, baselines: dict, records: dict, with_vegas: bool) -> dict:
    """{predictor: evaluate(...)} on `games` (every predictor on the same games), plus paired diffs and disagreements."""
    won = [g["home_won"] for g in games]
    margin, total = [g["margin"] for g in games], [g["total"] for g in games]
    base = baseline_probs(games, baselines, records)
    p_model = np.array([g["p_model"] for g in games])
    out = {"Model": evaluate(p_model, won, [g["proj_margin"] for g in games], margin,
                             [g["proj_total"] for g in games], total)}
    if with_vegas:
        spreads = np.array([g["spread_home"] for g in games])
        p_vegas = spread_win_prob(spreads, sd) if games else np.array([])
        out["Vegas (spread)"] = evaluate(p_vegas, won, -spreads, margin, [g["vegas_total"] for g in games], total)
    out["Home team"] = evaluate(base["home"], won)
    out["Better record"] = evaluate(base["record"], won)
    result = {"metrics": out, "n": len(games)}
    model_each = out["Model"]["each"]
    result["vs_baselines"] = {
        name: {k: paired(model_each[key], out[name]["each"][key])
               for k, key in (("accuracy", "hit"), ("brier", "brier"), ("log_loss", "log_loss"))}
        for name in ("Home team", "Better record")}
    if with_vegas and games:
        vegas = out["Vegas (spread)"]["each"]
        result["paired"] = {
            name: {k: paired(m["each"][key], vegas[key]) for k, key in
                   (("accuracy", "hit"), ("brier", "brier"), ("log_loss", "log_loss"),
                    ("margin_mae", "margin_err"), ("total_mae", "total_err"))
                   if key in m["each"] and key in vegas}
            for name, m in out.items() if name != "Vegas (spread)"}
        result["disagree"] = disagreements(p_model, p_vegas, won)
        proj_margin = np.array([g["proj_margin"] for g in games])
        proj_total = np.array([g["proj_total"] for g in games])
        vegas_total = np.array([g["vegas_total"] for g in games])
        # mean |model - Vegas|: the Vegas margin is -spread_home, so model margin - Vegas margin = proj + spread
        result["gap"] = {"margin": float(np.mean(np.abs(proj_margin + spreads))),
                         "total": float(np.mean(np.abs(proj_total - vegas_total)))}
        result["calibration"] = {"Model": calibration_bins(p_model, won), "Vegas (spread)": calibration_bins(p_vegas, won)}
        ml = [g for g in games if g.get("p_ml") is not None]
        if ml:
            ml_won = [g["home_won"] for g in ml]
            result["ml"] = {"Model": evaluate([g["p_model"] for g in ml], ml_won),
                            "Vegas (moneyline, no vig)": evaluate([g["p_ml"] for g in ml], ml_won),
                            "Vegas (spread)": evaluate(spread_win_prob([g["spread_home"] for g in ml], sd), ml_won)}
    else:
        result["calibration"] = {"Model": calibration_bins(p_model, won)} if games else {}
    return result


# ---------------------------------------------------------------------------
# markdown + chart
# ---------------------------------------------------------------------------

def fmt(value, kind):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return {"pct": f"{100 * value:.1f}%", "brier": f"{value:.4f}", "ll": f"{value:.4f}", "pts": f"{value:.2f}"}[kind]


def md_table(headers, rows) -> str:
    lines_ = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    lines_ += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines_)


def metrics_table(metrics: dict) -> str:
    rows = [[name, m["n"], fmt(m["accuracy"], "pct"), fmt(m["brier"], "brier"), fmt(m["log_loss"], "ll"),
             fmt(m["margin_mae"], "pts"), fmt(m["total_mae"], "pts"), fmt(m["confidence"], "pct")]
            for name, m in metrics.items()]
    return md_table(["Predictor", "Games", "Winner accuracy", "Brier", "Log loss", "Margin MAE", "Total MAE",
                     "Avg confidence"], rows)


def diff_cell(d: dict, kind: str, scale: float = 1.0) -> str:
    if d is None or math.isnan(d.get("se", float("nan"))):
        return "–"
    digits = {"pct": 1, "brier": 4, "ll": 4, "pts": 2}[kind]
    unit = " pts" if kind == "pct" else ""
    return f"{scale * d['diff']:+.{digits}f}{unit} ± {scale * d['se']:.{digits}f} ({d['z']:+.1f} SE)"


def paired_table(paired_: dict) -> str:
    rows = []
    for name, p in paired_.items():
        rows.append([name, diff_cell(p.get("accuracy"), "pct", 100), diff_cell(p.get("brier"), "brier"),
                     diff_cell(p.get("log_loss"), "ll"), diff_cell(p.get("margin_mae"), "pts"),
                     diff_cell(p.get("total_mae"), "pts")])
    return md_table(["vs Vegas (spread)", "Accuracy", "Brier", "Log loss", "Margin MAE", "Total MAE"], rows)


def baseline_table(vs: dict) -> str:
    rows = [[name, diff_cell(p["accuracy"], "pct", 100), diff_cell(p["brier"], "brier"), diff_cell(p["log_loss"], "ll")]
            for name, p in vs.items()]
    return md_table(["Model − baseline", "Accuracy", "Brier", "Log loss"], rows)


def calibration_table(cal: dict) -> str:
    rows = []
    for name, bins in cal.items():
        for lo, hi, n, conf, acc in bins:
            rows.append([name, f"{100 * lo:.0f}–{100 * hi:.0f}%", n, fmt(conf, "pct"), fmt(acc, "pct"),
                         f"{100 * (acc - conf):+.1f}"])
    return md_table(["Predictor", "Confidence bin", "Games", "Avg confidence", "Accuracy", "Accuracy − confidence (pts)"],
                    rows)


def draw_chart(path, panels: list) -> None:
    """Reliability diagram: one panel per (title, {predictor: bins}); point size ~ games, bars = ±1 binomial SE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Model": "#2563eb", "Vegas (spread)": "#dc2626", "Model, training pass (secondary)": "#9ca3af"}
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.6), squeeze=False)
    for ax, (title, cal) in zip(axes[0], panels):
        ax.plot([0.5, 1], [0.5, 1], color="#9ca3af", linestyle="--", linewidth=1, label="perfect calibration")
        for name, bins in cal.items():
            if not bins:
                continue
            conf = np.array([b[3] for b in bins])
            acc = np.array([b[4] for b in bins])
            n = np.array([b[2] for b in bins])
            se = np.sqrt(np.clip(acc * (1 - acc), 1e-9, None) / n)
            ax.errorbar(conf, acc, yerr=se, color=colors.get(name, "#374151"), linewidth=1.2, capsize=3,
                        marker="o", markersize=0, label=f"{name} ({int(n.sum())} games)")
            ax.scatter(conf, acc, s=12 + 140 * n / n.max(), color=colors.get(name, "#374151"), zorder=3)
        ax.set_xlim(0.48, 1.0)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("Confidence in the pick (bin average)")
        ax.set_ylabel("Share of picks that won")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Calibration: predicted confidence vs actual accuracy (bars: ±1 SE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verdict(d: dict, lower_is_better: bool) -> str:
    """"better" / "worse" for the model when the paired difference is at least 2 SE, else "same"."""
    if d is None or math.isnan(d["z"]) or abs(d["z"]) < 2:
        return "same"
    return "better" if (d["diff"] < 0) == lower_is_better else "worse"


def baseline_phrase(label: str, d: dict, model: dict, base: dict) -> str:
    """How the model did against one baseline, in words that follow the 2-SE rule on Brier and on winners."""
    acc, bri = verdict(d["accuracy"], lower_is_better=False), verdict(d["brier"], lower_is_better=True)
    accs = f"{fmt(model['accuracy'], 'pct')} vs {fmt(base['accuracy'], 'pct')}"
    briers = f"{fmt(model['brier'], 'brier')} vs {fmt(base['brier'], 'brier')}"
    if acc == bri == "better":
        return f"beat {label} on both winners picked ({accs}) and Brier ({briers})"
    if acc == bri == "worse":
        return f"trailed {label} on both winners picked ({accs}) and Brier ({briers})"
    words = {"better": "beat", "worse": "trailed", "same": "was level with"}
    second = {"better": "beat it", "worse": "trailed it", "same": "was level"}
    return f"{words[bri]} {label} on Brier ({briers}) and {second[acc]} on winners picked ({accs})"


def resume_bullets(train: dict, season: dict) -> list:
    """
    3-4 lines built only from the report's numbers. Better / worse / level follow the paired 2-SE rule;
    "close to Vegas" is used only when the difference is within noise, otherwise the gap is stated.
    """
    t = train["result"]
    m, v = t["metrics"]["Model"], t["metrics"]["Vegas (spread)"]
    home = t["metrics"]["Home team"]
    brier = t["paired"]["Model"]["brier"]
    n = t["n"]
    vegas = verdict(brier, lower_is_better=True)
    if vegas == "same":
        vs = (f"close to Vegas closing lines on Brier score ({fmt(m['brier'], 'brier')} vs {fmt(v['brier'], 'brier')}, "
              f"difference {brier['diff']:+.4f} ± {brier['se']:.4f}) and picked {fmt(m['accuracy'], 'pct')} of "
              f"winners vs {fmt(v['accuracy'], 'pct')}")
    elif vegas == "better":
        vs = (f"beat Vegas closing lines on Brier score ({fmt(m['brier'], 'brier')} vs {fmt(v['brier'], 'brier')}, "
              f"{abs(brier['z']):.1f} SE)")
    else:
        gap = home["brier"] - v["brier"]
        share = (home["brier"] - m["brier"]) / gap if gap > 0 else None
        vs = (f"picked {fmt(m['accuracy'], 'pct')} of winners vs {fmt(v['accuracy'], 'pct')} for Vegas closing "
              f"lines (Brier {fmt(m['brier'], 'brier')} vs {fmt(v['brier'], 'brier')})"
              + (f", closing {share:.0%} of the Brier gap between always picking the home team and Vegas"
                 if share is not None and share > 0 else ""))
    bullets = [
        "Built a Bayesian NBA game predictor in Python: Four Factors offense/defense ratings updated game by game "
        "with conjugate Normal (Kalman) updates, and a 10,000-draw Monte Carlo game simulation with team pace, "
        "home-court and rest effects fitted by one-step-ahead likelihood.",
        f"Backtested the live procedure (preseason ratings trained only on earlier seasons, then updated "
        f"after every game) on {n:,} games of 2024-25 and 2025-26: the model {vs}.",
        f"On the same games the model {baseline_phrase('always picking the home team', t['vs_baselines']['Home team'], m, home)}, "
        f"and {baseline_phrase('picking the team with the better record', t['vs_baselines']['Better record'], m, t['metrics']['Better record'])}.",
    ]
    s_all, s_vegas = season["all"], season["vegas"]
    sm = s_all["metrics"]["Model"]
    if sm["n"]:
        line = (f"Runs live as a daily pipeline (NBA.com data, crash-safe updates, every prediction logged before "
                f"tip-off): {fmt(sm['accuracy'], 'pct')} of winners and Brier {fmt(sm['brier'], 'brier')} over the "
                f"first {sm['n']} games of 2026-27 (±{100 * math.sqrt(0.25 / sm['n']):.1f} pts), vs "
                f"{fmt(s_all['metrics']['Home team']['accuracy'], 'pct')} for the home team and "
                f"{fmt(s_all['metrics']['Better record']['accuracy'], 'pct')} for the better record")
        if s_vegas["n"]:
            pv = s_vegas["paired"]["Model"]["brier"]
            line += (f"; on the {s_vegas['n']} games with saved closing lines, Brier "
                     f"{fmt(s_vegas['metrics']['Model']['brier'], 'brier')} vs Vegas "
                     f"{fmt(s_vegas['metrics']['Vegas (spread)']['brier'], 'brier')} ({pv['diff']:+.4f} ± {pv['se']:.4f})")
        bullets.append(line + ".")
    return bullets


METRIC_WORDS = (("accuracy", "winners picked", False), ("brier", "Brier", True), ("log_loss", "log loss", True),
                ("margin_mae", "margin MAE", True), ("total_mae", "total MAE", True))


def significance_sentence(paired_model: dict) -> str:
    """Which metrics the model is 2+ SE better / worse than Vegas on, in one sentence."""
    groups = {"better": [], "worse": []}
    for key, words, lower in METRIC_WORDS:
        v = verdict(paired_model.get(key), lower)
        if v in groups:
            groups[v].append(words)
    parts = []
    if groups["worse"]:
        parts.append(f"Vegas is better by 2+ SE on {', '.join(groups['worse'])}")
    if groups["better"]:
        parts.append(f"the model is better by 2+ SE on {', '.join(groups['better'])}")
    return ("; ".join(parts) + ".") if parts else "No metric differs from Vegas by 2 SE or more."


def calibration_sentence(label: str, m: dict) -> str:
    """Average confidence vs accuracy in words: over/underconfident when they differ by 2+ binomial SEs."""
    gap = m["accuracy"] - m["confidence"]
    se = math.sqrt(m["accuracy"] * (1 - m["accuracy"]) / m["n"])
    if abs(gap) < 2 * se:
        word = "calibrated within noise"
    else:
        word = "underconfident" if gap > 0 else "overconfident"
    return (f"{label} {word} (average confidence {fmt(m['confidence'], 'pct')} vs {fmt(m['accuracy'], 'pct')} of "
            f"winners picked)")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_section(L: list, result: dict, by_season: dict, calibration: bool = True) -> None:
    """The tables of one training backtest: metrics, paired vs Vegas, vs baselines, disagreements, by season, calibration."""
    L.append(metrics_table(result["metrics"]) + "\n")
    L.append("**Paired differences vs Vegas (spread)** (negative Brier / log loss / MAE = better than Vegas):\n")
    L.append(paired_table(result["paired"]) + "\n")
    L.append("**Model vs the baselines** (positive accuracy / negative Brier and log loss = the model is better):\n")
    L.append(baseline_table(result["vs_baselines"]) + "\n")
    d = result["disagree"]
    L.append(f"**Disagreements:** the model and Vegas picked different winners in {d['games']} games "
             f"({100 * d['games'] / result['n']:.1f}%); the model was right in {d['model_right']} "
             f"({100 * d['model_right'] / max(d['games'], 1):.1f}%), Vegas in {d['vegas_right']}. "
             f"({d['vegas_pickem']} pick'em lines had no Vegas pick.) "
             f"On average the model's margin is {result['gap']['margin']:.2f} pts and its total "
             f"{result['gap']['total']:.2f} pts from the Vegas numbers.\n")
    L.append("**By season** (same games, split):\n")
    rows = []
    for season_name, r in by_season.items():
        mm, vv = r["metrics"]["Model"], r["metrics"]["Vegas (spread)"]
        rows.append([season_name, r["n"], fmt(mm["accuracy"], "pct"), fmt(vv["accuracy"], "pct"),
                     fmt(mm["brier"], "brier"), fmt(vv["brier"], "brier"), diff_cell(r["paired"]["Model"]["brier"], "brier")])
    L.append(md_table(["Season", "Games", "Model acc.", "Vegas acc.", "Model Brier", "Vegas Brier",
                       "Brier diff (model − Vegas)"], rows) + "\n")
    if calibration:
        L.append("**Calibration**\n")
        L.append(calibration_table(result["calibration"]) + "\n")


def write_report(train: dict, season: dict, out_path: Path, betting_csv: Path, lines_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart = out_path.parent / CHART_NAME
    t, tp, sd = train["result"], train["pass_result"], train["sd"]
    s_all, s_vegas = season["all"], season["vegas"]
    live_info = train["live"]
    panel = dict(t["calibration"])
    panel["Model, training pass (secondary)"] = tp["calibration"]["Model"]
    panels = [(f"Training seasons, live-procedure backtest ({t['n']:,} games)", panel)]
    if s_all["n"]:
        cal = dict(s_all["calibration"])
        if s_vegas["n"]:
            cal["Vegas (spread)"] = s_vegas["calibration"]["Vegas (spread)"]
        panels.append((f"2026-27 so far (model: {s_all['n']} games; Vegas: {s_vegas['n']})", cal))
    draw_chart(chart, panels)

    m_tr, v_tr = t["metrics"]["Model"], t["metrics"]["Vegas (spread)"]
    b = train["baselines"]
    match = train["match"]
    proc = train["procedures"]
    L = []
    L.append("# Accuracy report: the live model vs Vegas and simple baselines\n")
    L.append(f"Generated {date.today().isoformat()} by `python code/report.py`. Live model as in `params.py`: "
             f"exact engine, fitted σ in the update and the simulation, team paces, home-court boost, rest "
             f"(back-to-backs), 10,000-draw simulation.\n")
    L.append("## Summary\n")
    pb = t["paired"]["Model"]
    L.append(f"- **Training seasons, live-procedure backtest (2024-25, 2025-26; {t['n']:,} games):** model "
             f"{fmt(m_tr['accuracy'], 'pct')} winners, Brier {fmt(m_tr['brier'], 'brier')}; Vegas closing spread "
             f"{fmt(v_tr['accuracy'], 'pct')}, Brier {fmt(v_tr['brier'], 'brier')}. Model − Vegas: Brier "
             f"{diff_cell(pb['brier'], 'brier')}, accuracy {diff_cell(pb['accuracy'], 'pct', 100)}. "
             + significance_sentence(pb))
    if proc["n"]:
        L.append(f"- **The live procedure vs the training-pass backtest** (secondary, {proc['n']:,} common games): "
                 f"Brier {diff_cell(proc['brier'], 'brier')}, accuracy {diff_cell(proc['accuracy'], 'pct', 100)} "
                 f"(live − training pass).")
    L.append("- **Calibration:** " + calibration_sentence("live-procedure backtest", m_tr)
             + ("; " + calibration_sentence(f"2026-27 ({s_all['n']} games)", s_all["metrics"]["Model"])
                if s_all["n"] else "") + ".")
    if s_vegas["n"]:
        pv = s_vegas["paired"]["Model"]
        L.append(f"- **2026-27 with closing lines ({s_vegas['n']} games):** model "
                 f"{fmt(s_vegas['metrics']['Model']['accuracy'], 'pct')}, Brier "
                 f"{fmt(s_vegas['metrics']['Model']['brier'], 'brier')}; Vegas "
                 f"{fmt(s_vegas['metrics']['Vegas (spread)']['accuracy'], 'pct')}, "
                 f"{fmt(s_vegas['metrics']['Vegas (spread)']['brier'], 'brier')} (Brier difference "
                 f"{diff_cell(pv['brier'], 'brier')}).")
    else:
        L.append(f"- **2026-27 with closing lines: {s_vegas['n']} games so far.** Line collection started on "
                 f"2026-11-19, after the {s_all['n']} games logged so far; this comparison grows as "
                 f"`update.py --update` / `lines.py` save lines and the games are played.")
    if s_all["n"]:
        sm = s_all["metrics"]
        L.append(f"- **2026-27, all {s_all['n']} logged games (no Vegas yet):** model "
                 f"{fmt(sm['Model']['accuracy'], 'pct')} winners, Brier {fmt(sm['Model']['brier'], 'brier')}; "
                 f"home team {fmt(sm['Home team']['accuracy'], 'pct')}; better record "
                 f"{fmt(sm['Better record']['accuracy'], 'pct')}.")
    L.append("")

    L.append("## Resume bullets\n")
    L.extend(f"- {line}" for line in resume_bullets(train, season))
    L.append("\nEvery number above is from this report; the training-season numbers are backtests (see the caveats).\n")

    L.append("## Method\n")
    seasons_text = "; ".join(f"{name}: priors trained on {info['train_games']:,} games ({info['trained_on']}, cutoff "
                             f"{info['cutoff']}), then {info['games']:,} games played"
                             for name, info in live_info["seasons"].items())
    L.append(f"- **Model, main training backtest (`backtest.py`): the live in-season procedure.** For each season, "
             f"preseason priors come from the live training recipe (flat start, every game of the 3 years before "
             f"Oct 3 of that season, as `priors_pre_2026-27_pace-home-rest.json` was trained before 2026-10-03), then "
             f"the season reset (N = {live_info['n']}: stds 0.1/√{live_info['n']}), then each game is predicted with "
             f"`predict.predict_game` (the prediction log's seed) and applied with drift (M = {live_info['m']}), "
             f"as `update.py --update` does. {seasons_text}. The Kaggle file has no game types for 2021-22, so "
             f"2024-25's priors rest on 2022-23 and 2023-24 only. Playoff and play-in games are included.")
    L.append(f"- **Model, secondary training backtest:** `train_calibration.py`'s one-step-ahead pass over the 3-year "
             f"window before 2026-10-03 ({train['first']} to {train['last']}): ratings start flat, every game updates "
             f"them with the TRAINING update (tau = 0, no season reset), and each game after the first season "
             f"({train['scored']:,} games) is simulated from the ratings as they stood. Its ratings never forget, "
             f"so by 2025-26 a game from 2023-24 weighs as much as last night's.")
    L.append(f"- **Model, 2026-27:** the logged predictions, each made before the game "
             f"({', '.join(f'`{n}` ({k})' for n, k in season['logs']) or 'no logs'}).")
    L.append(f"- **Vegas (spread):** P(home wins) = Φ(−spread_home / SD), SD = **{sd['sd']:.2f} points**, fitted on "
             f"the {sd['n']:,} training games with a line (the RMS of actual home margin + spread; the mean residual "
             f"is {sd['bias']:+.2f}, i.e. lines are unbiased to within a fraction of a point). No 2026-27 game is "
             f"used. Margin projection = −spread, total projection = the posted total.")
    L.append("- **Vegas (moneyline, no vig):** the median across books of each book's vig-free home probability "
             "(`lines.py`). The Kaggle file's moneylines stop in mid 2022-23, so this exists only for 2026-27 games.")
    L.append(f"- **Home team:** always the home team, P = {b['home']:.3f} (home win rate over the {b['n']:,} training "
             f"games). **Better record:** the team with the better win% in that season before that day (0 games = "
             f".500), P = {b['record']:.3f} (how often that pick won in training, {b['record_n']:,} games); equal "
             f"records fall back to the home team.")
    L.append("- **Metrics:** winner accuracy (a 50/50 pick counts half), Brier = mean (P(home) − home won)², log loss "
             "(P clipped to [0.0001, 0.9999]), margin / total MAE vs the actual result, calibration in confidence "
             "bins. All predictors in a table are scored on the same games. Paired differences are per-game "
             "differences, mean ± SE (SD/√n); |z| < 2 is within noise.")
    L.append(f"- **Data:** Kaggle \"NBA Betting Data | October 2007 to June 2026\" (`{betting_csv.name}`, SHA-256 "
             f"`{sha256(betting_csv)[:16]}…`), matched to {match['matched']:,} of {match['games']:,} training games "
             f"({100 * match['rate']:.1f}%). 2026-27 lines: "
             f"`{(lines_path.resolve().relative_to(REPO_ROOT) if lines_path.resolve().is_relative_to(REPO_ROOT) else lines_path).as_posix()}`, "
             f"{season['snapshots']} snapshots of {season['snapshot_games']} games; the closing line is a game's last "
             f"snapshot before tip-off"
             + (f" (median {season['hours_before']:.1f} h before tip-off)." if season["hours_before"] is not None else "."))
    L.append("")

    L.append(f"## Training seasons, main: live-procedure backtest ({t['n']:,} games)\n")
    training_section(L, t, train["by_season"])
    L.append(f"![Calibration chart]({CHART_NAME})\n")

    L.append(f"## Training seasons, secondary: training-pass backtest ({tp['n']:,} games)\n")
    L.append("The same seasons predicted from the training pass (no season reset, no drift), the setup of "
             "`train_calibration.py`. Kept for comparison; it is not how the model runs live.\n")
    training_section(L, tp, train["pass_by_season"], calibration=False)
    if proc["n"]:
        L.append(f"**Live procedure − training pass** on the {proc['n']:,} games both scored (negative Brier / log "
                 f"loss / MAE and positive accuracy = the live procedure is better):\n")
        L.append(md_table(["", "Accuracy", "Brier", "Log loss", "Margin MAE"],
                          [["Live − training pass", diff_cell(proc["accuracy"], "pct", 100),
                            diff_cell(proc["brier"], "brier"), diff_cell(proc["log_loss"], "ll"),
                            diff_cell(proc["margin_mae"], "pts")]]) + "\n")

    L.append("## 2026-27 season\n")
    L.append(f"### With Vegas closing lines ({s_vegas['n']} games)\n")
    if s_vegas["n"]:
        L.append(metrics_table(s_vegas["metrics"]) + "\n")
        L.append(paired_table(s_vegas["paired"]) + "\n")
        d = s_vegas["disagree"]
        L.append(f"**Disagreements:** {d['games']} games; model right in {d['model_right']}, Vegas in "
                 f"{d['vegas_right']}.\n")
        if "ml" in s_vegas:
            L.append("**Games with a moneyline:**\n")
            L.append(metrics_table(s_vegas["ml"]) + "\n")
        L.append(calibration_table(s_vegas["calibration"]) + "\n")
        if s_vegas["n"] < 100:
            L.append(f"With only {s_vegas['n']} games these numbers are very noisy; read the SEs.\n")
    else:
        L.append("No 2026-27 game has both a prediction and a closing line yet. Lines are saved from 2026-11-19 on "
                 "(each `update.py --update` and `python code/lines.py` run); the games enter the prediction log "
                 "when `--update` applies them, so this section fills in over the season. Re-run "
                 "`python code/report.py` to update it.\n")
    L.append(f"### All logged games, model vs simple baselines ({s_all['n']} games, no Vegas)\n")
    if s_all["n"]:
        L.append(metrics_table(s_all["metrics"]) + "\n")
        L.append(baseline_table(s_all["vs_baselines"]) + "\n")
        L.append(calibration_table(s_all["calibration"]) + "\n")
        L.append(f"With {s_all['n']} games, one standard error of the accuracy is about "
                 f"{100 * math.sqrt(0.25 / s_all['n']):.1f} points.\n")

    L.append("## Caveats\n")
    L.append("- **The training seasons are a backtest, not a forecast.** Each prediction uses only earlier games for "
             "the team ratings, but the model's constants (game noise σ, home boost, rest effect, pace noise, points "
             "formula) were fitted on the 2023-24 to 2025-26 window, which flatters the model a little. The Vegas SD "
             "(one number) was fitted on it too. 2026-27 is the only true out-of-sample test.")
    L.append(f"- **The season reset and drift (N = {live_info['n']}, M = {live_info['m']}) are the live settings,** "
             "used as they are; no other values were tried on these backtests.")
    L.append("- **The live-procedure backtest differs from the live run only in its data:** the games come from the "
             "Kaggle CSV (the same NBA.com four factors and possessions), and playoff / play-in games are included.")
    L.append("- **Kaggle lines:** the dataset gives one spread and total per game and doesn't say exactly when they "
             "were taken; treat them as closing or near-closing lines.")
    L.append("- **2026-27 \"closing\" lines** are the last snapshot before tip-off, which can be hours before the true "
             "close when lines were only fetched in the morning (median shown above).")
    L.append("- **The 2026-27 season so far is small.** Its first 210 games' accuracy was about 2 SE above what their "
             "own margin errors predict (CLAUDE.md, Phase 7), so expect it to drift down.")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def build(training_csv, betting_csv, log_paths, lines_path, out_path) -> dict:
    train = training_sample(training_csv, betting_csv)
    sd = train["sd"]["sd"]
    train["result"] = score_games(train["rows"], sd, train["baselines"], train["records"], with_vegas=True)
    train["pass_result"] = score_games(train["pass_rows"], sd, train["baselines"], train["records"], with_vegas=True)
    for key, rows_key in (("by_season", "rows"), ("pass_by_season", "pass_rows")):
        train[key] = {}
        for season in sorted({r["season"] for r in train[rows_key]}):
            rows = [r for r in train[rows_key] if r["season"] == season]
            train[key][season] = score_games(rows, sd, train["baselines"], train["records"], with_vegas=True)
    train["procedures"] = compare_procedures(train["rows"], train["pass_rows"])

    s = season_sample(log_paths, lines_path)
    season = {"all": score_games(s["games"], sd, train["baselines"], s["records"], with_vegas=False),
              "vegas": score_games(s["rows"], sd, train["baselines"], s["records"], with_vegas=True),
              **{k: s[k] for k in ("logs", "snapshots", "snapshot_games", "hours_before")}}
    write_report(train, season, Path(out_path), Path(betting_csv), Path(lines_path))
    return {"train": train, "season": season}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Accuracy report: the live model vs Vegas and simple baselines.")
    parser.add_argument("--betting-csv", default=str(historical_lines.DEFAULT_BETTING_CSV),
                        help="Kaggle nba_2008-2026.csv (see historical_lines.py)")
    parser.add_argument("--training-csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--lines", default=str(lines.LINES_PATH), help="2026-27 betting lines CSV (lines.py)")
    parser.add_argument("--log", nargs="+", default=[str(p) for p in LIVE_LOGS],
                        help="2026-27 prediction logs, in priority order")
    parser.add_argument("--out", default=str(REPORT_PATH), help="report markdown (the chart goes next to it)")
    args = parser.parse_args(argv)
    if not Path(args.betting_csv).exists():
        parser.error(f"{args.betting_csv} not found; see code/historical_lines.py for the download")
    if Path(args.out).suffix != ".md":
        parser.error("--out must be a .md file")
    result = build(args.training_csv, args.betting_csv, args.log, args.lines, args.out)
    t, s = result["train"]["result"], result["season"]
    print(f"Training: {t['n']} games; 2026-27: {s['all']['n']} logged, {s['vegas']['n']} with closing lines.")
    print(f"Wrote {args.out} and {Path(args.out).parent / CHART_NAME}")


if __name__ == "__main__":
    main()
