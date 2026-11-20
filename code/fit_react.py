"""
Rating reactivity (Phase 10c): diagnose why the live ratings under-react, tune the in-season procedure's
season reset N and drift M on 2024-25 only, and test the result on 2025-26 and 2026-27.

    python code/fit_react.py --diagnose          # slope by month / rating, one knob at a time (~5 min); writes nothing
    python code/fit_react.py                     # N x M grid on the 2024-25 backtest -> model_params.json["react"] (~8 min)
    python code/fit_react.py --overwrite         # refit (replaces the react section)
    python code/fit_react.py --evaluate --old-log log_pace-home-rest.csv --new-log log_react.csv
                                                 # old vs new on 2025-26 (backtest) and 2026-27 (logs); writes nothing

The problem
-----------
In the live-procedure backtest (backtest.py; N=20, M=30) actual margins are ~1.11x the projected
ones in both seasons (slope 1.094 in 2024-25, 1.126 in 2025-26): the ratings are too compressed.

Diagnosis (--diagnose; 2026-11-19)
----------------------------------
  * By month the slope GROWS through the season: Oct-Dec ~0.7-1.2 (pooled ~1.0), Feb-Apr 1.13-1.56
    (pooled ~1.3). The projections are fine at the start and fall behind later.
  * The preseason priors alone (never updated) predict less and less: their slope falls from ~0.75-0.9
    in Oct/Nov to 0.1-0.7 by March. In October it is already below 1, so the preseason means are
    not too shrunk (stretching them away from the league average by 1.2x doesn't help: Brier 0.2126
    vs 0.2124 on 2024-25). They go stale: teams change during the season.
  * By rating, the miss grows with distance from average (+0.11 points of margin per point of net
    rating, 2.4 SE): good teams are underrated, bad ones overrated, a uniform ~10% compression.
  * One knob at a time on 2024-25: N=5 Brier 0.2124 -> 0.2103 (slope 1.02), M=18 -> 0.2118 (slope
    1.04; it moves the Feb-Apr slope most), update sigma x0.8 -> 0.2112. Scaling the update sigma by c
    gives the SAME rating means as N x c^2, M x c (the exact update depends only on variance ratios;
    checked: identical means, slope, MAE), so it is not a separate cause and not a third parameter.
    The fitted sigma itself is right (Phase 10b: residuals have 91-98% of the predicted variance).
  * So the compression comes from the season reset: N=20 makes the 3-year preseason ratings worth
    20 games, and with M=30 that anchor fades too slowly, so the ratings stay tied to stale preseason
    means. Lowering N removes the anchor; lowering M only fades it (and adds noise), so on Brier N is
    the main lever and, once N is small, M between 18 and 40 hardly matters.

The fit
-------
Grid N in {1, 2, 3, 5, 8, 12, 20} x M in {10, 14, 18, 24, 30, 40}, each a full live-procedure run of
2024-25 (backtest.run_season, live simulation, the log's seed); the pick is the lowest one-step-ahead
Brier (log loss reported too). 2025-26 and 2026-27 are test sets. The calibration temperature (Phase
10b) depends on how spread the projections are, so it is refitted on 2024-25 with the chosen N, M
(fit_calib.fit_temperature) and saved in the same section (used when --calib and --react are both on).

Results (2026-11-19)
--------------------
  * Fit (2024-25, 1,316 games): N=2, M=24. Brier 0.2124 -> 0.2100, log loss 0.6130 -> 0.6075, slope
    1.094 -> 0.965, winners 66.7% -> 65.7%. The surface is flat for N 2-5 and M 18-40 (within 0.0004);
    N=3, M=24 is a near-tie (0.21005 vs 0.21004). Calibration temperature refitted: 1.2718 (SE 0.089),
    vs 1.3612 at N=20, M=30: wider projections need less sharpening.
  * Test, 2025-26 backtest (1,322): Brier 0.2145 -> 0.2092 (-0.0054 ± 0.0013, -4.2 SE), log loss
    -0.0123 (-4.3 SE), winners 66.2% -> 67.1% (+1.1 SE), spread MAE 11.67 -> 11.54 (-2.0 SE), slope
    1.126 -> 0.991. With calibration on both: Brier 0.2122 -> 0.2074 (-3.1 SE).
  * Test, 2026-27 (log_react.csv vs log_pace-home-rest.csv, 210 games, Oct 20 - Nov 17): Brier
    0.2110 -> 0.2115 (+0.0005 ± 0.0059, +0.1 SE), log loss +0.0041 (+0.3 SE), winners 67.6% -> 70.0%,
    spread MAE 11.96 -> 12.23 (+1.0 SE), slope 1.067 -> 0.748. No gain on these games. In the backtest
    seasons the gain is largest in the first 210 games (-0.0077 ± 0.0048 and -0.0155 ± 0.0053), so
    2026-27 so far sits 1.2-2.0 SE below what the backtests lead one to expect: inconclusive, not
    contradicted. Re-run the replay + --evaluate as the season goes on (2 s + 40 s).
"""
import argparse
import copy
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

import backtest
import dynamic_model
import fit_calib
import params
import predlog
import report
import simulation
import update
from schedule import NBA_TEAMS

FIT_SEASONS = ("2024-25",)
TEST_SEASONS = ("2025-26",)
GRID_N = (1, 2, 3, 5, 8, 12, 20)
GRID_M = (10, 14, 18, 24, 30, 40)
BOOTSTRAP = 2000
EARLY, LATE = ("10", "11", "12"), ("02", "03", "04")   # months: Oct-Dec, Feb-Apr


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def margin_slope(proj, margin) -> dict:
    """OLS slope of the actual margin on the projected one (with an intercept), with its SE."""
    proj, margin = np.asarray(proj, dtype=float), np.asarray(margin, dtype=float)
    if proj.size < 3:
        raise ValueError("need at least three games for a slope")
    x = np.column_stack([np.ones_like(proj), proj])
    beta, *_ = np.linalg.lstsq(x, margin, rcond=None)
    resid = margin - x @ beta
    cov = np.linalg.inv(x.T @ x) * float(resid @ resid) / (proj.size - 2)
    return {"slope": float(beta[1]), "se": math.sqrt(cov[1, 1])}


def paired_slope(proj_old, proj_new, margin, n_boot: int = BOOTSTRAP, seed: int = 0) -> dict:
    """new - old margin slope on the same games, SE from a paired bootstrap over games."""
    proj_old, proj_new = np.asarray(proj_old, dtype=float), np.asarray(proj_new, dtype=float)
    margin = np.asarray(margin, dtype=float)
    diff = margin_slope(proj_new, margin)["slope"] - margin_slope(proj_old, margin)["slope"]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, margin.size, margin.size)
        boots.append(margin_slope(proj_new[i], margin[i])["slope"] - margin_slope(proj_old[i], margin[i])["slope"])
    se = float(np.std(boots, ddof=1))
    return {"diff": diff, "se": se, "z": diff / se if se > 0 else float("nan")}


def metrics(rows: list) -> dict:
    """report.evaluate of backtest / log rows plus the margin slope and the profiled temperature."""
    p = np.array([r["p_model"] for r in rows])
    y = np.array([r["home_won"] for r in rows], dtype=float)
    m = report.evaluate(p, y, [r["proj_margin"] for r in rows], [r["margin"] for r in rows])
    m["slope"] = margin_slope([r["proj_margin"] for r in rows], [r["margin"] for r in rows])["slope"]
    m["temperature"] = fit_calib.fit_temperature(p, y)["temperature"]
    return m


def compare(old_rows: list, new_rows: list, old_a=None, new_a=None) -> dict:
    """
    Old vs new on the same games (matched by game_id, in old's order): each one's metrics (own projections)
    and the paired new - old differences. old_a / new_a: also compare both with the win % calibrated at those
    temperatures (Phase 10b; fit_calib.calibrated), as "old_calib" / "new_calib".
    """
    new_by_id = {r["game_id"]: r for r in new_rows}
    pairs = [(r, new_by_id[r["game_id"]]) for r in old_rows if r["game_id"] in new_by_id]
    if len(pairs) < 3:
        raise ValueError("old and new have fewer than three games in common")
    for o, n in pairs:
        if o["margin"] != n["margin"]:
            raise ValueError(f"game {o['game_id']}: old and new disagree on the final score")
    y = np.array([o["home_won"] for o, _ in pairs], dtype=float)
    margin = np.array([o["margin"] for o, _ in pairs], dtype=float)
    proj = {"old": [o["proj_margin"] for o, _ in pairs], "new": [n["proj_margin"] for _, n in pairs]}
    probs = {"old": np.array([o["p_model"] for o, _ in pairs]), "new": np.array([n["p_model"] for _, n in pairs])}
    out = {"n": len(pairs), "slope_diff": paired_slope(proj["old"], proj["new"], margin)}
    variants = [("old", "old", None), ("new", "new", None)]
    if old_a is not None and new_a is not None:
        variants += [("old_calib", "old", old_a), ("new_calib", "new", new_a)]
    for name, base, a in variants:
        p = probs[base] if a is None else fit_calib.calibrated(probs[base], a)
        m = report.evaluate(p, y, proj[base], margin)
        m["slope"] = margin_slope(proj[base], margin)["slope"]
        m["each"]["gap"] = m["each"]["confidence"] - m["each"]["hit"]
        out[name] = m
    for new, old in (("new", "old"), ("new_calib", "old_calib")):
        if new not in out:
            continue
        out[f"{new}-{old}"] = {k: report.paired(out[new]["each"][key], out[old]["each"][key]) for k, key in
                               (("accuracy", "hit"), ("brier", "brier"), ("log_loss", "log_loss"),
                                ("margin_mae", "margin_err"), ("conf_minus_acc", "gap"))}
    return out


def format_compare(label: str, c: dict) -> str:
    lines = [f"{label}: {c['n']} games",
             f"  {'':<11}{'winners':>9}{'confidence':>12}{'conf-acc':>10}{'Brier':>9}{'log loss':>10}"
             f"{'spread MAE':>12}{'slope':>8}"]
    for name in ("old", "new", "old_calib", "new_calib"):
        if name not in c:
            continue
        m = c[name]
        lines.append(f"  {name:<11}{m['accuracy']:>9.1%}{m['confidence']:>12.1%}"
                     f"{m['confidence'] - m['accuracy']:>+10.1%}{m['brier']:>9.4f}{m['log_loss']:>10.4f}"
                     f"{m['margin_mae']:>12.2f}{m['slope']:>8.3f}")
    for key in ("new-old", "new_calib-old_calib"):
        if key not in c:
            continue
        cells = []
        for k, digits, scale in (("accuracy", 1, 100), ("brier", 4, 1), ("log_loss", 4, 1), ("margin_mae", 2, 1),
                                 ("conf_minus_acc", 1, 100)):
            d = c[key][k]
            cells.append(f"{k} {scale * d['diff']:+.{digits}f} ± {scale * d['se']:.{digits}f}"
                         + (f" ({d['z']:+.1f} SE)" if not math.isnan(d["z"]) else ""))
        if key == "new-old":
            s = c["slope_diff"]
            cells.append(f"slope {s['diff']:+.3f} ± {s['se']:.3f} ({s['z']:+.1f} SE)")
        lines.append(f"  {key}: " + "; ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the grid (2024-25 only)
# ---------------------------------------------------------------------------

_worker = {}


def _load(csv_path, seasons) -> dict:
    """{season: (preseason priors, games)} for the live-procedure backtest."""
    data = dynamic_model.load_training_rows(csv_path, dynamic_model.TRAINING_CUTOFF, 3)
    return {s: (backtest.preseason_priors(csv_path, s)["priors"], backtest.season_games(data, s)[0]) for s in seasons}


def _init_worker(csv_path, seasons):
    _worker.update(_load(csv_path, seasons))


def _grid_point(args) -> dict:
    n, m = args
    rows = []
    for pre, games in _worker.values():
        rows += backtest.run_season(pre, games, n=n, m=m)
    s = metrics(rows)
    return {"n": n, "m": m, "brier": s["brier"], "log_loss": s["log_loss"], "accuracy": s["accuracy"],
            "confidence": s["confidence"], "margin_mae": s["margin_mae"], "slope": s["slope"],
            "temperature": s["temperature"]}


def run_grid(csv_path, seasons=FIT_SEASONS, ns=GRID_N, ms=GRID_M, workers: int = 3, loaded=None) -> list:
    """Every (N, M) on the seasons' live-procedure backtest; one dict per point. loaded: _load's result (tests)."""
    points = [(n, m) for n in ns for m in ms]
    if loaded is not None or workers <= 1:
        _worker.clear()
        _worker.update(loaded if loaded is not None else _load(csv_path, seasons))
        return [_grid_point(p) for p in points]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(csv_path), tuple(seasons))) as ex:
        return list(ex.map(_grid_point, points))


def best_point(grid: list) -> dict:
    """The lowest Brier (ties: lower log loss)."""
    return min(grid, key=lambda g: (round(g["brier"], 10), g["log_loss"]))


def format_grid(grid: list, key: str, digits: int = 4) -> str:
    ns, ms = sorted({g["n"] for g in grid}), sorted({g["m"] for g in grid})
    cell = {(g["n"], g["m"]): g[key] for g in grid}
    lines = [f"{key}  (rows N, columns M)", "  N \\ M " + "".join(f"{m:>9}" for m in ms)]
    for n in ns:
        lines.append(f"  {n:>5} " + "".join(f"{cell[(n, m)]:>9.{digits}f}" if (n, m) in cell else f"{'':>9}"
                                            for m in ms))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------

def league_average(priors: dict) -> dict:
    return {side: {f: float(np.mean([priors[t][side][f]["mean"] for t in NBA_TEAMS])) for f in params.FACTORS}
            for side in ("offense", "defense")}


def net_rating(priors: dict, team: str, avg: dict) -> float:
    """A team's points per 100 possessions minus its opponent's, vs a league-average team (mean factors, no noise)."""
    o, d = priors[team]["offense"], priors[team]["defense"]
    c = params.points_params()
    scored = simulation.points({f: (o[f]["mean"] + avg["defense"][f]) / 2 for f in params.FACTORS}, 100, c)
    allowed = simulation.points({f: (avg["offense"][f] + d[f]["mean"]) / 2 for f in params.FACTORS}, 100, c)
    return float(scored - allowed)


def stretched(priors: dict, k: float) -> dict:
    """The NBA teams' factor means moved k times as far from the league average (k = 1: unchanged)."""
    out = copy.deepcopy(priors)
    avg = league_average(priors)
    for t in NBA_TEAMS:
        for side in ("offense", "defense"):
            for f in params.FACTORS:
                out[t][side][f]["mean"] = avg[side][f] + k * (priors[t][side][f]["mean"] - avg[side][f])
    return out


def run_variant(pre: dict, games: list, n: float = update.DEFAULT_N, m: float = update.DEFAULT_M,
                sigma_scale: float = 1.0, stretch: float = 1.0, extra: bool = False) -> list:
    """
    backtest.run_season with diagnosis knobs: the update sigma scaled by sigma_scale, the preseason means
    stretched (stretched(pre, k)); n / m may be fractional. extra: each row also gets both teams' net
    ratings before the game ("net_home", "net_away") and the projection of the never-updated reset priors
    ("proj_pre").
    """
    priors = update.reset_priors(stretched(pre, stretch) if stretch != 1 else pre, n)
    frozen = copy.deepcopy(priors) if extra else None
    usig = {f: v * sigma_scale for f, v in params.update_sigma().items()}
    rows = []
    for game in games:
        h, a = game["home"], game["away"]
        kw = {"seed": predlog.PREDICT_SEED, "neutral": game["neutral"], "rest_a": game["rest"]["home"],
              "rest_b": game["rest"]["away"]}
        sim = simulation.simulate_factors(h["id"], a["id"], priors, **kw)
        margin = h["pts"] - a["pts"]
        row = {"game_id": game["game_id"], "date": game["date"], "home_won": margin > 0, "margin": margin,
               "p_model": sim["team_a_win_pct"] / 100,
               "proj_margin": sim["team_a_avg_score"] - sim["team_b_avg_score"]}
        if extra:
            avg = league_average(priors)
            row["net_home"], row["net_away"] = net_rating(priors, h["id"], avg), net_rating(priors, a["id"], avg)
            pre_sim = simulation.simulate_factors(h["id"], a["id"], frozen, **kw)
            row["proj_pre"] = pre_sim["team_a_avg_score"] - pre_sim["team_b_avg_score"]
        rows.append(row)
        update.apply_game(priors, game, m, sigma=usig)
    return rows


def factor_means(priors: dict) -> np.ndarray:
    return np.array([priors[t][side][f]["mean"] for t in sorted(NBA_TEAMS) for side in ("offense", "defense")
                     for f in params.FACTORS])


def sigma_equivalence(pre: dict, games: list, c: float = 0.8, m: float = update.DEFAULT_M,
                      n: float = update.DEFAULT_N) -> float:
    """
    Largest difference in final factor means between update sigma x c and (N c^2, M c) with the fitted
    sigma, over a season: 0 up to rounding, since the exact update depends only on variance ratios.
    """
    a = update.reset_priors(pre, n)
    b = update.reset_priors(pre, n * c ** 2)
    scaled = {f: v * c for f, v in params.update_sigma().items()}
    for game in games:
        update.apply_game(a, game, m, sigma=scaled, pace="off")
        update.apply_game(b, game, m * c, pace="off")
    return float(np.max(np.abs(factor_means(a) - factor_means(b))))


def by_month(rows: list) -> list:
    out = []
    for month in sorted({r["date"][:7] for r in rows}):
        rr = [r for r in rows if r["date"][:7] == month]
        if len(rr) < 30:
            continue
        live = margin_slope([r["proj_margin"] for r in rr], [r["margin"] for r in rr])
        pre = margin_slope([r["proj_pre"] for r in rr], [r["margin"] for r in rr])
        out.append({"month": month, "n": len(rr), "slope": live["slope"], "se": live["se"],
                    "pre_slope": pre["slope"], "pre_se": pre["se"]})
    return out


def by_rating(rows: list, cuts=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) -> dict:
    """Each team-game's net rating vs its margin miss (actual - projected, from its side), in rating bins."""
    team_rows = np.array([(r["net_home"], r["margin"] - r["proj_margin"]) for r in rows]
                         + [(r["net_away"], r["proj_margin"] - r["margin"]) for r in rows])
    edges = np.quantile(team_rows[:, 0], cuts)
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (team_rows[:, 0] >= lo) & ((team_rows[:, 0] < hi) if i < len(edges) - 2 else (team_rows[:, 0] <= hi))
        miss = team_rows[sel, 1]
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "rating": float(team_rows[sel, 0].mean()),
                     "miss": float(miss.mean()), "se": float(miss.std(ddof=1) / math.sqrt(sel.sum()))})
    fit = margin_slope(team_rows[:, 0], team_rows[:, 1])
    return {"bins": bins, "per_point": fit["slope"], "per_point_se": fit["se"]}


def months_slope(rows: list, months) -> float:
    rr = [r for r in rows if r["date"][5:7] in months]
    return margin_slope([r["proj_margin"] for r in rr], [r["margin"] for r in rr])["slope"]


SENSITIVITY = (("live (N=20, M=30)", {}), ("N=5", {"n": 5}), ("N=10", {"n": 10}), ("M=18", {"m": 18}),
               ("M=12", {"m": 12}), ("update sigma x0.8", {"sigma_scale": 0.8}),
               ("N=12.8, M=24 (= sigma x0.8)", {"n": 12.8, "m": 24}), ("preseason means x1.2", {"stretch": 1.2}),
               ("preseason means x0.8", {"stretch": 0.8}))


def diagnose(csv_path=None, loaded=None) -> dict:
    csv_path = csv_path or dynamic_model.DEFAULT_CSV
    loaded = loaded or _load(csv_path, backtest.SEASONS)
    out = {"seasons": {}, "sensitivity": []}
    for season, (pre, games) in loaded.items():
        rows = run_variant(pre, games, extra=True)
        out["seasons"][season] = {"slope": margin_slope([r["proj_margin"] for r in rows], [r["margin"] for r in rows]),
                                  "early": months_slope(rows, EARLY), "late": months_slope(rows, LATE),
                                  "months": by_month(rows), "rating": by_rating(rows)}
        out.setdefault("rows", []).extend(rows)
    both = out.pop("rows")
    out["both"] = {"slope": margin_slope([r["proj_margin"] for r in both], [r["margin"] for r in both]),
                   "early": months_slope(both, EARLY), "late": months_slope(both, LATE), "rating": by_rating(both)}
    pre, games = loaded[FIT_SEASONS[0]]
    for label, kw in SENSITIVITY:
        rows = run_variant(pre, games, **kw)
        s = metrics(rows)
        out["sensitivity"].append({"label": label, "slope": s["slope"], "early": months_slope(rows, EARLY),
                                   "late": months_slope(rows, LATE), "brier": s["brier"], "log_loss": s["log_loss"],
                                   "accuracy": s["accuracy"], "margin_mae": s["margin_mae"]})
    out["sigma_equivalence"] = sigma_equivalence(pre, games)
    return out


def format_diagnosis(d: dict) -> str:
    L = []
    for season, s in d["seasons"].items():
        L.append(f"{season}: slope of actual on projected margin {s['slope']['slope']:.3f} ± {s['slope']['se']:.3f} "
                 f"(Oct-Dec {s['early']:.3f}, Feb-Apr {s['late']:.3f})")
        L.append(f"  {'month':<9}{'games':>6}{'live slope':>16}{'preseason-only':>18}")
        L += [f"  {r['month']:<9}{r['n']:>6}{r['slope']:>9.3f} ± {r['se']:.3f}{r['pre_slope']:>11.3f} ± {r['pre_se']:.3f}"
              for r in s["months"]]
    b = d["both"]
    L.append(f"\nboth seasons: slope {b['slope']['slope']:.3f} ± {b['slope']['se']:.3f} (Oct-Dec {b['early']:.3f}, "
             f"Feb-Apr {b['late']:.3f})")
    L.append("  team net rating (pts/100 vs average) -> mean margin miss from its side (actual - projected):")
    L += [f"    {x['lo']:+6.1f} to {x['hi']:+6.1f}  {x['n']:>5} team-games  rating {x['rating']:+6.2f}  "
          f"miss {x['miss']:+.2f} ± {x['se']:.2f}" for x in b["rating"]["bins"]]
    L.append(f"    miss per rating point: {b['rating']['per_point']:+.3f} ± {b['rating']['per_point_se']:.3f}")
    L.append(f"\none knob at a time, {FIT_SEASONS[0]}:")
    L.append(f"  {'':<30}{'slope':>7}{'Oct-Dec':>9}{'Feb-Apr':>9}{'Brier':>9}{'log loss':>10}{'winners':>9}{'MAE':>8}")
    L += [f"  {s['label']:<30}{s['slope']:>7.3f}{s['early']:>9.3f}{s['late']:>9.3f}{s['brier']:>9.4f}"
          f"{s['log_loss']:>10.4f}{s['accuracy']:>9.1%}{s['margin_mae']:>8.2f}" for s in d["sensitivity"]]
    L.append(f"  update sigma x0.8 vs N x0.64, M x0.8: largest factor-mean difference after the season "
             f"{d['sigma_equivalence']:.1e}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def write_section(path: Path, section: dict) -> None:
    model_params = json.loads(path.read_text(encoding="utf-8"))
    model_params["react"] = section
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(model_params, indent=4) + "\n", encoding="utf-8")
    tmp.replace(path)


def fit(csv_path, workers: int = 3, loaded=None, ns=GRID_N, ms=GRID_M) -> dict:
    """The grid on FIT_SEASONS, the best point, and the temperature refitted there. Returns the react section."""
    grid = run_grid(csv_path, FIT_SEASONS, ns, ms, workers=workers, loaded=loaded)
    best = best_point(grid)
    pre_games = loaded or _load(csv_path, FIT_SEASONS)
    rows = []
    for season in FIT_SEASONS:
        pre, games = pre_games[season]
        rows += backtest.run_season(pre, games, n=best["n"], m=best["m"])
    p = np.array([r["p_model"] for r in rows])
    t = fit_calib.fit_temperature(p, [r["home_won"] for r in rows])
    live = next((g for g in grid if (g["n"], g["m"]) == (update.DEFAULT_N, update.DEFAULT_M)), None)
    return {
        "n": best["n"], "m": best["m"],
        "calib_temperature": round(t["temperature"], 4), "calib_temperature_se": round(t["se"], 4),
        "method": "season reset N (std 0.1/sqrt(N)) and drift M (tau 0.1/M) of the live in-season procedure, "
                  "picked on a grid by the lowest one-step-ahead Brier; calib_temperature = the Phase 10b "
                  "temperature refitted with them (maximum likelihood)",
        "fit": {"data": f"live-procedure backtest (backtest.run_season), season(s) {', '.join(FIT_SEASONS)} only; "
                        f"{', '.join(TEST_SEASONS)} and 2026-27 are test sets",
                "games": len(rows), "grid_n": list(ns), "grid_m": list(ms),
                "best": {k: round(v, 5) if isinstance(v, float) else v for k, v in best.items()},
                "live": {k: round(v, 5) if isinstance(v, float) else v for k, v in live.items()} if live else None,
                "grid": [{k: round(v, 5) if isinstance(v, float) else v for k, v in g.items()} for g in grid],
                "fitted": date.today().isoformat()},
    }


def evaluate(csv_path, old_log=None, new_log=None) -> str:
    old_a, new_a = params.calib_temperature("off"), params.calib_temperature("on")
    s = params.react_settings()
    loaded = _load(csv_path, TEST_SEASONS)
    old_rows, new_rows = [], []
    for season in TEST_SEASONS:
        pre, games = loaded[season]
        old_rows += backtest.run_season(pre, games)
        new_rows += backtest.run_season(pre, games, react="on")
    head = (f"react: N={s['n']}, M={s['m']} vs live N={update.DEFAULT_N}, M={update.DEFAULT_M}; calibration "
            f"temperature {old_a:.4f} (live) / {new_a:.4f} (react)\n")
    out = [head, format_compare(f"{', '.join(TEST_SEASONS)} live-procedure backtest (test)",
                                compare(old_rows, new_rows, old_a, new_a))]
    if old_log:
        rows = {}
        for key, path in (("old", old_log), ("new", new_log)):
            rows[key] = [report.log_row_to_game(r) for r in predlog.read_log(path)]
        out.append("\n" + format_compare(f"2026-27: {Path(old_log).name} vs {Path(new_log).name} (test)",
                                         compare(rows["old"], rows["new"], old_a, new_a)))
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit / test the rating reactivity N, M (Phase 10c).")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--params", default=str(params.PARAMS_PATH), help="model_params.json to add the section to")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing react section")
    parser.add_argument("--workers", type=int, default=3, help="parallel processes for the grid (default 3)")
    parser.add_argument("--diagnose", action="store_true", help="print the diagnosis only (both backtest seasons)")
    parser.add_argument("--evaluate", action="store_true",
                        help="old vs new on the test sets: 2025-26 (backtest) and, with --old-log/--new-log, 2026-27")
    parser.add_argument("--old-log", help="with --evaluate: the live model's 2026-27 log (e.g. log_pace-home-rest.csv)")
    parser.add_argument("--new-log", help="with --evaluate: the same replay with --react on (e.g. log_react.csv)")
    args = parser.parse_args(argv)
    if bool(args.old_log) != bool(args.new_log) or (args.old_log and not args.evaluate):
        parser.error("--old-log and --new-log go together, with --evaluate")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    start = time.perf_counter()

    if args.diagnose:
        print(format_diagnosis(diagnose(args.csv)))
    elif args.evaluate:
        print(evaluate(args.csv, args.old_log, args.new_log))
    else:
        out = Path(args.params)
        if "react" in json.loads(out.read_text(encoding="utf-8")) and not args.overwrite:
            parser.error(f"{out.name} already has a react section; pass --overwrite to refit")
        section = fit(args.csv, workers=args.workers)
        grid = section["fit"]["grid"]
        for key, digits in (("brier", 4), ("log_loss", 4), ("accuracy", 4), ("slope", 3), ("temperature", 3)):
            print(format_grid(grid, key, digits) + "\n")
        write_section(out, section)
        b, live = section["fit"]["best"], section["fit"]["live"]
        print(f"Best on {', '.join(FIT_SEASONS)} ({section['fit']['games']} games): N={section['n']}, M={section['m']}: "
              f"Brier {b['brier']:.4f}, log loss {b['log_loss']:.4f}, slope {b['slope']:.3f}"
              + (f" (live N={update.DEFAULT_N}, M={update.DEFAULT_M}: {live['brier']:.4f}, {live['log_loss']:.4f}, "
                 f"{live['slope']:.3f})" if live else "")
              + f"; calibration temperature refitted: {section['calib_temperature']} "
                f"(SE {section['calib_temperature_se']}) -> {out.name}")
    print(f"({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
