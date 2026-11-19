"""
How well calibrated is each simulation setting? Checked on the TRAINING data (no 2026-27 game).

    python code/train_calibration.py                                   # every variant, home on (the default)
    python code/train_calibration.py --home off on                     # ... and again without the home boost
    python code/train_calibration.py --variants old-live live --home off on   # old vs new live simulation
    python code/train_calibration.py --variants live --rest off on      # the live model without / with rest

It runs the exact training pass over the Kaggle window (flat start, tau = 0, fitted update
sigma, plus team paces), and before each NBA-vs-NBA game after the first season it simulates
that game with every variant from the curves as they stood: one-step-ahead forecasts, like
fit_sigma.py, for ~2,600 games instead of this season's ~200. The home team is team_a.
With --home on, both the training pass and the simulation use the home boost
(model_params.json["home"]; none at neutral sites), so each home setting is its own pass.
Likewise --rest on uses the rest effect (model_params.json["rest"], both teams' rest from the
training seasons' NBA.com schedules) in the training pass and the simulation; each
(home, rest) combination is its own pass.

Variants (the simulation settings, every switch spelled out so the params.py defaults don't
leak in; the update is always the same):
    old-live     legacy sim sigma, sim_corr off, pace off (the live simulation before pace-home;
                 with home off: the whole Phase 6 live model)
    corr         fitted sim sigma, sim_corr on, pace off
    pace         legacy sim sigma, sim_corr off, pace on
    corr-pace    fitted sim sigma, sim_corr on, pace on
    live         fitted sim sigma, sim_corr off, pace on (the live simulation; = the old "pace-fitted";
                 with home on: the whole live model)
    fitted       fitted sim sigma, sim_corr off, pace off

Per (variant, home) it prints Brier, winner accuracy, average confidence and their gap,
projected vs actual total and home margin, the simulated vs real margin SD, the slope of
actual on projected margin (1 = right scale), and the paired Brier difference vs the first
column with its standard error. Prints only; writes nothing (the training seasons'
schedules are downloaded into stats/schedules/ once, for --home on).
"""
import argparse
import math
import time
from datetime import date

import numpy as np

import bayes_update
import dynamic_model
import params
import simulation
from schedule import NBA_TEAMS
from state_manager import default_priors

VARIANTS = {
    "old-live": {"sigma": "legacy", "sim_corr": "off", "pace": "off"},
    "corr": {"sigma": "fitted", "sim_corr": "on", "pace": "off"},
    "pace": {"sigma": "legacy", "sim_corr": "off", "pace": "on"},
    "corr-pace": {"sigma": "fitted", "sim_corr": "on", "pace": "on"},
    "live": {"sigma": "fitted", "sim_corr": "off", "pace": "on"},
    "fitted": {"sigma": "fitted", "sim_corr": "off", "pace": "off"},
}
BURN_IN_DAYS = 365


def run_pass(data: dict, home: str, variants: list, rest: str = params.DEFAULT_REST_MODE,
             with_ids: bool = False) -> dict:
    """
    {variant: [(p_home, actual home margin, projected margin, projected total, actual total, sim margin SD)]}
    for every scored game, simulated with `home` and `rest` from the one-step-ahead curves of the training pass.
    with_ids=True appends the game's 10-digit NBA.com gameId to each record (report.py matches Vegas lines by it).
    """
    cleaned = data["cleaned"]
    with_home, with_rest = params.is_on(home, "home"), params.is_on(rest, "rest")
    sides = dynamic_model.row_sides(data) if with_home else None
    offsets = dynamic_model.row_offsets(sides) if with_home else [None] * len(data["rows"])
    rest_rows = dynamic_model.row_rest(data) if with_rest else [None] * len(data["rows"])
    if with_rest:
        offsets = [params.combine_offsets(h, r) for h, r in zip(offsets, dynamic_model.row_rest_offsets(rest_rows))]
    first_day = cleaned["gameDateTimeEst"].min().toordinal()

    by_game = {}
    for i, ((_, row), (gid, when, home_flag, score)) in enumerate(zip(
            data["rows"].iterrows(),
            zip(cleaned["gameId"], cleaned["gameDateTimeEst"], cleaned["home"], cleaned["teamScore"]))):
        g = dynamic_model.row_to_game(row)
        g = {**g, **{f: float(g[f]) for f in params.FACTORS}}
        entry = by_game.setdefault(str(gid), {"rows": [], "when": when})
        entry["rows"].append((g, offsets[i], int(home_flag) == 1, float(score), sides[i] if sides else None,
                              rest_rows[i]))
    possessions = {g["game_id"]: g for g in data["possessions"]}

    pp = params.pace_params()
    priors = default_priors(data["team_ids"])
    dynamic_model.start_pace(priors, data["team_ids"], pp["start_mean"], pp["start_std"])
    update_sigma = params.update_sigma("fitted")
    settings = {v: {**VARIANTS[v], "sigma": params.sim_sigma(VARIANTS[v]["sigma"])} for v in variants}
    out = {v: [] for v in variants}

    for k, (gid, entry) in enumerate(by_game.items()):
        rows = entry["rows"]
        homes = [r for r in rows if r[2]]
        scored = (len(rows) == 2 and len(homes) == 1 and gid in possessions
                  and rows[0][0]["team"] in NBA_TEAMS and rows[0][0]["opponent"] in NBA_TEAMS
                  and entry["when"].toordinal() >= first_day + BURN_IN_DAYS)
        if scored:
            home_row = homes[0]
            away_row = next(r for r in rows if not r[2])
            home_id, away_id = home_row[0]["team"], away_row[0]["team"]
            neutral = with_home and home_row[4] is None
            margin, total = home_row[3] - away_row[3], home_row[3] + away_row[3]
            days = {"rest_a": home_row[5][0]["rest"], "rest_b": home_row[5][1]["rest"]} if with_rest else {}
            for v, kw in settings.items():
                r = simulation.simulate_factors(home_id, away_id, priors, seed=k, home=home, neutral=neutral,
                                                rest=rest, **days, **kw)
                record = (r["team_a_win_pct"] / 100, margin, r["team_a_avg_score"] - r["team_b_avg_score"],
                          r["team_a_avg_score"] + r["team_b_avg_score"], total, r["margin_sd"])
                out[v].append(record + (str(int(gid)).zfill(10),) if with_ids else record)
        for g, offset, *_ in rows:
            bayes_update.update_game(priors, g, tau=dynamic_model.TRAINING_TAU, engine="exact", sigma=update_sigma,
                                     offset=offset)
        if gid in possessions:
            p = possessions[gid]
            bayes_update.update_pace(priors, p["team"], p["opponent"], p["possessions"], sigma=pp["sigma"])
    return out


def summarize(records: list) -> dict:
    p, margin, proj, proj_total, total, sim_sd = (np.array(x, dtype=float) for x in zip(*records))
    favored = np.where(p >= 0.5, p, 1 - p)
    hit = np.where(p >= 0.5, margin > 0, margin < 0)
    return {"brier_each": (favored - hit) ** 2, "accuracy": hit.mean(), "confidence": favored.mean(),
            "proj_total": proj_total.mean(), "actual_total": total.mean(),
            "proj_home": proj.mean(), "actual_home": margin.mean(),
            "sim_sd": sim_sd.mean(), "real_sd": np.std(margin - proj), "slope": (proj @ margin) / (proj @ proj)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibration of the simulation settings on the training data.")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                        help="simulation settings to check (default all)")
    parser.add_argument("--home", nargs="+", choices=params.SWITCH_MODES, default=[params.DEFAULT_HOME_MODE],
                        help="home boost setting(s); each is its own training pass "
                             f"(default {params.DEFAULT_HOME_MODE})")
    parser.add_argument("--rest", nargs="+", choices=params.SWITCH_MODES, default=[params.DEFAULT_REST_MODE],
                        help="rest effect setting(s); each is its own training pass "
                             f"(default {params.DEFAULT_REST_MODE})")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--before", default=dynamic_model.TRAINING_CUTOFF, metavar="YYYY-MM-DD")
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error("--before must be a date like 2026-10-03")
    if before > dynamic_model.TRAINING_CUTOFF:
        parser.error(f"--before must be on or before {dynamic_model.TRAINING_CUTOFF}: 2026-27 games are the test set")

    start = time.perf_counter()
    data = dynamic_model.load_training_rows(args.csv, before, args.years)
    columns = {}
    for home in dict.fromkeys(args.home):
        for rest in dict.fromkeys(args.rest):
            for v, records in run_pass(data, home, args.variants, rest).items():
                columns[f"{v}, home {home}" + ("" if args.rest == [params.DEFAULT_REST_MODE] else f", rest {rest}")] = \
                    summarize(records)
    n = len(next(iter(columns.values()))["brier_each"])
    print(f"{n} NBA training games scored ({data['first']} to {data['last']}, after the first season; "
          f"one-step-ahead) in {time.perf_counter() - start:.0f} s\n")

    base_name = next(iter(columns))
    base = columns[base_name]["brier_each"]
    width = max(24, max(len(c) for c in columns) + 2)
    print(f"{'':<{width}}{'Brier':>8}{'acc':>8}{'conf':>8}{'conf-acc':>9}{'proj tot':>9}{'actual':>8}"
          f"{'proj home':>10}{'actual':>8}{'sim SD':>8}{'real SD':>8}{'slope':>7}   Brier vs {base_name}")
    for name, s in columns.items():
        d = s["brier_each"] - base
        se = d.std(ddof=1) / math.sqrt(len(d))
        versus = "" if name == base_name else f"{d.mean():+.4f} ({d.mean() / se:+.1f} SE)"
        print(f"{name:<{width}}{s['brier_each'].mean():>8.4f}{s['accuracy']:>8.1%}{s['confidence']:>8.1%}"
              f"{s['confidence'] - s['accuracy']:>+9.1%}{s['proj_total']:>9.1f}{s['actual_total']:>8.1f}"
              f"{s['proj_home']:>+10.2f}{s['actual_home']:>+8.2f}{s['sim_sd']:>8.1f}{s['real_sd']:>8.1f}"
              f"{s['slope']:>7.2f}   {versus}")


if __name__ == "__main__":
    main()
