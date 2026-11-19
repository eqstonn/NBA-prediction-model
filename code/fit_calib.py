"""
Win-probability calibration (Phase 10b): diagnose the live model's underconfidence, fit a one-parameter
fix on 2024-25 only, and test it on 2025-26 and 2026-27.

    python code/fit_calib.py --diagnose          # where the extra uncertainty comes from (~3 min); writes nothing
    python code/fit_calib.py                     # fit on the 2024-25 live-procedure backtest -> model_params.json["calib"]
    python code/fit_calib.py --overwrite         # refit (replaces the calib section)
    python code/fit_calib.py --evaluate --old-log log_pace-home-rest.csv --new-log log_calib.csv
                                                 # old vs new on 2025-26 (backtest) and 2026-27 (logs); writes nothing

The problem
-----------
In the live-procedure backtest (backtest.py, 2,638 games of 2024-25 and 2025-26) the model is
underconfident: average confidence 62.5% vs 66.5% of winners picked, in almost every bin.

Diagnosis (--diagnose; 2026-11-19, both backtest seasons)
---------------------------------------------------------
  * The simulated margin SD is 16.2 points; real margins land 14.6 from the projection (RMS).
  * Split of the simulated margin variance (each source switched off in turn): factor game noise
    94%, rating uncertainty carried into the simulation 6%, pace noise 0%. Scaling the rating
    uncertainty (or pace) can't close the gap: with none at all the SD is still 15.7.
  * The game noise itself is about right, factor by factor: the one-step-ahead residuals of the
    live ratings have 91-98% of the predicted variance (eFG sigma 0.062 vs 0.064, TOV 0.035 vs
    0.036, ORB 0.070 vs 0.072, FTR 0.085 vs 0.085). So no "noise too big" cause.
  * The win probability that would be calibrated is P = Phi(projected margin / 11.9) instead of
    the simulation's ~Phi(projected margin / 16.2). The 16.2 -> 11.9 gap has three parts of
    similar size:
      1. 16.2 -> 14.6: the simulation draws the 8 factor noises independently, but real factor
         errors are correlated so that they partly cancel in the margin (sim_corr on narrows the
         2024-25 simulation to 14.95 vs 14.27 real, but still needs a temperature of 1.24);
      2. 14.6 -> 13.2: projected margins are too small: actual margins are 1.11x the projected
         ones (slope 1.094 in 2024-25, 1.126 in 2025-26): the ratings under-react (an update
         question, N / M / update sigma, not a simulation one);
      3. 13.2 -> 11.9: real margins are fat-tailed (blowouts), so who wins is more predictable
         than a normal with the full RMS says.

The fix
-------
No single cause in the simulation's noise explains the gap, so the fix is a post-hoc temperature
on the probit scale (one parameter):

    P_calibrated = Phi(a * Phi^-1(P_simulated))

The simulation's P is within 0.007 of Phi(projected margin / simulated SD) (2024-25), so this is
the same as narrowing the simulated margin distribution by 1/a for the win % only. It is monotone,
so it never changes the pick (winner accuracy is identical) and it doesn't touch the projected
scores (spread MAE identical). a is fitted by maximum likelihood (Bernoulli, home win) on the
2024-25 live-procedure backtest ONLY, for the live simulation settings (fitted sigma, sim_corr
off, pace, home, rest); 2025-26 and 2026-27 are test sets. SE from the observed information.
"""
import argparse
import copy
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import backtest
import dynamic_model
import params
import predlog
import report
import simulation
import update

FIT_SEASONS = ("2024-25",)
TEST_SEASONS = ("2025-26",)
TINY = 1e-12   # "switched off": a std / sigma this small


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def loglik(q, y) -> float:
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))


def calibrated(p, a: float):
    """Phi(a * Phi^-1(p)) for an array (params.calibrate_win_prob is the scalar used in the simulation)."""
    q = np.clip(np.asarray(p, dtype=float), params.CALIB_CLIP, 1 - params.CALIB_CLIP)
    return norm.cdf(a * norm.ppf(q))


def fit_temperature(p_raw, home_won) -> dict:
    """
    The maximum-likelihood temperature a for P = Phi(a Phi^-1(p_raw)) (bounded to [0.2, 5]) with its SE
    (observed information) and the log-likelihoods before / after.
    """
    p_raw, y = np.asarray(p_raw, dtype=float), np.asarray(home_won, dtype=float)
    if p_raw.size < 2:
        raise ValueError("need at least two games to fit the temperature")
    res = minimize_scalar(lambda a: -loglik(calibrated(p_raw, a), y), bounds=(0.2, 5), method="bounded",
                          options={"xatol": 1e-7})
    a, h = float(res.x), 1e-4
    info = -(loglik(calibrated(p_raw, a + h), y) - 2 * loglik(calibrated(p_raw, a), y)
             + loglik(calibrated(p_raw, a - h), y)) / h ** 2
    return {"temperature": a, "se": 1 / math.sqrt(info) if info > 0 else float("nan"), "games": int(p_raw.size),
            "loglik_before": loglik(p_raw, y), "loglik_after": loglik(calibrated(p_raw, a), y)}


# ---------------------------------------------------------------------------
# evaluation: old vs new on the same games
# ---------------------------------------------------------------------------

def compare(old_p, new_p, home_won, proj_margin, margin) -> dict:
    """report.evaluate for both, plus paired new - old differences (Brier, log loss, accuracy, margin MAE)."""
    old = report.evaluate(old_p, home_won, proj_margin, margin)
    new = report.evaluate(new_p, home_won, proj_margin, margin)
    return {"n": old["n"], "old": old, "new": new,
            **{k: report.paired(new["each"][key], old["each"][key]) for k, key in
               (("accuracy", "hit"), ("brier", "brier"), ("log_loss", "log_loss"), ("margin_mae", "margin_err"))}}


def format_compare(label: str, c: dict) -> str:
    lines = [f"{label}: {c['n']} games"]
    lines.append(f"  {'':<6}{'winners':>9}{'confidence':>12}{'conf-acc':>10}{'Brier':>9}{'log loss':>10}{'spread MAE':>12}")
    for name in ("old", "new"):
        m = c[name]
        lines.append(f"  {name:<6}{m['accuracy']:>9.1%}{m['confidence']:>12.1%}{m['confidence'] - m['accuracy']:>+10.1%}"
                     f"{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['margin_mae']:>12.2f}")
    cells = []
    for key, digits, scale in (("accuracy", 1, 100), ("brier", 4, 1), ("log_loss", 4, 1), ("margin_mae", 2, 1)):
        d = c[key]
        cells.append(f"{key} {scale * d['diff']:+.{digits}f} ± {scale * d['se']:.{digits}f}"
                     + (f" ({d['z']:+.1f} SE)" if not math.isnan(d["z"]) else ""))
    lines.append("  new - old: " + "; ".join(cells))
    return "\n".join(lines)


def compare_logs(old_path, new_path) -> dict:
    """Old vs new prediction logs on their common games (e.g. log_pace-home-rest.csv vs a --calib on replay)."""
    old = {r["game_id"]: report.log_row_to_game(r) for r in predlog.read_log(old_path)}
    new = {r["game_id"]: report.log_row_to_game(r) for r in predlog.read_log(new_path)}
    ids = [g for g in old if g in new]
    if not ids:
        raise ValueError(f"{Path(old_path).name} and {Path(new_path).name} have no game in common")
    for g in ids:
        if abs(old[g]["proj_margin"] - new[g]["proj_margin"]) > 0.11:
            raise ValueError(f"game {g}: the logs' projected scores differ, so they aren't the same model "
                             "with and without calibration")
    return compare([old[g]["p_model"] for g in ids], [new[g]["p_model"] for g in ids],
                   [old[g]["home_won"] for g in ids], [old[g]["proj_margin"] for g in ids],
                   [old[g]["margin"] for g in ids])


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------

def without_rating_uncertainty(priors: dict, teams) -> dict:
    """A copy of `priors` in which the given teams' factor and pace curves have (almost) no std."""
    out = dict(priors)
    for t in teams:
        c = copy.deepcopy(priors[t])
        for side in ("offense", "defense"):
            c[side] = {f: {"mean": v["mean"], "std": TINY} for f, v in c[side].items()}
        if "pace" in c:
            c["pace"] = {"mean": c["pace"]["mean"], "std": TINY}
        out[t] = c
    return out


def noise_split(priors: dict, game: dict) -> dict:
    """
    The simulated margin SD of one game (live settings, the log's seed) with everything on, and with
    rating uncertainty / factor game noise / pace noise switched off one at a time.
    """
    teams = (game["home"]["id"], game["away"]["id"])
    kw = {"seed": predlog.PREDICT_SEED, "neutral": game["neutral"], "rest_a": game["rest"]["home"],
          "rest_b": game["rest"]["away"]}
    full = simulation.simulate_factors(*teams, priors, **kw)
    return {"proj": full["team_a_avg_score"] - full["team_b_avg_score"], "p": full["team_a_win_pct"] / 100,
            "full": full["margin_sd"],
            "no_rating": simulation.simulate_factors(*teams, without_rating_uncertainty(priors, teams), **kw)["margin_sd"],
            "no_game": simulation.simulate_factors(*teams, priors, sigma={f: TINY for f in params.FACTORS}, **kw)["margin_sd"],
            "no_pace": simulation.simulate_factors(*teams, priors, pace_sigma=TINY, **kw)["margin_sd"]}


def line_residuals(priors: dict, game: dict) -> list:
    """Each line's one-step-ahead (residual, rating variance) per factor, as fit_sigma.py scores them."""
    home_off, away_off = params.home_offsets("on", game["neutral"])
    rest_h, rest_a = params.rest_offsets("on", game["rest"]["home"], game["rest"]["away"])
    out = []
    for side, other, offset in (("home", "away", params.combine_offsets(home_off, rest_h)),
                                ("away", "home", params.combine_offsets(away_off, rest_a))):
        o, d = priors[game[side]["id"]]["offense"], priors[game[other]["id"]]["defense"]
        out.append({f: (game[side][f] - (o[f]["mean"] + d[f]["mean"]) / 2 - (offset[f] if offset else 0.0),
                        (o[f]["std"] ** 2 + d[f]["std"] ** 2) / 4) for f in params.FACTORS})
    return out


def diagnose(csv_path=None, seasons=backtest.SEASONS, data=None) -> dict:
    """The live-procedure backtest with each game's noise split and factor residuals (see the module docstring)."""
    csv_path = csv_path or dynamic_model.DEFAULT_CSV
    data = data or dynamic_model.load_training_rows(csv_path, dynamic_model.TRAINING_CUTOFF, 3)
    usig = params.update_sigma()
    rows, lines = [], []
    for season in seasons:
        games, _ = backtest.season_games(data, season)
        priors = update.reset_priors(backtest.preseason_priors(csv_path, season)["priors"], update.DEFAULT_N)
        for game in games:
            split = noise_split(priors, game)
            rows.append({**split, "season": season, "margin": game["home"]["pts"] - game["away"]["pts"]})
            lines += line_residuals(priors, game)
            update.apply_game(priors, game, update.DEFAULT_M, sigma=usig)
    return summarize_diagnosis(rows, lines)


def summarize_diagnosis(rows: list, lines: list) -> dict:
    margin = np.array([r["margin"] for r in rows], dtype=float)
    proj = np.array([r["proj"] for r in rows])
    sd = {k: np.array([r[k] for r in rows]) for k in ("full", "no_rating", "no_game", "no_pace")}
    var_full = float(np.mean(sd["full"] ** 2))
    share = {name: (var_full - float(np.mean(sd[k] ** 2))) / var_full
             for name, k in (("rating uncertainty", "no_rating"), ("factor game noise", "no_game"),
                             ("pace noise", "no_pace"))}
    y = (margin > 0).astype(float)
    best_s = minimize_scalar(lambda s: -loglik(norm.cdf(proj / s), y), bounds=(4, 30), method="bounded").x
    sigma = params.sim_sigma()
    factors = {}
    for f in params.FACTORS:
        e2 = np.mean([line[f][0] ** 2 for line in lines])
        rv = np.mean([line[f][1] for line in lines])
        factors[f] = {"residual_var": float(e2), "rating_var": float(rv), "sigma": sigma[f],
                      "ratio": float(e2 / (rv + sigma[f] ** 2)), "implied_sigma": float(math.sqrt(max(e2 - rv, 0)))}
    return {"games": len(rows), "sim_sd": math.sqrt(var_full), "real_sd": float(np.sqrt(np.mean((margin - proj) ** 2))),
            "slope": float((proj @ margin) / (proj @ proj)), "calibrated_sd": float(best_s), "share": share,
            "sd_without": {name: float(np.sqrt(np.mean(sd[k] ** 2))) for name, k in
                           (("rating uncertainty", "no_rating"), ("factor game noise", "no_game"),
                            ("pace noise", "no_pace"))},
            "factors": factors}


def format_diagnosis(d: dict) -> str:
    out = [f"{d['games']} games, live-procedure backtest",
           f"  simulated margin SD {d['sim_sd']:.2f}; real margins vs projected: RMS {d['real_sd']:.2f}; "
           f"slope of actual on projected margin {d['slope']:.3f}",
           f"  calibrated win probability ~ Phi(projected margin / {d['calibrated_sd']:.2f})",
           "  simulated margin variance by source (SD without it):"]
    out += [f"    {name:<20}{share:>6.0%}   ({d['sd_without'][name]:.2f})" for name, share in d["share"].items()]
    out.append("  factor level: residual variance / predicted (rating + sigma^2), implied sigma vs used:")
    out += [f"    {f}: {v['ratio']:.3f}   {v['implied_sigma']:.4f} vs {v['sigma']:.4f}" for f, v in d["factors"].items()]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def season_rows(rows: list, seasons) -> list:
    return [r for r in rows if r["season"] in seasons]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit / test the win-probability calibration (Phase 10b).")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--params", default=str(params.PARAMS_PATH), help="model_params.json to add the section to")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing calib section")
    parser.add_argument("--diagnose", action="store_true", help="print the diagnosis only (both backtest seasons)")
    parser.add_argument("--evaluate", action="store_true",
                        help="old vs new on the test sets: 2025-26 (backtest) and, with --old-log/--new-log, 2026-27")
    parser.add_argument("--old-log", help="with --evaluate: the live model's 2026-27 log (e.g. log_pace-home-rest.csv)")
    parser.add_argument("--new-log", help="with --evaluate: the same replay with --calib on (e.g. log_calib.csv)")
    args = parser.parse_args(argv)
    if bool(args.old_log) != bool(args.new_log) or (args.old_log and not args.evaluate):
        parser.error("--old-log and --new-log go together, with --evaluate")
    start = time.perf_counter()

    if args.diagnose:
        print(format_diagnosis(diagnose(args.csv)))
        print(f"({time.perf_counter() - start:.0f} s)")
        return

    if args.evaluate:
        a = params.calib_temperature()
        print(f"temperature a = {a:.4f} (model_params.json, fitted on {', '.join(FIT_SEASONS)})\n")
        rows = backtest.run_backtest(args.csv, seasons=TEST_SEASONS)["rows"]
        p = np.array([r["p_model"] for r in rows])
        print(format_compare(f"{', '.join(TEST_SEASONS)} live-procedure backtest (test)",
                             compare(p, calibrated(p, a), [r["home_won"] for r in rows],
                                     [r["proj_margin"] for r in rows], [r["margin"] for r in rows])))
        if args.old_log:
            print()
            print(format_compare(f"2026-27: {Path(args.old_log).name} vs {Path(args.new_log).name} (test)",
                                 compare_logs(args.old_log, args.new_log)))
        print(f"\n({time.perf_counter() - start:.0f} s)")
        return

    out = Path(args.params)
    model_params = json.loads(out.read_text(encoding="utf-8"))
    if "calib" in model_params and not args.overwrite:
        parser.error(f"{out.name} already has a calib section; pass --overwrite to refit")
    rows = backtest.run_backtest(args.csv, seasons=FIT_SEASONS)["rows"]
    p = np.array([r["p_model"] for r in rows])
    y = np.array([r["home_won"] for r in rows], dtype=float)
    fit = fit_temperature(p, y)
    after = calibrated(p, fit["temperature"])
    hit = np.where(p > 0.5, y, 1 - y).mean()
    model_params["calib"] = {
        "temperature": round(fit["temperature"], 4),
        "temperature_se": round(fit["se"], 4),
        "method": "P_calibrated = Phi(a * Phi^-1(P_simulated)); maximum likelihood (Bernoulli, home win); "
                  "SE from the observed information",
        "fit": {"data": f"live-procedure backtest (backtest.py, N={update.DEFAULT_N}, M={update.DEFAULT_M}), "
                        f"season(s) {', '.join(FIT_SEASONS)} only; {', '.join(TEST_SEASONS)} and 2026-27 are test sets",
                "games": fit["games"], "loglik_before": round(fit["loglik_before"], 2),
                "loglik_after": round(fit["loglik_after"], 2), "accuracy": round(float(hit), 4),
                "confidence_before": round(float(np.mean(np.maximum(p, 1 - p))), 4),
                "confidence_after": round(float(np.mean(np.maximum(after, 1 - after))), 4),
                "simulation": {"sigma_sim": params.DEFAULT_SIM_SIGMA_MODE, "sim_corr": params.DEFAULT_SIM_CORR_MODE,
                               "pace": params.DEFAULT_PACE_MODE, "home": params.DEFAULT_HOME_MODE,
                               "rest": params.DEFAULT_REST_MODE},
                "fitted": date.today().isoformat()},
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(model_params, indent=4) + "\n", encoding="utf-8")
    tmp.replace(out)
    c = model_params["calib"]
    print(f"Fitted on {fit['games']} games ({', '.join(FIT_SEASONS)}): temperature a = {c['temperature']} "
          f"(SE {c['temperature_se']}); log-likelihood {c['fit']['loglik_before']} -> {c['fit']['loglik_after']}; "
          f"confidence {c['fit']['confidence_before']:.1%} -> {c['fit']['confidence_after']:.1%} vs accuracy "
          f"{c['fit']['accuracy']:.1%} -> {out.name} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
