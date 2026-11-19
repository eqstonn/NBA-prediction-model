"""
Estimate the home-court boost from the TRAINING data only and save it in
model_params.json["home"] (read by params.py -> bayes_update.py via update.py /
dynamic_model.py, and simulation.py). Every other section is kept as it is.

    python code/fit_home.py              # Kaggle, 3 years before 2026-10-03 -> model_params.json
    python code/fit_home.py --overwrite  # refit and replace the "home" section

No 2026-27 game is used. Neutral-site games (boxscores.neutral_flags, from NBA.com's
schedules of the training seasons) and All-Star games get no boost.

The boost
---------
For each factor f, a home game shifts the expected value of both offensive lines:
    home team's line:  (o_home + d_away) / 2 + home["offense"][f]
    away team's line:  (o_away + d_home) / 2 + home["defense"][f]
so the 8 numbers are the home team's offense boost and defense boost per factor (the
defense number is what being at home does to the line the home defense allows).

What the data can tell apart
----------------------------
Per factor the data only pin down the GAP between the two lines, delta_f = offense - defense.
A shift common to both lines in every home game is the same thing as a different league-wide
level of the ratings; only neutral-site games could separate the two, and the training window
has 15 of them (after the first season the log-likelihood moves by < 1.5 for a common shift of
+-0.01). So the ratings are taken to be neutral-site strength and the gap is split
evenly: offense = +delta_f / 2, defense = -delta_f / 2.

Method: one-step-ahead (prequential) predictive likelihood, as fit_sigma.py
---------------------------------------------------------------------------
For a candidate delta run the exact training pass (flat start, tau = 0, fitted sigma) with
the boost in both the forecast and the update, and add up each row's log density under its
forecast N((m_o + m_d)/2 + boost, (s_o^2 + s_d^2)/4 + sigma^2). The factors don't interact,
so each gets its own delta: grid -0.04 to 0.04 in steps of 0.002, then 0.0001 steps within
+-0.002 of the best. The standard error comes from the curvature of the log-likelihood.
"""
import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import bayes_update
import dynamic_model
import params
from state_manager import default_priors

FACTORS = params.FACTORS
COARSE = [round(0.002 * i, 4) for i in range(-20, 21)]     # -0.04 .. 0.04
FINE_STEP, FINE_STEPS = 0.0001, 20                         # best +- 20 steps of 0.0001
CURVATURE_STEP = 0.002
LOG_2PI = math.log(2 * math.pi)


def split(delta: dict) -> dict:
    """The 8 curve boosts from the per-factor gap: offense +delta/2, defense -delta/2."""
    return {"offense": {f: delta[f] / 2 for f in FACTORS}, "defense": {f: -delta[f] / 2 for f in FACTORS}}


def training_games(data: dict) -> list:
    """update_game inputs, oldest first, each with "side": "home" / "away" / None (dynamic_model.row_sides)."""
    sides = dynamic_model.row_sides(data)
    games = [dynamic_model.row_to_game(row) for _, row in data["rows"].iterrows()]
    return [{**g, **{f: float(g[f]) for f in FACTORS}, "side": s} for g, s in zip(games, sides)]


def prequential(games: list, team_ids, sigma: dict, boost: dict, score=None):
    """
    ({factor: total log predictive density}, {factor: mean squared standardized residual}) of
    the exact training pass with the home `boost` ({"offense", "defense"}). score(i, game):
    which rows count (default all).
    """
    priors = default_priors(team_ids)
    loglik = {f: 0.0 for f in FACTORS}
    z2 = {f: 0.0 for f in FACTORS}
    n = 0
    for i, g in enumerate(games):
        offset = boost["offense"] if g["side"] == "home" else boost["defense"] if g["side"] == "away" else None
        if score is None or score(i, g):
            offense, defense = priors[g["team"]]["offense"], priors[g["opponent"]]["defense"]
            prior = {"o": {f: (offense[f]["mean"], offense[f]["std"]) for f in FACTORS},
                     "d": {f: (defense[f]["mean"], defense[f]["std"]) for f in FACTORS}}
            for f in FACTORS:
                mean, var = bayes_update.predictive(prior, f, sigma)
                if offset is not None:
                    mean += offset[f]
                r2 = (g[f] - mean) ** 2
                loglik[f] -= 0.5 * (LOG_2PI + math.log(var) + r2 / var)
                z2[f] += r2 / var
            n += 1
        bayes_update.update_game(priors, g, tau=dynamic_model.TRAINING_TAU, engine="exact", sigma=sigma, offset=offset)
    return loglik, {f: z2[f] / max(n, 1) for f in FACTORS}


def estimate_home(games: list, team_ids, sigma: dict) -> dict:
    """Per-factor gap delta maximizing the one-step-ahead log-likelihood, with its standard error."""
    best = {f: (-math.inf, None) for f in FACTORS}

    def consider(delta):
        loglik, _ = prequential(games, team_ids, sigma, split(delta))
        for f in FACTORS:
            if loglik[f] > best[f][0]:
                best[f] = (loglik[f], delta[f])
        return loglik

    for d in COARSE:
        consider({f: d for f in FACTORS})
    centers = {f: best[f][1] for f in FACTORS}
    for k in range(-FINE_STEPS, FINE_STEPS + 1):
        consider({f: round(centers[f] + k * FINE_STEP, 4) for f in FACTORS})

    delta = {f: best[f][1] for f in FACTORS}
    loglik, z2 = prequential(games, team_ids, sigma, split(delta))
    up = prequential(games, team_ids, sigma, split({f: delta[f] + CURVATURE_STEP for f in FACTORS}))[0]
    down = prequential(games, team_ids, sigma, split({f: delta[f] - CURVATURE_STEP for f in FACTORS}))[0]
    none = prequential(games, team_ids, sigma, split({f: 0.0 for f in FACTORS}))[0]
    se = {}
    for f in FACTORS:
        curvature = (up[f] + down[f] - 2 * loglik[f]) / CURVATURE_STEP ** 2
        se[f] = math.sqrt(-1 / curvature) if curvature < 0 else math.inf
    return {"delta": delta, "se": se, "log_likelihood": loglik, "log_likelihood_without_home": none, "mean_z2": z2}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the home-court boost on the training data.")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--before", default=dynamic_model.TRAINING_CUTOFF, metavar="YYYY-MM-DD",
                        help=f"only games before this date (default {dynamic_model.TRAINING_CUTOFF})")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--params", default=str(params.PARAMS_PATH), help="model_params.json to add the section to")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing home section")
    args = parser.parse_args(argv)

    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error("--before must be a date like 2026-10-03")
    if before > dynamic_model.TRAINING_CUTOFF:
        parser.error(f"--before must be on or before {dynamic_model.TRAINING_CUTOFF}: 2026-27 games are the test set")
    out = Path(args.params)
    if not out.exists():
        parser.error(f"{out.name} not found; run python code/fit_sigma.py first (the fit uses its sigma)")
    model_params = json.loads(out.read_text(encoding="utf-8"))
    if "home" in model_params and not args.overwrite:
        parser.error(f"{out.name} already has a home section; pass --overwrite to refit")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")
    sigma = params.check_sigma(model_params["sigma"])

    start = time.perf_counter()
    data = dynamic_model.load_training_rows(args.csv, before, args.years)
    games = training_games(data)
    fit = estimate_home(games, data["team_ids"], sigma)
    seconds = time.perf_counter() - start

    neutral_ids = sorted({dynamic_model.boxscores.game_key(gid)
                          for gid, g in zip(data["cleaned"]["gameId"], games)
                          if g["side"] is None and g["team"] in dynamic_model.NBA_TEAMS
                          and g["opponent"] in dynamic_model.NBA_TEAMS})
    boost = split(fit["delta"])
    model_params["home"] = {
        "offense": {f: round(boost["offense"][f], 5) for f in FACTORS},
        "defense": {f: round(boost["defense"][f], 5) for f in FACTORS},
        "fit": {
            "method": "per-factor gap delta = offense - defense maximizing the one-step-ahead (prequential) "
                      "predictive log-likelihood of the exact training pass with the boost (flat start, tau=0, "
                      "fitted sigma); split evenly, offense +delta/2 and defense -delta/2 (only the gap is "
                      "identifiable; the ratings are neutral-site strength); grid -0.04..0.04 step 0.002, then "
                      "+-0.002 in steps of 0.0001; SE from the log-likelihood curvature",
            "data": {"csv": Path(args.csv).name, "before": before, "years": args.years, "rows": len(games),
                     "games": int(data["games"]), "first": data["first"], "last": data["last"]},
            "delta": {f: fit["delta"][f] for f in FACTORS},
            "delta_se": {f: round(fit["se"][f], 5) for f in FACTORS},
            "log_likelihood": {f: round(fit["log_likelihood"][f], 2) for f in FACTORS},
            "log_likelihood_without_home": {f: round(fit["log_likelihood_without_home"][f], 2) for f in FACTORS},
            "mean_squared_standardized_residual": {f: round(fit["mean_z2"][f], 4) for f in FACTORS},
            "rows_with_boost": sum(g["side"] is not None for g in games),
            "neutral_site_games": neutral_ids,
            "pace": "no pace boost: a shift of every home game's possessions is the same thing as a different "
                    "league pace level (only neutral-site games could tell them apart), see fit_home.py",
            "fitted": date.today().isoformat(),
        },
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(model_params, indent=4) + "\n", encoding="utf-8")
    tmp.replace(out)

    print(f"Fitted on {len(games)} rows ({data['games']} games, {data['first']} to {data['last']}) in {seconds:.0f} s; "
          f"{len(neutral_ids)} neutral-site games get no boost\n")
    print(f"{'factor':<7}{'gap':>9}{'SE':>9}{'offense':>10}{'defense':>10}   {'log-lik gain':>12}   mean z^2")
    for f in FACTORS:
        gain = fit["log_likelihood"][f] - fit["log_likelihood_without_home"][f]
        print(f"{f:<7}{fit['delta'][f]:>+9.4f}{fit['se'][f]:>9.4f}{boost['offense'][f]:>+10.5f}"
              f"{boost['defense'][f]:>+10.5f}   {gain:>+12.1f}   {fit['mean_z2'][f]:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
