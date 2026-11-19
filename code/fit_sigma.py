"""
Estimate the game-to-game noise sigma of each four factor from the TRAINING data only and
save it to model_params.json (read by params.py -> bayes_update.py and simulation.py).

    python code/fit_sigma.py              # Kaggle, 3 years before 2026-10-03 -> model_params.json
    python code/fit_sigma.py --overwrite  # refit and replace model_params.json

No 2026-27 game is used (cutoff 2026-10-03, the start of the 2026-27 preseason), so the
season stays a fair test.

Method: one-step-ahead (prequential) predictive likelihood
------------------------------------------------------------
Run the exact training pass (flat start, tau = 0) with a candidate sigma. Just before each
row is applied, the model's forecast for that row's factor is

    y ~ Normal( (m_o + m_d) / 2 ,  (s_o^2 + s_d^2) / 4 + sigma^2 )

from the curves as they stood before the game. Add up log density(actual y) over all rows.
A sigma that is too small makes ordinary surprises look impossible; one that is too large
makes every forecast vague. The best total is the sigma that forecasts unseen games best.
The four factors don't interact in the model, so each gets its own sigma: a coarse grid
(0.005 to 0.200 in 0.005 steps), then 0.0001 steps within +-0.005 of the best.
At the best sigma the squared standardized residuals average about 1 (a calibration check).
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
COARSE = [round(0.005 * i, 4) for i in range(1, 41)]   # 0.005 .. 0.200
FINE_STEP, FINE_STEPS = 0.0001, 50                     # best +- 50 steps of 0.0001
LOG_2PI = math.log(2 * math.pi)


def prequential(games: list, team_ids, sigma: dict, residuals=None):
    """
    ({factor: total log predictive density}, {factor: mean squared standardized residual}) of
    the exact training pass with `sigma`. `games` = update_game inputs, oldest first.
    If `residuals` is a list, each row's standardized residuals {factor: (y - mean) / sqrt(var)}
    are appended to it, in row order (fit_sim.py uses them for the factor correlation).
    """
    priors = default_priors(team_ids)
    loglik = {f: 0.0 for f in FACTORS}
    z2 = {f: 0.0 for f in FACTORS}
    for g in games:
        offense, defense = priors[g["team"]]["offense"], priors[g["opponent"]]["defense"]
        prior = {"o": {f: (offense[f]["mean"], offense[f]["std"]) for f in FACTORS},
                 "d": {f: (defense[f]["mean"], defense[f]["std"]) for f in FACTORS}}
        z = {}
        for f in FACTORS:
            mean, var = bayes_update.predictive(prior, f, sigma)
            r2 = (g[f] - mean) ** 2
            loglik[f] -= 0.5 * (LOG_2PI + math.log(var) + r2 / var)
            z2[f] += r2 / var
            z[f] = (g[f] - mean) / math.sqrt(var)
        if residuals is not None:
            residuals.append(z)
        bayes_update.update_game(priors, g, tau=dynamic_model.TRAINING_TAU, engine="exact", sigma=sigma)
    return loglik, {f: z2[f] / len(games) for f in FACTORS}


def estimate_sigma(games: list, team_ids) -> dict:
    """Per-factor sigma maximizing the one-step-ahead predictive log-likelihood (coarse grid, then fine)."""
    best = {f: (-math.inf, None) for f in FACTORS}

    def consider(sigma):
        loglik, _ = prequential(games, team_ids, sigma)
        for f in FACTORS:
            if loglik[f] > best[f][0]:
                best[f] = (loglik[f], sigma[f])

    for s in COARSE:
        consider({f: s for f in FACTORS})
    centers = {f: best[f][1] for f in FACTORS}
    for k in range(-FINE_STEPS, FINE_STEPS + 1):
        consider({f: max(FINE_STEP, round(centers[f] + k * FINE_STEP, 4)) for f in FACTORS})

    sigma = {f: best[f][1] for f in FACTORS}
    loglik, z2 = prequential(games, team_ids, sigma)
    return {"sigma": sigma, "log_likelihood": loglik, "mean_z2": z2}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit per-factor game noise sigma on the training data.")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--before", default=dynamic_model.TRAINING_CUTOFF, metavar="YYYY-MM-DD",
                        help=f"only games before this date (default {dynamic_model.TRAINING_CUTOFF})")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--out", default=str(params.PARAMS_PATH), help="where to write (default model_params.json)")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    args = parser.parse_args(argv)

    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error("--before must be a date like 2026-10-03")
    if before > dynamic_model.TRAINING_CUTOFF:
        parser.error(f"--before must be on or before {dynamic_model.TRAINING_CUTOFF}: 2026-27 games are the test set")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        parser.error(f"{out.name} already exists; pass --overwrite to refit it")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")

    start = time.perf_counter()
    data = dynamic_model.load_training_rows(args.csv, before, args.years)
    games = [dynamic_model.row_to_game(row) for _, row in data["rows"].iterrows()]
    games = [{**g, **{f: float(g[f]) for f in FACTORS}} for g in games]
    fit = estimate_sigma(games, data["team_ids"])
    legacy_loglik, legacy_z2 = prequential(games, data["team_ids"], params.LEGACY_UPDATE_SIGMA)
    seconds = time.perf_counter() - start

    # keep the other sections (factor_corr, points, pace from fit_sim.py); replace only these two
    result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    result.update({
        "sigma": {f: fit["sigma"][f] for f in FACTORS},
        "sigma_fit": {
            "method": "per-factor maximum of the one-step-ahead (prequential) predictive log-likelihood "
                      "of the exact training update (flat start, tau=0); grid 0.005-0.200 step 0.005, "
                      "then +-0.005 around the best in steps of 0.0001",
            "data": {"csv": Path(args.csv).name, "before": before, "years": args.years,
                     "rows": len(games), "games": int(data["games"]), "first": data["first"], "last": data["last"]},
            "log_likelihood": {f: round(fit["log_likelihood"][f], 2) for f in FACTORS},
            "log_likelihood_at_0.05": {f: round(legacy_loglik[f], 2) for f in FACTORS},
            "mean_squared_standardized_residual": {f: round(fit["mean_z2"][f], 4) for f in FACTORS},
            "fitted": date.today().isoformat(),
        },
    })
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=4) + "\n", encoding="utf-8")
    tmp.replace(out)

    print(f"Fitted on {len(games)} rows ({data['games']} games, {data['first']} to {data['last']}) in {seconds:.0f} s\n")
    print(f"{'factor':<7}{'sigma':>8}{'was':>7}   {'log-lik':>10} {'at 0.05':>10} {'gain':>8}   mean z^2")
    for f in FACTORS:
        gain = fit["log_likelihood"][f] - legacy_loglik[f]
        print(f"{f:<7}{fit['sigma'][f]:>8.4f}{0.05:>7.2f}   {fit['log_likelihood'][f]:>10.1f} "
              f"{legacy_loglik[f]:>10.1f} {gain:>+8.1f}   {fit['mean_z2'][f]:.3f}")
    print(f"\n(the simulation used {params.LEGACY_SIM_SIGMA} before Phase 6)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
