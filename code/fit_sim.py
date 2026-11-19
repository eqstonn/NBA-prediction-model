"""
Estimate the simulation's Phase 7 parameters from the TRAINING data only and add them to
model_params.json (read by params.py -> simulation.py, update.py, dynamic_model.py).

    python code/fit_sim.py              # Kaggle, 3 years before 2026-10-03 -> model_params.json
    python code/fit_sim.py --overwrite  # refit and replace these three sections

It writes only the "factor_corr", "points" and "pace" sections; "sigma" / "sigma_fit"
(code/fit_sigma.py) are kept exactly as they are. No 2026-27 game is used.
The fits are scored on NBA-vs-NBA games only (the 7 All-Star games are not basketball as
the model knows it).

1. factor_corr: how the four factors' game-to-game errors move together
------------------------------------------------------------------------
Run the exact training pass (flat start, tau = 0, fitted sigma) and take each row's
standardized one-step-ahead residual z_f = (y_f - forecast_f) / sqrt(forecast variance),
the same residuals fit_sigma.py scores. A game gives two rows: the team's offensive line
and the opponent's. Each game's 8 residuals (team eFG/TOV/ORB/FTR, opponent eFG/TOV/ORB/FTR)
are one sample; each game is used in both orders (team <-> opponent), so the matrix is the
same whichever team is called "team". The first season of the window is skipped (burn-in:
every team starts from the same flat curves, so early residuals are mostly rating error,
not game noise). The simulation draws the 8 game noises with this correlation.

2. points: the formula from four factors + possessions to points
-----------------------------------------------------------------
NBA.com's own definitions (checked on the training box scores):
    raw  = FGA + 0.44 FTA + TOV - ORB          (TOV includes team turnovers)
    TOV% = TOV / raw,  FTR = FTA / FGA,  ORB% = ORB / (NBA.com's rebound chances),
    NBA.com possessions = raw - possession_offset (about 2)
Solving for FGA, with ORB = rebound_chances x ORB% x FGA x (1 - eFG):
    FGA    = (possessions + possession_offset) (1 - TOV%) / (1 + 0.44 FTR - rebound_chances ORB% (1 - eFG))
    points = FGA (2 eFG + FTR ft_pct)
possession_offset = mean(raw - possessions), rebound_chances = sum ORB / sum ORB% FGA (1 - eFG),
ft_pct = league FTM / FTA. The old formula, possessions (1 - TOV)(1 + ORB (1 - eFG)), leaves
out the possessions that end in free throws: +9.7% FGA, +21.6 points per game.

3. pace: possessions per game
-----------------------------
Every team starts from the same flat curve (start_mean = the league average, start_std),
then each game's possessions ~ N((pace_a + pace_b) / 2, sigma^2) updates both teams exactly
(bayes_update.update_pace). sigma maximizes the one-step-ahead predictive likelihood,
like fit_sigma.py: grid 0.5-15 in steps of 0.25, then 0.01 steps within +-0.25.
"""
import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np

import bayes_update
import dynamic_model
import fit_sigma
import params
import simulation
from schedule import NBA_TEAMS

FACTORS = params.FACTORS
BURN_IN_DAYS = 365          # skip the first season of the window for the correlation
PACE_START_STD = 5.0        # possessions; like the factors' flat std 0.05, wider than the spread between teams
PACE_COARSE = [round(0.25 * i, 2) for i in range(2, 61)]   # 0.5 .. 15
PACE_FINE_STEP, PACE_FINE_STEPS = 0.01, 25
LOG_2PI = math.log(2 * math.pi)


def is_nba_game(team: str, opponent: str) -> bool:
    return str(team) in NBA_TEAMS and str(opponent) in NBA_TEAMS


# ---------------------------------------------------------------------------
# 1. factor correlation
# ---------------------------------------------------------------------------

def residual_pairs(games: list, team_ids, sigma: dict, keep=None) -> list:
    """
    [(game_id, z_team, z_opponent)] per game, two lists of 4 standardized one-step-ahead
    residuals, in game order. `games` = update_game inputs with a "game_id", oldest first;
    games without exactly two rows, or for which keep(row) is False, are left out.
    """
    residuals = []
    fit_sigma.prequential(games, team_ids, sigma, residuals=residuals)
    by_game = {}
    for g, z in zip(games, residuals):
        by_game.setdefault(g["game_id"], []).append((g, [z[f] for f in FACTORS]))
    pairs = []
    for game_id, rows in by_game.items():
        if len(rows) != 2 or (keep is not None and not keep(rows[0][0])):
            continue
        (g1, z1), (g2, z2) = rows
        if g1["team"] != g2["opponent"] or g1["opponent"] != g2["team"]:
            continue
        pairs.append((game_id, z1, z2))
    return pairs


def correlation_from_pairs(pairs: list) -> np.ndarray:
    """8x8 correlation of [z_team, z_opponent], every game in both orders (so it's exchangeable)."""
    if len(pairs) < 10:
        raise ValueError("need at least 10 games to estimate the factor correlation")
    samples = np.array([z1 + z2 for _, z1, z2 in pairs] + [z2 + z1 for _, z1, z2 in pairs], dtype=float)
    corr = np.corrcoef(samples, rowvar=False)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return corr


def estimate_corr(games: list, team_ids, sigma: dict, burn_in=frozenset(), keep=None) -> dict:
    """
    {"matrix" (8x8 ndarray), "games", "matrix_with_burn_in"}: correlation of the game noise
    over the games kept by keep(row), leaving out the game_ids in `burn_in`.
    """
    pairs = residual_pairs(games, team_ids, sigma, keep=keep)
    used = [p for p in pairs if p[0] not in burn_in]
    corr = correlation_from_pairs(used)
    np.linalg.cholesky(corr)   # raises if it isn't positive definite
    return {"matrix": corr, "games": len(used), "burn_in_games": len(pairs) - len(used),
            "matrix_with_burn_in": correlation_from_pairs(pairs)}


# ---------------------------------------------------------------------------
# 2. points formula
# ---------------------------------------------------------------------------

def _num(df, column):
    return df[column].astype(float).to_numpy()


def estimate_points(cleaned) -> dict:
    """Constants of simulation.points from NBA-vs-NBA training rows, plus how well each formula fits them."""
    nba = cleaned[[is_nba_game(dynamic_model.team_key(t), dynamic_model.team_key(o))
                   for t, o in zip(cleaned["teamId"], cleaned["opponentTeamId"])]]
    cols = ["fieldGoalsAttempted", "freeThrowsAttempted", "freeThrowsMade", "turnovers", "turnoversTeam",
            "reboundsOffensive", "possessions", "teamScore", "effectiveFieldGoalPercentage",
            "teamTurnoverPercentage", "offensiveReboundPercentage", "freeThrowAttemptRate"]
    nba = nba.dropna(subset=cols)
    fga, fta, ftm = _num(nba, "fieldGoalsAttempted"), _num(nba, "freeThrowsAttempted"), _num(nba, "freeThrowsMade")
    tov = _num(nba, "turnovers") + _num(nba, "turnoversTeam")
    orb, poss, pts = _num(nba, "reboundsOffensive"), _num(nba, "possessions"), _num(nba, "teamScore")
    factors = {"eFG": _num(nba, "effectiveFieldGoalPercentage"), "TOV": _num(nba, "teamTurnoverPercentage"),
               "ORB": _num(nba, "offensiveReboundPercentage"), "FTR": _num(nba, "freeThrowAttemptRate")}

    constants = {
        "ft_pct": float(ftm.sum() / fta.sum()),
        "possession_offset": float(np.mean(fga + 0.44 * fta + tov - orb - poss)),
        "rebound_chances": float(orb.sum() / (factors["ORB"] * fga * (1 - factors["eFG"])).sum()),
    }

    game_ids = nba["gameId"].to_numpy()

    def game_totals(team_points):
        totals = {}
        for gid, p in zip(game_ids, team_points):
            totals.setdefault(gid, []).append(p)
        return np.array([sum(v) for v in totals.values() if len(v) == 2])

    actual = game_totals(pts)
    new = game_totals(simulation.points(factors, poss, constants)) - actual
    old_actual = game_totals(simulation.legacy_points(factors, poss)) - actual
    old_100 = game_totals(simulation.legacy_points(factors, 100.0)) - actual
    return {**constants, "check": {
        "games": int(len(actual)), "actual_game_total": round(float(actual.mean()), 2),
        "game_total_bias": round(float(new.mean()), 2), "game_total_mae": round(float(np.abs(new).mean()), 2),
        "legacy_bias_actual_possessions": round(float(old_actual.mean()), 2),
        "legacy_bias_100_possessions": round(float(old_100.mean()), 2),
        "possessions_per_team_game": round(float(poss.mean()), 2),
    }}


# ---------------------------------------------------------------------------
# 3. pace
# ---------------------------------------------------------------------------

def pace_prequential(games: list, team_ids, sigma: float, start_mean: float, start_std: float, keep=None):
    """(total log predictive density, mean squared standardized residual, games scored) of the pace training pass."""
    priors = {t: {"pace": {"mean": start_mean, "std": start_std}} for t in team_ids}
    loglik, z2, n = 0.0, 0.0, 0
    for g in games:
        if keep is None or keep(g):
            mean, var = bayes_update.pace_predictive(priors, g["team"], g["opponent"], sigma)
            r2 = (g["possessions"] - mean) ** 2
            loglik -= 0.5 * (LOG_2PI + math.log(var) + r2 / var)
            z2 += r2 / var
            n += 1
        bayes_update.update_pace(priors, g["team"], g["opponent"], g["possessions"],
                                 tau=dynamic_model.TRAINING_TAU, sigma=sigma)
    return loglik, z2 / max(n, 1), n


def estimate_pace_sigma(games: list, team_ids, start_mean: float, start_std: float = PACE_START_STD, keep=None) -> dict:
    """Game noise sigma (possessions) maximizing the one-step-ahead predictive likelihood (coarse, then fine grid)."""
    def loglik(s):
        return pace_prequential(games, team_ids, s, start_mean, start_std, keep)[0]

    best = max(PACE_COARSE, key=loglik)
    fine = [round(best + k * PACE_FINE_STEP, 2) for k in range(-PACE_FINE_STEPS, PACE_FINE_STEPS + 1)]
    best = max((s for s in fine if s > 0), key=loglik)
    ll, z2, n = pace_prequential(games, team_ids, best, start_mean, start_std, keep)
    return {"sigma": best, "log_likelihood": ll, "mean_z2": z2, "games": n}


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the simulation's factor correlation, points formula and pace.")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--before", default=dynamic_model.TRAINING_CUTOFF, metavar="YYYY-MM-DD",
                        help=f"only games before this date (default {dynamic_model.TRAINING_CUTOFF})")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--params", default=str(params.PARAMS_PATH), help="model_params.json to add the sections to")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing factor_corr/points/pace")
    args = parser.parse_args(argv)

    try:
        before = date.fromisoformat(args.before).isoformat()
    except ValueError:
        parser.error("--before must be a date like 2026-10-03")
    if before > dynamic_model.TRAINING_CUTOFF:
        parser.error(f"--before must be on or before {dynamic_model.TRAINING_CUTOFF}: 2026-27 games are the test set")
    out = Path(args.params)
    if not out.exists():
        parser.error(f"{out.name} not found; run python code/fit_sigma.py first (the correlation uses its sigma)")
    model_params = json.loads(out.read_text(encoding="utf-8"))
    present = [k for k in ("factor_corr", "points", "pace") if k in model_params]
    if present and not args.overwrite:
        parser.error(f"{out.name} already has {', '.join(present)}; pass --overwrite to refit")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")
    sigma = params.check_sigma(model_params["sigma"])

    start = time.perf_counter()
    data = dynamic_model.load_training_rows(args.csv, before, args.years)
    games = [{**dynamic_model.row_to_game(row), "game_id": str(row["gameId"])} for _, row in data["rows"].iterrows()]
    games = [{**g, **{f: float(g[f]) for f in FACTORS}} for g in games]
    # burn-in: the window's first season (games in the first BURN_IN_DAYS days)
    first_day = date.fromisoformat(data["first"]).toordinal()
    burn_in = {str(gid) for gid, when in zip(data["cleaned"]["gameId"], data["cleaned"]["gameDateTimeEst"])
               if when.toordinal() < first_day + BURN_IN_DAYS}
    corr = estimate_corr(games, data["team_ids"], sigma, burn_in=burn_in,
                         keep=lambda g: is_nba_game(g["team"], g["opponent"]))
    skip, corr_all = corr["burn_in_games"], corr["matrix_with_burn_in"]

    points = estimate_points(data["cleaned"])

    pace_games = data["possessions"]
    nba_pace = [g for g in pace_games if is_nba_game(g["team"], g["opponent"])]
    start_mean = round(float(np.mean([g["possessions"] for g in nba_pace])), 2)
    pace = estimate_pace_sigma(pace_games, data["team_ids"], start_mean, PACE_START_STD,
                               keep=lambda g: is_nba_game(g["team"], g["opponent"]))
    seconds = time.perf_counter() - start

    source = {"csv": Path(args.csv).name, "before": before, "years": args.years,
              "games": int(data["games"]), "first": data["first"], "last": data["last"]}
    model_params["factor_corr"] = {
        "labels": list(params.CORR_LABELS),
        "matrix": [[round(float(v), 4) for v in row] for row in corr["matrix"]],
        "fit": {"method": "correlation of the standardized one-step-ahead residuals of the exact training pass "
                          "(flat start, tau=0, fitted sigma); one sample per game = the team's and the opponent's "
                          "4 residuals, each game in both orders; NBA-vs-NBA games after a one-season burn-in",
                "data": source, "games_used": corr["games"], "burn_in_games": skip,
                "max_abs_change_without_burn_in": round(float(np.abs(corr_all - corr["matrix"]).max()), 4),
                "fitted": date.today().isoformat()},
    }
    model_params["points"] = {
        "ft_pct": round(points["ft_pct"], 4),
        "possession_offset": round(points["possession_offset"], 3),
        "rebound_chances": round(points["rebound_chances"], 4),
        "formula": "FGA = (possessions + possession_offset) (1 - TOV) / (1 + 0.44 FTR - rebound_chances ORB (1 - eFG)); "
                   "points = FGA (2 eFG + FTR ft_pct)",
        "fit": {"method": "NBA.com definitions solved for FGA; ft_pct = league FTM/FTA, possession_offset = "
                          "mean(FGA + 0.44 FTA + TOV - ORB - possessions), rebound_chances = sum ORB / "
                          "sum ORB% FGA (1 - eFG); NBA-vs-NBA rows",
                "data": source, "check_on_actual_factors_and_possessions": points["check"],
                "fitted": date.today().isoformat()},
    }
    model_params["pace"] = {
        "sigma": pace["sigma"], "start_mean": start_mean, "start_std": PACE_START_STD,
        "fit": {"method": "maximum of the one-step-ahead predictive log-likelihood of the exact pace training "
                          "pass (flat start, tau=0), possessions = average of the two teams' counts; scored on "
                          "NBA-vs-NBA games; grid 0.5-15 step 0.25, then +-0.25 in steps of 0.01",
                "data": source, "games_scored": pace["games"], "log_likelihood": round(pace["log_likelihood"], 2),
                "mean_squared_standardized_residual": round(pace["mean_z2"], 4),
                "fitted": date.today().isoformat()},
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(model_params, indent=4) + "\n", encoding="utf-8")
    tmp.replace(out)

    print(f"Fitted on {data['games']} games ({data['first']} to {data['last']}) in {seconds:.0f} s\n")
    print(f"factor correlation ({corr['games']} NBA games after {skip} burn-in games):")
    short = [lab.replace("opponent ", "opp ").replace("team ", "") for lab in params.CORR_LABELS]
    print("          " + "".join(f"{s:>8}" for s in short))
    for s, row in zip(short, corr["matrix"]):
        print(f"{s:<10}" + "".join(f"{v:>8.3f}" for v in row))
    print(f"  (max change if the burn-in season is kept: {np.abs(corr_all - corr['matrix']).max():.3f})")
    c = points["check"]
    print(f"\npoints formula: ft_pct {points['ft_pct']:.4f}, possession_offset {points['possession_offset']:.3f}, "
          f"rebound_chances {points['rebound_chances']:.4f}")
    print(f"  on actual factors + possessions ({c['games']} games, actual total {c['actual_game_total']}): "
          f"bias {c['game_total_bias']:+.2f}, MAE {c['game_total_mae']:.2f} per game "
          f"(old formula {c['legacy_bias_actual_possessions']:+.2f}; with 100 possessions "
          f"{c['legacy_bias_100_possessions']:+.2f})")
    print(f"\npace: start {start_mean:.2f} +- {PACE_START_STD}, game noise sigma {pace['sigma']:.2f} possessions, "
          f"mean z^2 {pace['mean_z2']:.3f} over {pace['games']} games")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
