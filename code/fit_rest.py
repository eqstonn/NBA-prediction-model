"""
Estimate the rest effect (back-to-backs) from the TRAINING data only and save it in
model_params.json["rest"] (read by params.py -> update.py / dynamic_model.py /
simulation.py with --rest on). Every other section is kept as it is.

    python code/fit_rest.py              # Kaggle, 3 years before 2026-10-03 -> model_params.json
    python code/fit_rest.py --overwrite  # refit and replace the "rest" section

No 2026-27 game is used. Rest comes from NBA.com's schedules of the training seasons
(rest.py, cached in stats/schedules/): "b2b" (0 days off), "1", "2+" (and every opener),
plus whether a back-to-back's second night was in a different arena ("traveled").

The model
---------
Each training row is one team's offensive line against the opponent's defense. With rest on,
its expected value gets a shift from the two teams' rest, on top of the home boost:
    (o + d) / 2 + home + rest["offense"][rest of the offense team] + rest["defense"][rest of the defense team]
with "2+" as the baseline (shift 0). Candidate forms, per factor:
    both        separate offense and defense effects per category (a tired offense and a
                tired defense move the line independently)
    difference  defense = -offense: the line moves by e(r_offense) - e(r_defense), so only the
                rest GAP matters and two equally rested teams get nothing
each with categories b2b + 1 day, b2b only, or (difference) linear in days off (capped at 2),
and with back-to-backs split by travel as a check.

Method: one-step-ahead (prequential) likelihood, as fit_home.py, solved exactly
-----------------------------------------------------------------------------------
The score is the same as fit_home.py's: run the exact training pass (flat start, tau = 0,
fitted sigma, the saved home boost) with the shifts in the forecast AND the update, and add
up each row's log density under its forecast N((m_o + m_d)/2 + shifts, (s_o^2 + s_d^2)/4 +
sigma^2). In the exact (Kalman) update the variances don't depend on the observations and the
means are linear in y - shift, so every forecast residual is exactly linear in the rest
parameters: e(theta) = e(0) - Z theta, with column k of Z from one extra training pass with a
unit shift on parameter k. The log-likelihood is then exactly quadratic, its maximum is a
weighted least-squares solution, and the standard errors are exact (the curvature fit_home.py
measures numerically). Factors are fitted separately, as in fit_home.py.

The form is chosen by AIC (2 x log-likelihood gain - 2 x parameters) among the forms the
update can use (travel is a check only), with likelihood-ratio tests of the nested pairs and a
held-out check (fit on the first two seasons, score the third).

Pace: the same fit for game possessions, shift = p[r_a] + p[r_b] (both teams share the
possessions, so there is no difference form). Saved only if it is meaningful.
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
import fit_home
import params
import rest
import simulation
from state_manager import default_priors

FACTORS = params.FACTORS
# basis parameters: a shift of the line when the offense / defense team is in a category
BASIS = ("off_b2bT", "off_b2bS", "off_1", "def_b2bT", "def_b2bS", "def_1")
UNIT = 0.01
LOG_2PI = math.log(2 * math.pi)


def basis_category(info: dict) -> str:
    """b2bT / b2bS (back-to-back, traveled or not; unknown arena counts as traveled), "1", "2+"."""
    if info["rest"] == "b2b":
        return "b2bS" if info["traveled"] is False else "b2bT"
    return info["rest"]


def _cols(*weights):
    v = np.zeros(len(BASIS))
    for name, w in weights:
        v[BASIS.index(name)] += w
    return v


def _form(*columns):
    return np.column_stack(columns)


B2B_OFF = (("off_b2bT", 1), ("off_b2bS", 1))
B2B_DEF = (("def_b2bT", 1), ("def_b2bS", 1))
B2B_DIFF = B2B_OFF + (("def_b2bT", -1), ("def_b2bS", -1))
ONE_DIFF = (("off_1", 1), ("def_1", -1))
# name -> (basis x parameters map, parameter names, usable by the update)
FORMS = {
    "both, b2b + 1 day + travel": (np.eye(len(BASIS)), list(BASIS), False),
    "both, b2b + 1 day": (_form(_cols(*B2B_OFF), _cols(("off_1", 1)), _cols(*B2B_DEF), _cols(("def_1", 1))),
                          ["off_b2b", "off_1", "def_b2b", "def_1"], True),
    "both, b2b": (_form(_cols(*B2B_OFF), _cols(*B2B_DEF)), ["off_b2b", "def_b2b"], True),
    "difference, b2b + 1 day + travel": (_form(_cols(("off_b2bT", 1), ("def_b2bT", -1)),
                                               _cols(("off_b2bS", 1), ("def_b2bS", -1)), _cols(*ONE_DIFF)),
                                         ["b2b traveled", "b2b same arena", "1 day"], False),
    "difference, b2b + 1 day": (_form(_cols(*B2B_DIFF), _cols(*ONE_DIFF)), ["b2b", "1 day"], True),
    "difference, b2b": (_form(_cols(*B2B_DIFF)), ["b2b"], True),
    "difference, linear in days": (_form(_cols(("off_b2bT", -2), ("off_b2bS", -2), ("off_1", -1),
                                               ("def_b2bT", 2), ("def_b2bS", 2), ("def_1", 1))),
                                   ["per day of rest"], True),
}
# likelihood-ratio tests: (bigger form, nested form, what it tests)
LR_TESTS = [("both, b2b + 1 day", "difference, b2b + 1 day", "both teams vs difference (b2b + 1 day)"),
            ("both, b2b", "difference, b2b", "both teams vs difference (b2b)"),
            ("difference, b2b + 1 day", "difference, b2b", "a 1-day effect"),
            ("difference, b2b + 1 day + travel", "difference, b2b + 1 day", "travel on back-to-backs")]


def chi2_sf(x: float, k: int) -> float:
    """P(chi^2_k > x) (series for the regularized gamma function; fine for the small k used here)."""
    if x <= 0:
        return 1.0
    a, term = k / 2, 1.0 / (k / 2)
    total = term
    for n in range(1, 500):
        term *= (x / 2) / (a + n)
        total += term
        if term < 1e-15 * total:
            break
    return max(0.0, 1.0 - math.exp(-x / 2 + a * math.log(x / 2) - math.lgamma(a)) * total)


# ---------------------------------------------------------------------------
# the linear map from parameters to forecast residuals
# ---------------------------------------------------------------------------

def residuals(games: list, team_ids, sigma: dict, offsets: list):
    """(residuals, variances), each rows x factors, of the exact training pass with per-row `offsets`."""
    priors = default_priors(team_ids)
    e, v = np.zeros((len(games), len(FACTORS))), np.zeros((len(games), len(FACTORS)))
    for i, (g, off) in enumerate(zip(games, offsets)):
        o, d = priors[g["team"]]["offense"], priors[g["opponent"]]["defense"]
        prior = {"o": {f: (o[f]["mean"], o[f]["std"]) for f in FACTORS},
                 "d": {f: (d[f]["mean"], d[f]["std"]) for f in FACTORS}}
        for k, f in enumerate(FACTORS):
            mean, var = bayes_update.predictive(prior, f, sigma)
            e[i, k] = g[f] - mean - (off[f] if off else 0.0)
            v[i, k] = var
        bayes_update.update_game(priors, g, tau=dynamic_model.TRAINING_TAU, engine="exact", sigma=sigma, offset=off)
    return e, v


def basis_shifts(row_cats: list, name: str, unit: float) -> list:
    """Per row: `unit` when the row's offense (off_*) / defense (def_*) team is in the basis parameter's category."""
    side, cat = name.split("_")
    return [unit if c is not None and c[0 if side == "off" else 1] == cat else 0.0 for c in row_cats]


def design(games: list, team_ids, sigma: dict, base: list, row_cats: list, extra=None):
    """
    e0 (residuals at theta = 0, i.e. with the `base` offsets only), v (variances) and Z
    ({column: rows x factors}) with e(theta) = e0 - sum_k Z[k] theta_k, for the BASIS columns
    and any `extra` {name: per-row offsets added to base}.
    """
    e0, v = residuals(games, team_ids, sigma, base)
    columns = {name: [{f: s for f in FACTORS} if s else None for s in basis_shifts(row_cats, name, UNIT)]
               for name in BASIS}
    columns.update(extra or {})
    z = {}
    for name, shifts in columns.items():
        offsets = [params.combine_offsets(b, s) for b, s in zip(base, shifts)]
        e1, v1 = residuals(games, team_ids, sigma, offsets)
        if not np.allclose(v1, v):
            raise RuntimeError("forecast variances changed with the offsets; the linear method doesn't apply")
        z[name] = (e0 - e1) / UNIT
    return e0, v, z


def loglik(e, v) -> float:
    return float(-0.5 * np.sum(np.log(v) + LOG_2PI + e ** 2 / v))


def wls(e0, v, x, fit_rows=None):
    """theta maximizing -0.5 sum (e0 - x theta)^2 / v over fit_rows (default all), and its covariance."""
    rows = slice(None) if fit_rows is None else fit_rows
    w = 1 / v[rows]
    xtwx = x[rows].T @ (x[rows] * w[:, None])
    cov = np.linalg.inv(xtwx)
    return cov @ (x[rows].T @ (w * e0[rows])), cov


def fit_form(e0, v, z, a, fit_rows=None, score_rows=None):
    """Per factor: theta, se, cov and the log-likelihood gain over theta = 0 (on score_rows, default all)."""
    out = {}
    score = slice(None) if score_rows is None else score_rows
    for k, f in enumerate(FACTORS):
        x = np.column_stack([z[name][:, k] for name in BASIS]) @ a
        theta, cov = wls(e0[:, k], v[:, k], x, fit_rows)
        gain = loglik((e0[:, k] - x @ theta)[score], v[score, k]) - loglik(e0[score, k], v[score, k])
        out[f] = {"theta": theta, "se": np.sqrt(np.diag(cov)), "cov": cov, "gain": gain}
    return out


# ---------------------------------------------------------------------------
# margin in points, pace, the home boost
# ---------------------------------------------------------------------------

def league_line(data: dict) -> tuple:
    """Average NBA line ({factor: mean}) and average game possessions in the training window."""
    rows = data["rows"]
    line = {"eFG": float(rows["effectiveFieldGoalPercentage"].mean()), "TOV": float(rows["teamTurnoverPercentage"].mean()),
            "ORB": float(rows["offensiveReboundPercentage"].mean()), "FTR": float(rows["freeThrowAttemptRate"].mean())}
    possessions = float(np.mean([g["possessions"] for g in data["possessions"]]))
    return line, possessions


def margin_points(effects: dict, rest_a: str, rest_b: str, line: dict, possessions: float) -> float:
    """Points of margin (team a minus team b) the rest effects add at league-average factors and pace."""
    a_shift, b_shift = params.rest_offsets("on", rest_a, rest_b, effects)
    constants = params.points_params()
    pts = lambda shift: simulation.points({f: line[f] + shift[f] for f in FACTORS}, possessions, constants)  # noqa: E731
    zero = {f: 0.0 for f in FACTORS}
    return (pts(a_shift) - pts(b_shift)) - (pts(zero) - pts(zero))


def margin_gradient(line: dict, possessions: float) -> dict:
    """d(points of a line)/d(factor) at the league-average line (for the delta-method SE)."""
    constants, h = params.points_params(), 1e-6
    base = simulation.points(line, possessions, constants)
    return {f: (simulation.points({**line, f: line[f] + h}, possessions, constants) - base) / h for f in FACTORS}


def pace_check(data: dict, rest_rows_by_game: dict) -> dict:
    """The same fit for game possessions: shift = p[rest_a] + p[rest_b] for p in (b2b, 1); returns estimates."""
    pp = params.pace_params()
    games = [g for g in data["possessions"] if dynamic_model.boxscores.game_key(g["game_id"]) in rest_rows_by_game]
    cats = []
    for g in games:
        info = rest_rows_by_game[dynamic_model.boxscores.game_key(g["game_id"])]
        cats.append((info[g["team"]]["rest"], info[g["opponent"]]["rest"]))

    def run(shift):
        priors = default_priors(data["team_ids"])
        dynamic_model.start_pace(priors, data["team_ids"], pp["start_mean"], pp["start_std"])
        e, v = np.zeros(len(games)), np.zeros(len(games))
        for i, (g, c) in enumerate(zip(games, cats)):
            off = sum(shift.get(x, 0.0) for x in c)
            mean, var = bayes_update.pace_predictive(priors, g["team"], g["opponent"], pp["sigma"])
            e[i], v[i] = g["possessions"] - mean - off, var
            bayes_update.update_pace(priors, g["team"], g["opponent"], g["possessions"] - off, sigma=pp["sigma"])
        return e, v

    e0, v = run({})
    x = np.column_stack([e0 - run({c: 1.0})[0] for c in ("b2b", "1")])
    theta, cov = wls(e0, v, x)
    return {"b2b": float(theta[0]), "1": float(theta[1]), "b2b_se": float(math.sqrt(cov[0, 0])),
            "1_se": float(math.sqrt(cov[1, 1])), "games": len(games),
            "log_likelihood_gain": loglik(e0 - x @ theta, v) - loglik(e0, v)}


# ---------------------------------------------------------------------------
# the whole fit
# ---------------------------------------------------------------------------

def estimate_rest(data: dict, sigma: dict, boost: dict) -> dict:
    games = fit_home.training_games(data)
    sides = dynamic_model.row_sides(data)
    base = dynamic_model.row_offsets(sides, boost)
    rest_rows = dynamic_model.row_rest(data)
    row_cats = [None if r is None else (basis_category(r[0]), basis_category(r[1])) for r in rest_rows]
    # a unit increase of the home gap (offense +u/2, defense -u/2), for the joint check
    home_unit = [None if s is None else {f: (UNIT / 2 if s == "home" else -UNIT / 2) for f in FACTORS} for s in sides]
    e0, v, z = design(games, data["team_ids"], sigma, base, row_cats, extra={"home_gap": home_unit})

    seasons = np.array([dynamic_model.boxscores.season_of(g) for g in data["cleaned"]["gameId"]])
    first_two = np.isin(seasons, sorted(set(seasons))[:2])
    forms = {}
    for name, (a, names, usable) in FORMS.items():
        fit = fit_form(e0, v, z, a)
        held_out = fit_form(e0, v, z, a, fit_rows=first_two, score_rows=~first_two)
        k = a.shape[1] * len(FACTORS)
        gain = sum(fit[f]["gain"] for f in FACTORS)
        forms[name] = {"fit": fit, "names": names, "usable": usable, "parameters": k, "log_likelihood_gain": gain,
                       "aic_gain": 2 * gain - 2 * k, "held_out_gain": sum(held_out[f]["gain"] for f in FACTORS)}
    chosen = max((n for n in forms if forms[n]["usable"]), key=lambda n: forms[n]["aic_gain"])

    lr = []
    for big, small, what in LR_TESTS:
        stat = 2 * (forms[big]["log_likelihood_gain"] - forms[small]["log_likelihood_gain"])
        df = forms[big]["parameters"] - forms[small]["parameters"]
        lr.append({"test": what, "bigger": big, "nested": small, "chi2": stat, "df": df, "p": chi2_sf(stat, df)})

    # the chosen form as offense / defense effects per category
    a = FORMS[chosen][0]
    basis_values = {f: a @ forms[chosen]["fit"][f]["theta"] for f in FACTORS}
    effects = {"offense": {}, "defense": {}}
    for side, prefix in (("offense", "off"), ("defense", "def")):
        for cat, names in (("b2b", (f"{prefix}_b2bT", f"{prefix}_b2bS")), ("1", (f"{prefix}_1",))):
            values = {f: float(basis_values[f][BASIS.index(names[0])]) for f in FACTORS}
            if any(abs(values[f]) > 0 for f in FACTORS):
                effects[side][cat] = values

    # joint check: the chosen form plus the home gap, refitted together
    joint = {}
    for k, f in enumerate(FACTORS):
        x = np.column_stack([np.column_stack([z[name][:, k] for name in BASIS]) @ a, z["home_gap"][:, k]])
        theta, cov = wls(e0[:, k], v[:, k], x)
        joint[f] = {"home_gap_change": float(theta[-1]), "home_gap_change_se": float(math.sqrt(cov[-1, -1])),
                    "rest": [float(t) for t in theta[:-1]]}

    by_game = rest.rest_games(sorted({dynamic_model.boxscores.game_key(g) for g, r in
                                      zip(data["cleaned"]["gameId"], rest_rows) if r is not None}))
    return {"forms": forms, "chosen": chosen, "lr": lr, "effects": effects, "joint_home": joint,
            "pace": pace_check(data, by_game), "rest_rows": rest_rows, "row_cats": row_cats, "games": games}


def summarize_counts(rest_rows: list, games: list) -> dict:
    team_games = {}
    for r in rest_rows:
        if r is not None:
            key = basis_category(r[0])
            team_games[key] = team_games.get(key, 0) + 1
    with_b2b = sum(1 for r, g in zip(rest_rows, games)
                   if r is not None and g["side"] != "away" and "b2b" in (r[0]["rest"], r[1]["rest"]))
    both_b2b = sum(1 for r, g in zip(rest_rows, games)
                   if r is not None and g["side"] != "away" and r[0]["rest"] == r[1]["rest"] == "b2b")
    openers = sum(1 for r in rest_rows if r is not None and r[0]["opener"])
    return {"team_games": team_games, "games_with_a_b2b_team": with_b2b, "games_both_b2b": both_b2b,
            "team_openers": openers}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the rest (back-to-back) effect on the training data.")
    parser.add_argument("--csv", default=str(dynamic_model.DEFAULT_CSV), help="Kaggle TeamStatisticsExtended.csv")
    parser.add_argument("--before", default=dynamic_model.TRAINING_CUTOFF, metavar="YYYY-MM-DD",
                        help=f"only games before this date (default {dynamic_model.TRAINING_CUTOFF})")
    parser.add_argument("--years", type=int, default=3, help="window length before --before (default 3)")
    parser.add_argument("--params", default=str(params.PARAMS_PATH), help="model_params.json to add the section to")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing rest section")
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
    if "rest" in model_params and not args.overwrite:
        parser.error(f"{out.name} already has a rest section; pass --overwrite to refit")
    for needed in ("sigma", "home", "points", "pace"):
        if needed not in model_params:
            parser.error(f"{out.name} has no {needed!r} section (needed for the fit)")
    if not Path(args.csv).exists():
        parser.error(f"{args.csv} not found (download the Kaggle CSV into stats/)")
    sigma = params.check_sigma(model_params["sigma"])
    boost = params.check_home(model_params["home"])

    start = time.perf_counter()
    data = dynamic_model.load_training_rows(args.csv, before, args.years)
    fit = estimate_rest(data, sigma, boost)
    seconds = time.perf_counter() - start
    counts = summarize_counts(fit["rest_rows"], fit["games"])
    chosen, forms, effects = fit["chosen"], fit["forms"], fit["effects"]
    line, possessions = league_line(data)
    grad = margin_gradient(line, possessions)

    # margin of a back-to-back team vs a rested one: its line moves by offense["b2b"], the opponent's by
    # defense["b2b"]; per factor that is d(points)/d(factor) x (offense - defense) to first order
    b2b_fit = forms[chosen]["fit"]
    names = forms[chosen]["names"]
    margin = {}
    for cat in ("b2b", "1"):
        if cat in effects["offense"] or cat in effects["defense"]:
            margin[f"{cat} vs 2+"] = margin_points(effects, cat, "2+", line, possessions)
    by_factor = {f: grad[f] * (effects["offense"].get("b2b", {f: 0})[f] - effects["defense"].get("b2b", {f: 0})[f])
                 for f in FACTORS}
    margin_se = None
    if chosen == "difference, b2b":   # one parameter per factor: margin = sum_f 2 grad_f beta_f
        margin_se = math.sqrt(sum((2 * grad[f] * b2b_fit[f]["se"][0]) ** 2 for f in FACTORS))

    pace = fit["pace"]
    pace_meaningful = abs(pace["b2b"]) > 2 * pace["b2b_se"] and pace["log_likelihood_gain"] > 2
    model_params["rest"] = {
        "offense": {cat: {f: round(v, 5) for f, v in values.items()} for cat, values in effects["offense"].items()},
        "defense": {cat: {f: round(v, 5) for f, v in values.items()} for cat, values in effects["defense"].items()},
        "fit": {
            "method": "one-step-ahead (prequential) log-likelihood of the exact training pass (flat start, tau=0, "
                      "fitted sigma, saved home boost) with the rest shift in the forecast and the update; solved "
                      "exactly by weighted least squares (the forecast residuals are linear in the shifts), SE from "
                      "the exact curvature; factors fitted separately; form chosen by AIC among the forms the update "
                      "can use",
            "data": {"csv": Path(args.csv).name, "before": before, "years": args.years, "rows": len(fit["games"]),
                     "games": int(data["games"]), "first": data["first"], "last": data["last"],
                     "rest_source": "NBA.com ScheduleLeagueV2 (stats/schedules/), see rest.py"},
            "form": chosen,
            "counts": counts,
            "estimate": {f: {n: {"value": round(float(t), 5), "se": round(float(s), 5)}
                             for n, t, s in zip(names, b2b_fit[f]["theta"], b2b_fit[f]["se"])} for f in FACTORS},
            "forms": {n: {"parameters": fm["parameters"], "log_likelihood_gain": round(fm["log_likelihood_gain"], 2),
                          "aic_gain": round(fm["aic_gain"], 2), "held_out_gain": round(fm["held_out_gain"], 2),
                          "usable": fm["usable"],
                          "estimate": {f: {n2: [round(float(t), 5), round(float(s), 5)] for n2, t, s in
                                           zip(fm["names"], fm["fit"][f]["theta"], fm["fit"][f]["se"])}
                                       for f in FACTORS}}
                      for n, fm in forms.items()},
            "likelihood_ratio_tests": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
                                       for t in fit["lr"]],
            "held_out": "fit on the first two training seasons, log-likelihood gain on the third",
            "margin_points": {k: round(v, 3) for k, v in margin.items()},
            "margin_points_se": None if margin_se is None else round(margin_se, 3),
            "margin_points_by_factor_b2b": {f: round(by_factor[f], 3) for f in FACTORS},
            "margin_basis": {"line": {f: round(line[f], 4) for f in FACTORS}, "possessions": round(possessions, 2)},
            "pace": {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in pace.items()},
                     "used": bool(pace_meaningful)},
            "joint_home_check": {f: {"home_gap_change": round(j["home_gap_change"], 5),
                                     "se": round(j["home_gap_change_se"], 5)} for f, j in fit["joint_home"].items()},
            "fitted": date.today().isoformat(),
        },
    }
    if pace_meaningful:
        raise SystemExit("the pace effect came out meaningful; the update has no pace rest shift yet - extend "
                         "bayes_update.update_pace / simulation before saving it")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(model_params, indent=4) + "\n", encoding="utf-8")
    tmp.replace(out)

    tg = counts["team_games"]
    print(f"Fitted on {len(fit['games'])} rows ({data['games']} games, {data['first']} to {data['last']}) "
          f"in {seconds:.0f} s")
    print(f"Team-games by rest: b2b traveled {tg.get('b2bT', 0)}, b2b same arena {tg.get('b2bS', 0)}, "
          f"1 day {tg.get('1', 0)}, 2+ {tg.get('2+', 0)} (openers {counts['team_openers']}); "
          f"games with a team on a b2b: {counts['games_with_a_b2b_team']} (both: {counts['games_both_b2b']})\n")
    print(f"{'form':<36}{'params':>7}{'loglik gain':>12}{'AIC gain':>10}{'held-out':>10}")
    for n, fm in forms.items():
        mark = " <- chosen" if n == chosen else ("" if fm["usable"] else "  (check only)")
        print(f"{n:<36}{fm['parameters']:>7}{fm['log_likelihood_gain']:>12.2f}{fm['aic_gain']:>10.2f}"
              f"{fm['held_out_gain']:>10.2f}{mark}")
    print("\nLikelihood-ratio tests:")
    for t in fit["lr"]:
        print(f"  {t['test']:<42} chi2 {t['chi2']:6.2f} on {t['df']} df   p = {t['p']:.2f}")
    print(f"\nChosen: {chosen}. Effect on a line (offense on a b2b; the opposite sign when the defense is):")
    print(f"{'factor':<7}" + "".join(f"{n:>22}" for n in names) + f"{'pts of margin':>16}")
    for f in FACTORS:
        cells = "".join(f"{float(t):>+14.4f} ({float(s):.4f})" for t, s in zip(b2b_fit[f]["theta"], b2b_fit[f]["se"]))
        print(f"{f:<7}{cells}{by_factor[f]:>+16.2f}")
    for k, m in margin.items():
        print(f"Margin, {k} (league-average teams and pace): {m:+.2f} pts"
              + (f" (SE {margin_se:.2f})" if margin_se is not None and k == "b2b vs 2+" else ""))
    print(f"\nPace (possessions per team on rest): b2b {pace['b2b']:+.3f} ({pace['b2b_se']:.3f}), "
          f"1 day {pace['1']:+.3f} ({pace['1_se']:.3f}); log-lik gain {pace['log_likelihood_gain']:.2f} "
          f"-> {'used' if pace_meaningful else 'not used'}")
    print("Home gap refitted jointly with rest (change vs the saved gap): "
          + ", ".join(f"{f} {j['home_gap_change']:+.4f} ({j['home_gap_change_se']:.4f})"
                      for f, j in fit["joint_home"].items()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
