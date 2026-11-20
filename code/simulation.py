"""
Monte Carlo simulation of one game from two teams' curves (10,000 draws).

Per draw:
  1. each side's matchup expectation per factor: (team offense + opponent defense) / 2, with
     the offense and defense values drawn from their curves (rating uncertainty)
  2. game noise on top, sigma_f per factor (params.sim_sigma(): fitted by default)
       sim_corr "off" (default)  every one of the 8 game factors gets independent noise
       sim_corr "on"             the 8 noises are drawn jointly with model_params.json's
                                 factor correlation (both teams, both sides of the ball)
  3. points from the four factors
       pace "off"            100 possessions and the original formula
                             possessions (1 - TOV)(1 + ORB (1 - eFG)) (2 eFG + 0.77 FTR):
                             ~22 points per game too high (it ignores free-throw possessions)
       pace "on" (default)   possessions ~ N((pace_a + pace_b) / 2, sigma_pace^2) with each
                             team's pace drawn from its curve, and the corrected formula
                             points() (constants from model_params.json["points"])
Both off (with legacy sigma and home off) = the original simulation, draw for draw.

home "on" (Phase 8, default): team_a is the home team; the home boost (model_params.json["home"]) is
added to each side's matchup expectation in step 1, except at neutral sites. "off" adds
nothing, so the draws are the same numbers as before.

rest "on" (Phase 9, default "on"): each side's matchup expectation also gets the rest effect
(model_params.json["rest"]) from both teams' rest, rest_a / rest_b ("b2b", "1", "2+"; see
rest.py), added to the home boost. "off" adds nothing.

calib "on" (Phase 10b, default "off"): the win % from the draws, P, is reported as
Phi(a * Phi^-1(P)) (params.calibrate_win_prob, temperature a from model_params.json["calib"]);
the draws, scores and margin SD are unchanged, and "team_a_win_pct_raw" keeps P.
"""
import json

import numpy as np

import params

N_SIM = 10000
FACTORS = params.FACTORS
LEGACY_PACE = 100       # the original fixed possessions per team
LEGACY_FT_PCT = 0.77    # the original free-throw %


def load_priors(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def _draw_curves(rng, curves: dict, n: int) -> dict:
    """One draw per simulation from each factor's curve, in FACTORS order."""
    return {f: rng.normal(loc=curves[f]["mean"], scale=curves[f]["std"], size=n) for f in FACTORS}


def _expected(o, d, offset, f):
    """Matchup expectation (o + d)/2, plus the home-court offset when there is one (None: nothing added)."""
    return (o[f] + d[f]) / 2 if offset is None else (o[f] + d[f]) / 2 + offset[f]


def independent_factors(rng, a: dict, b: dict, sigma: dict, n: int = N_SIM, offset_a=None, offset_b=None):
    """
    The original draws, in the original order: A's offense, B's defense, A's game factors
    (independent noise), then B's offense, A's defense, B's game factors.
    Returns ({factor: A's values}, {factor: B's values}); A's are A's offensive line.
    offset_a / offset_b: {factor: shift} added to A's / B's expected line (home court), or None.
    """
    a_o = _draw_curves(rng, a["offense"], n)
    b_d = _draw_curves(rng, b["defense"], n)
    a_game = {f: rng.normal(loc=_expected(a_o, b_d, offset_a, f), scale=sigma[f], size=n) for f in FACTORS}
    b_o = _draw_curves(rng, b["offense"], n)
    a_d = _draw_curves(rng, a["defense"], n)
    b_game = {f: rng.normal(loc=_expected(b_o, a_d, offset_b, f), scale=sigma[f], size=n) for f in FACTORS}
    return a_game, b_game


def correlated_factors(rng, a: dict, b: dict, sigma: dict, corr, n: int = N_SIM, offset_a=None, offset_b=None):
    """
    Like independent_factors, but the 8 game noises (A's eFG/TOV/ORB/FTR, then B's; the
    order of params.CORR_LABELS) are one multivariate normal draw: sigma_f * (L z), L the
    Cholesky factor of `corr`, z independent standard normals.
    """
    a_o = _draw_curves(rng, a["offense"], n)
    b_d = _draw_curves(rng, b["defense"], n)
    b_o = _draw_curves(rng, b["offense"], n)
    a_d = _draw_curves(rng, a["defense"], n)
    chol = np.linalg.cholesky(np.asarray(corr, dtype=float))
    z = chol @ rng.standard_normal((len(params.CORR_LABELS), n))
    k = len(FACTORS)
    a_game = {f: _expected(a_o, b_d, offset_a, f) + sigma[f] * z[i] for i, f in enumerate(FACTORS)}
    b_game = {f: _expected(b_o, a_d, offset_b, f) + sigma[f] * z[k + i] for i, f in enumerate(FACTORS)}
    return a_game, b_game


def draw_possessions(rng, a: dict, b: dict, sigma_pace: float, n: int = N_SIM):
    """Game possessions per draw: each team's pace from its curve, then N((pace_a + pace_b) / 2, sigma_pace^2)."""
    pace_a = rng.normal(loc=a["pace"]["mean"], scale=a["pace"]["std"], size=n)
    pace_b = rng.normal(loc=b["pace"]["mean"], scale=b["pace"]["std"], size=n)
    return rng.normal(loc=(pace_a + pace_b) / 2, scale=sigma_pace, size=n)


def legacy_points(g: dict, possessions=LEGACY_PACE):
    """The original formula (pace "off"): possessions (1 - TOV)(1 + ORB (1 - eFG)) shots, FT% 0.77."""
    effective_pos = possessions * (1 - g["TOV"]) * (1 + (g["ORB"] * (1 - g["eFG"])))
    return effective_pos * (2 * g["eFG"] + g["FTR"] * LEGACY_FT_PCT)


def points(g: dict, possessions, constants=None):
    """
    The corrected formula (pace "on"), from NBA.com's definitions (see fit_sim.py):
        FGA    = (possessions + offset)(1 - TOV) / (1 + 0.44 FTR - rebound_chances ORB (1 - eFG))
        points = FGA (2 eFG + FTR ft_pct)
    constants: params.points_params() by default.
    """
    c = constants or params.points_params()
    fga = (possessions + c["possession_offset"]) * (1 - g["TOV"]) / (
        1 + 0.44 * g["FTR"] - c["rebound_chances"] * g["ORB"] * (1 - g["eFG"]))
    return fga * (2 * g["eFG"] + g["FTR"] * c["ft_pct"])


def simulate_factors(team_a, team_b, data: dict, seed=None, sigma=None,
                     sim_corr: str = params.DEFAULT_SIM_CORR_MODE, pace: str = params.DEFAULT_PACE_MODE,
                     home: str = params.DEFAULT_HOME_MODE, neutral: bool = False,
                     rest: str = params.DEFAULT_REST_MODE, rest_a=None, rest_b=None,
                     calib: str = params.DEFAULT_CALIB_MODE, pace_sigma=None, calib_temperature=None):
    """
    Simulate team_a vs team_b from the curves in `data` (priors). seed=None: fresh randomness
    (website); an int: reproducible (prediction log). sigma: {factor: game noise}, default
    params.sim_sigma() (fitted). sim_corr / pace: "on" or "off" (see the module docstring).
    home "on": team_a is the home team; its offensive line gets the home boost's "offense"
    shifts and team_b's line the "defense" shifts (params.home_offsets), unless neutral=True.
    rest "on": both lines also get the rest effect from rest_a / rest_b (params.rest_offsets).
    calib "on": the win % is calibrated (params.calibrate_win_prob); team_b's is 100 minus team_a's.
    calib_temperature: the temperature to calibrate with (default params.calib_temperature(), the
    live season settings'; params.calib_temperature("on") for react on, Phase 10c).
    pace_sigma: overrides the game's pace noise (default params.pace_sigma(); fit_calib.py's diagnosis).
    """
    n_sim = N_SIM
    rng = np.random.default_rng(seed)
    sigma = params.check_sigma(sigma) if sigma is not None else params.sim_sigma()
    corr = params.sim_corr(sim_corr)
    with_calib = params.is_on(calib, "calib")
    with_pace = params.is_on(pace, "pace")
    offset_a, offset_b = params.home_offsets(home, neutral)
    if params.is_on(rest, "rest"):
        if rest_a is None or rest_b is None:
            raise ValueError("rest is on but the teams' rest (rest_a, rest_b) is missing")
        rest_shift_a, rest_shift_b = params.rest_offsets(rest, rest_a, rest_b)
        offset_a = params.combine_offsets(offset_a, rest_shift_a)
        offset_b = params.combine_offsets(offset_b, rest_shift_b)
    a, b = data[team_a], data[team_b]
    if with_pace:
        for team, curves in ((team_a, a), (team_b, b)):
            if "pace" not in curves:
                raise KeyError(f"team {team} has no pace curve; use priors trained with --pace on")

    if corr is None:
        a_game, b_game = independent_factors(rng, a, b, sigma, n_sim, offset_a, offset_b)
    else:
        a_game, b_game = correlated_factors(rng, a, b, sigma, corr, n_sim, offset_a, offset_b)

    if with_pace:
        possessions = draw_possessions(rng, a, b, params.pace_sigma() if pace_sigma is None else pace_sigma, n_sim)
        constants = params.points_params()
        a_points, b_points = points(a_game, possessions, constants), points(b_game, possessions, constants)
    else:
        possessions = LEGACY_PACE
        a_points, b_points = legacy_points(a_game), legacy_points(b_game)

    a_mean_points = np.mean(a_points)
    b_mean_points = np.mean(b_points)

    a_games_won = a_points > b_points
    b_games_won = b_points > a_points

    a_win_pct = np.sum(a_games_won)
    b_win_pct = np.sum(b_games_won)
    raw_a = (a_win_pct / n_sim) * 100   # Convert raw count to percentage
    if with_calib:
        team_a_pct = 100 * params.calibrate_win_prob(raw_a / 100, calib, calib_temperature)
        team_b_pct = 100 - team_a_pct
    else:
        team_a_pct, team_b_pct = raw_a, (b_win_pct / n_sim) * 100

    return {
        "team_a_avg_score": a_mean_points,
        "team_b_avg_score": b_mean_points,
        "team_a_win_pct": team_a_pct,
        "team_b_win_pct": team_b_pct,
        "team_a_win_pct_raw": raw_a,                      # before any calibration
        "margin_sd": float(np.std(a_points - b_points)),   # spread of the simulated final margins
        "possessions": float(np.mean(possessions)),
    }
if __name__ == "__main__":
    # Load the team profiles into memory
    priors_data = load_priors("priors.json")

    # temporary games since off season currently
    games = [
        {"a_name": "BOS", "a_id": "1610612738", "b_name": "WAS", "b_id": "1610612764"},
        {"a_name": "LAL", "a_id": "1610612747", "b_name": "DEN", "b_id": "1610612743"},
        {"a_name": "GSW", "a_id": "1610612744", "b_name": "PHX", "b_id": "1610612756"}
    ]

    print("Running 10,000 Monte Carlo Simulations for 3 games...\n")

    all_cards = []
    
    for game in games:
        # Run the simulation for this matchup (no dates here: both teams rested, so rest adds nothing)
        results = simulate_factors(game["a_id"], game["b_id"], priors_data, rest_a="2+", rest_b="2+")

        # Determine who is favored for the spread text
        spread = results['team_a_avg_score'] - results['team_b_avg_score']
        if spread > 0:
            spread_text = f"{game['a_name']} by {spread:.1f}"
        else:
            spread_text = f"{game['b_name']} by {abs(spread):.1f}"

        # Create a list of strings for this specific game's "card"
        # .ljust(35) forces every line to be exactly 35 characters wide so they align perfectly
        card = [
            "=" * 35,
            f" {game['a_name']} vs {game['b_name']}".center(35),
            "=" * 35,
            f"{game['a_name']} Win %: {results['team_a_win_pct']:.1f}%".ljust(35),
            f"{game['b_name']} Win %: {results['team_b_win_pct']:.1f}%".ljust(35),
            "-" * 35,
            "Projected Final Score:".ljust(35),
            f"{game['a_name']}: {results['team_a_avg_score']:.1f}".ljust(35),
            f"{game['b_name']}: {results['team_b_avg_score']:.1f}".ljust(35),
            "-" * 35,
            f"Spread: {spread_text}".ljust(35),
            "=" * 35
        ]
        all_cards.append(card)

    # The Magic Trick: zip(*all_cards) groups row 1 of all cards, then row 2, etc.
    for row in zip(*all_cards):
        # Join the horizontal rows with 4 spaces of padding in between
        print("    ".join(row))