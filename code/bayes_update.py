"""
The per-game Bayesian update, shared by training (dynamic_model.py, tau=0) and the
in-season updater (update.py, tau=0.1/M).

One call = one team's offensive line from one game, exactly like one row of the
training data:

    game = {"team": "1610612765", "opponent": "1610612738",       # DET's offense vs BOS's defense
            "eFG": 0.541, "TOV": 0.168, "ORB": 0.354, "FTR": 0.372}

For each factor f the model is
    offense o ~ Normal(m_o, s_o^2),  defense d ~ Normal(m_d, s_d^2)
    observed y ~ Normal((o + d) / 2, sigma_f^2)
and the team's offense curve and the opponent's defense curve are replaced by their
posterior mean/std. A full game is two calls, one per side.

sigma_f is the game-to-game noise of factor f: by default the fitted values in
model_params.json (params.update_sigma()); pass sigma=params.LEGACY_UPDATE_SIGMA (0.05 for
every factor) for the pre-Phase-6 model. (The simulation's noise is set separately.)

Two engines compute that posterior:
  "exact" (default)  the closed-form conjugate (Kalman) update below: exact, instant, no noise
  "pymc"             the original PyMC model, sampled with NUTS: the same answer plus
                     sampling noise (median 0.0012 off over a season), ~2 s per call
Both write back each curve's own (marginal) mean and std.

Pace (Phase 7): update_pace updates both teams' pace curves (priors[team]["pace"]) from a
game's possessions with the same exact formula (conjugate_pair), pace_a and pace_b in
place of offense and defense.
"""
import math
import os

# PyTensor reads PYTENSOR_FLAGS once, when it is first imported, so this has to run
# before `import pymc`. cxx= : there is no C++ compiler on this machine; don't look for one.
_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "cxx=" not in _flags:
    os.environ["PYTENSOR_FLAGS"] = f"{_flags},cxx=" if _flags else "cxx="

import params  # noqa: E402

FACTORS = params.FACTORS
ENGINES = ("exact", "pymc")
DEFAULT_ENGINE = "exact"


def widen(std: float, tau: float) -> float:
    """Add tau of drift to a curve before a game: sqrt(std^2 + tau^2)."""
    return math.sqrt(std ** 2 + tau ** 2)


def _pymc():
    """Import pymc on first use only (the exact engine never needs it). PYTENSOR_FLAGS is already set above."""
    import pymc  # noqa: E402  (must come after PYTENSOR_FLAGS is set)
    return pymc


def predictive(prior: dict, f: str, sigma: dict):
    """(mean, variance) of factor f's observation before the game: ((m_o + m_d)/2, (s_o^2 + s_d^2)/4 + sigma_f^2)."""
    (m_o, s_o), (m_d, s_d) = prior["o"][f], prior["d"][f]
    return (m_o + m_d) / 2, (s_o ** 2 + s_d ** 2) / 4 + sigma[f] ** 2


def conjugate_pair(m_1: float, s_1: float, m_2: float, s_2: float, y: float, sigma: float):
    """
    Exact posterior of two Normal curves x1 ~ N(m_1, s_1^2), x2 ~ N(m_2, s_2^2) after one
    observation y ~ N((x1 + x2) / 2, sigma^2):
        V        = (s_1^2 + s_2^2) / 4 + sigma^2      variance of the observation before the game
        surprise = y - (m_1 + m_2) / 2
        m_i'     = m_i + (s_i^2 / 2) / V * surprise
        s_i'     = sqrt(s_i^2 - (s_i^2 / 2)^2 / V)
    Returns ((m_1', s_1'), (m_2', s_2')). Used for every factor (offense + defense) and for pace.
    """
    v = (s_1 ** 2 + s_2 ** 2) / 4 + sigma ** 2
    surprise = y - (m_1 + m_2) / 2
    out = []
    for m, s in ((m_1, s_1), (m_2, s_2)):
        covariance = s ** 2 / 2   # between this curve and the observation
        out.append((m + covariance / v * surprise, math.sqrt(s ** 2 - covariance ** 2 / v)))
    return tuple(out)


def _fit_exact(prior: dict, observed: dict, sigma: dict) -> dict:
    """Closed-form posterior of the model above, per factor (conjugate_pair with offense and defense)."""
    out = {"o": {}, "d": {}}
    for f in FACTORS:
        (m_o, s_o), (m_d, s_d) = prior["o"][f], prior["d"][f]
        out["o"][f], out["d"][f] = conjugate_pair(m_o, s_o, m_d, s_d, observed[f], sigma[f])
    return out


def _fit_pymc(prior: dict, observed: dict, random_seed, sigma: dict) -> dict:
    """The PyMC model (as in the original dynamic_model.py, with per-factor sigma), sampled with NUTS."""
    pm = _pymc()
    with pm.Model():
        # current baseline offensively, then defensively
        o_skill = {f: pm.Normal(f"o_{f}_skill", mu=prior["o"][f][0], sigma=prior["o"][f][1]) for f in FACTORS}
        d_skill = {f: pm.Normal(f"d_{f}_skill", mu=prior["d"][f][0], sigma=prior["d"][f][1]) for f in FACTORS}

        # account for strength of schedule, then give the model the evidence
        for f in FACTORS:
            expected = (o_skill[f] + d_skill[f]) / 2
            pm.Normal(f"{f}_evidence", mu=expected, sigma=sigma[f], observed=observed[f])

        # simulate to find the new baseline (new bell curve)
        trace = pm.sample(1000, progressbar=False, cores=1, random_seed=random_seed)

    post = trace.posterior
    return {side: {f: (float(post[f"{side}_{f}_skill"].mean()), float(post[f"{side}_{f}_skill"].std()))
                   for f in FACTORS}
            for side in ("o", "d")}


def _fit(prior: dict, observed: dict, random_seed=None, engine: str = DEFAULT_ENGINE, sigma=None) -> dict:
    """
    prior    = {"o": {factor: (mean, std)}, "d": {factor: (mean, std)}}
    observed = {factor: value}
    sigma    = {factor: game noise}; None = params.update_sigma() (fitted)
    returns    {"o": {factor: (mean, std)}, "d": {factor: (mean, std)}} of the posterior
    """
    sigma = params.check_sigma(sigma) if sigma is not None else params.update_sigma()
    if engine == "exact":
        return _fit_exact(prior, observed, sigma)
    if engine == "pymc":
        return _fit_pymc(prior, observed, random_seed, sigma)
    raise ValueError(f"unknown engine {engine!r}; use one of {ENGINES}")


def update_game(priors: dict, game: dict, tau: float = 0.0, random_seed=None, engine: str = DEFAULT_ENGINE,
                sigma=None) -> dict:
    """
    Update `priors` in place from one team's offensive line in one game; returns `priors`.

    tau > 0 widens the curves in this fit (team offense, opponent defense) to
    sqrt(std^2 + tau^2) before fitting. Over a full game (one call per side) that widens
    each of the two teams' 8 curves exactly once. tau = 0 is the training update.
    random_seed only matters for the pymc engine. sigma: {factor: noise}, default fitted.
    """
    team, opponent = str(game["team"]), str(game["opponent"])
    offense = priors[team]["offense"]
    defense = priors[opponent]["defense"]

    prior = {
        "o": {f: (offense[f]["mean"], widen(offense[f]["std"], tau) if tau else offense[f]["std"]) for f in FACTORS},
        "d": {f: (defense[f]["mean"], widen(defense[f]["std"], tau) if tau else defense[f]["std"]) for f in FACTORS},
    }
    posterior = _fit(prior, {f: game[f] for f in FACTORS}, random_seed, engine=engine, sigma=sigma)

    # only written once the fit succeeded, so a failed fit leaves priors untouched
    for f in FACTORS:
        offense[f]["mean"], offense[f]["std"] = posterior["o"][f]
        defense[f]["mean"], defense[f]["std"] = posterior["d"][f]
    return priors


# ---------------------------------------------------------------------------
# pace (Phase 7)
# ---------------------------------------------------------------------------

def pace_predictive(priors: dict, team_a: str, team_b: str, sigma: float):
    """(mean, variance) of a game's possessions before it: ((p_a + p_b)/2, (s_a^2 + s_b^2)/4 + sigma^2)."""
    a, b = priors[str(team_a)]["pace"], priors[str(team_b)]["pace"]
    return (a["mean"] + b["mean"]) / 2, (a["std"] ** 2 + b["std"] ** 2) / 4 + sigma ** 2


def update_pace(priors: dict, team_a: str, team_b: str, possessions: float, tau: float = 0.0, sigma=None) -> dict:
    """
    Update both teams' pace curves in place from one game's possessions; returns `priors`.

    Model (the same shape as one factor): pace_a ~ N(m_a, s_a^2), pace_b ~ N(m_b, s_b^2),
    possessions ~ N((pace_a + pace_b) / 2, sigma^2), solved exactly with conjugate_pair.
    tau > 0 first widens both curves to sqrt(std^2 + tau^2), once per game.
    sigma: game noise in possessions; None = model_params.json["pace"]["sigma"].
    """
    sigma = params.pace_sigma() if sigma is None else float(sigma)
    if not sigma > 0:
        raise ValueError(f"pace sigma must be positive, got {sigma}")
    a, b = priors[str(team_a)]["pace"], priors[str(team_b)]["pace"]
    s_a = widen(a["std"], tau) if tau else a["std"]
    s_b = widen(b["std"], tau) if tau else b["std"]
    (a["mean"], a["std"]), (b["mean"], b["std"]) = conjugate_pair(a["mean"], s_a, b["mean"], s_b,
                                                                 float(possessions), sigma)
    return priors
