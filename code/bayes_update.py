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


def _fit_exact(prior: dict, observed: dict, sigma: dict) -> dict:
    """
    Closed-form posterior of the model above, per factor:
        V        = (s_o^2 + s_d^2) / 4 + sigma_f^2    variance of the observation before the game
        surprise = y - (m_o + m_d) / 2
        m_o'     = m_o + (s_o^2 / 2) / V * surprise
        s_o'     = sqrt(s_o^2 - (s_o^2 / 2)^2 / V)
    and the same for the defense with s_d.
    """
    out = {"o": {}, "d": {}}
    for f in FACTORS:
        (m_o, s_o), (m_d, s_d) = prior["o"][f], prior["d"][f]
        v = (s_o ** 2 + s_d ** 2) / 4 + sigma[f] ** 2
        surprise = observed[f] - (m_o + m_d) / 2
        for side, m, s in (("o", m_o, s_o), ("d", m_d, s_d)):
            covariance = s ** 2 / 2   # between this curve and the observation
            out[side][f] = (m + covariance / v * surprise, math.sqrt(s ** 2 - covariance ** 2 / v))
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
