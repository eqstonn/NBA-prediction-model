"""
The per-game Bayesian update, shared by training (dynamic_model.py, tau=0) and the
in-season updater (update.py, tau=0.1/M).

One call = one team's offensive line from one game, exactly like one row of the
training data:

    game = {"team": "1610612765", "opponent": "1610612738",       # DET's offense vs BOS's defense
            "eFG": 0.541, "TOV": 0.168, "ORB": 0.354, "FTR": 0.372}

It fits the team's offense curves and the opponent's defense curves to the observed
factors (expected factor = (offense + defense) / 2, observation sigma 0.05) and writes
the posterior mean/std back into `priors`. A full game is two calls, one per side.
"""
import math
import os

# PyTensor reads PYTENSOR_FLAGS once, when it is first imported, so this has to run
# before `import pymc`. cxx= : there is no C++ compiler on this machine; don't look for one.
_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "cxx=" not in _flags:
    os.environ["PYTENSOR_FLAGS"] = f"{_flags},cxx=" if _flags else "cxx="

import pymc as pm  # noqa: E402  (must come after PYTENSOR_FLAGS is set)

FACTORS = ("eFG", "TOV", "ORB", "FTR")
OBS_SIGMA = 0.05


def widen(std: float, tau: float) -> float:
    """Add tau of drift to a curve before a game: sqrt(std^2 + tau^2)."""
    return math.sqrt(std ** 2 + tau ** 2)


def _fit(prior: dict, observed: dict, random_seed=None) -> dict:
    """
    The PyMC model (unchanged from the original dynamic_model.py).

    prior    = {"o": {factor: (mean, std)}, "d": {factor: (mean, std)}}
    observed = {factor: value}
    returns    {"o": {factor: (mean, std)}, "d": {factor: (mean, std)}} of the posterior
    """
    with pm.Model():
        # current baseline offensively, then defensively
        o_skill = {f: pm.Normal(f"o_{f}_skill", mu=prior["o"][f][0], sigma=prior["o"][f][1]) for f in FACTORS}
        d_skill = {f: pm.Normal(f"d_{f}_skill", mu=prior["d"][f][0], sigma=prior["d"][f][1]) for f in FACTORS}

        # account for strength of schedule, then give the model the evidence
        for f in FACTORS:
            expected = (o_skill[f] + d_skill[f]) / 2
            pm.Normal(f"{f}_evidence", mu=expected, sigma=OBS_SIGMA, observed=observed[f])

        # simulate to find the new baseline (new bell curve)
        trace = pm.sample(1000, progressbar=False, cores=1, random_seed=random_seed)

    post = trace.posterior
    return {side: {f: (float(post[f"{side}_{f}_skill"].mean()), float(post[f"{side}_{f}_skill"].std()))
                   for f in FACTORS}
            for side in ("o", "d")}


def update_game(priors: dict, game: dict, tau: float = 0.0, random_seed=None) -> dict:
    """
    Update `priors` in place from one team's offensive line in one game; returns `priors`.

    tau > 0 widens the curves in this fit (team offense, opponent defense) to
    sqrt(std^2 + tau^2) before fitting. Over a full game (one call per side) that widens
    each of the two teams' 8 curves exactly once. tau = 0 is the original training update.
    """
    team, opponent = str(game["team"]), str(game["opponent"])
    offense = priors[team]["offense"]
    defense = priors[opponent]["defense"]

    prior = {
        "o": {f: (offense[f]["mean"], widen(offense[f]["std"], tau) if tau else offense[f]["std"]) for f in FACTORS},
        "d": {f: (defense[f]["mean"], widen(defense[f]["std"], tau) if tau else defense[f]["std"]) for f in FACTORS},
    }
    posterior = _fit(prior, {f: game[f] for f in FACTORS}, random_seed)

    # only written once the fit succeeded, so a failed fit leaves priors untouched
    for f in FACTORS:
        offense[f]["mean"], offense[f]["std"] = posterior["o"][f]
        defense[f]["mean"], defense[f]["std"] = posterior["d"][f]
    return priors
