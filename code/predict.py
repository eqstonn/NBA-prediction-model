"""
The one place the website talks to the model.

To use a retrained model, either overwrite priors.json, or point the site at a
different file:
    python code/build_site.py --priors path/to/new_priors.json
    (or set NBA_PRIORS=path/to/new_priors.json)

If the retrained model changes its interface, only predict_game() needs updating,
as long as it keeps returning the same dict.
"""
import os
from pathlib import Path

import params
from simulation import load_priors, simulate_factors

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRIORS = REPO_ROOT / "priors.json"


def load_model(priors_path=None):
    path = Path(priors_path or os.environ.get("NBA_PRIORS") or DEFAULT_PRIORS)
    return load_priors(path), path


def predict_game(home_id: str, away_id: str, model: dict, seed=None, sigma=None,
                 sim_corr: str = params.DEFAULT_SIM_CORR_MODE, pace: str = params.DEFAULT_PACE_MODE,
                 home: str = params.DEFAULT_HOME_MODE, neutral: bool = False,
                 rest: str = params.DEFAULT_REST_MODE, rest_home=None, rest_away=None,
                 calib: str = params.DEFAULT_CALIB_MODE):
    """
    Returns:
        {"home_win_pct": float, "away_win_pct": float,
         "home_score": float, "away_score": float}
    or None if either team isn't in the model. `seed` makes the simulation reproducible;
    `sigma` overrides the simulation's game noise (default params.sim_sigma(): fitted);
    sim_corr / pace / home: "on" or "off" (defaults in params.py: sim_corr off, pace on,
    home on; see simulation.py). pace on needs team paces in `model`. neutral=True:
    a neutral-site game, no home boost. rest="on" (the default): rest_home / rest_away are
    the teams' rest ("b2b", "1", "2+"; rest.py). calib="on": calibrated win % (Phase 10b,
    params.calibrate_win_prob; default "off"); the scores are the same either way.
    """
    if home_id not in model or away_id not in model:
        return None
    r = simulate_factors(home_id, away_id, model, seed=seed, sigma=sigma, sim_corr=sim_corr, pace=pace,
                         home=home, neutral=neutral, rest=rest, rest_a=rest_home, rest_b=rest_away, calib=calib)
    return {
        "home_win_pct": float(r["team_a_win_pct"]),
        "away_win_pct": float(r["team_b_win_pct"]),
        "home_score": float(r["team_a_avg_score"]),
        "away_score": float(r["team_b_avg_score"]),
    }
