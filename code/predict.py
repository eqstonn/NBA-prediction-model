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

from simulation import load_priors, simulate_factors

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRIORS = REPO_ROOT / "priors.json"


def load_model(priors_path=None):
    path = Path(priors_path or os.environ.get("NBA_PRIORS") or DEFAULT_PRIORS)
    return load_priors(path), path


def predict_game(home_id: str, away_id: str, model: dict, seed=None):
    """
    Returns:
        {"home_win_pct": float, "away_win_pct": float,
         "home_score": float, "away_score": float}
    or None if either team isn't in the model. `seed` makes the simulation reproducible.
    """
    if home_id not in model or away_id not in model:
        return None
    r = simulate_factors(home_id, away_id, model, seed=seed)
    return {
        "home_win_pct": float(r["team_a_win_pct"]),
        "away_win_pct": float(r["team_b_win_pct"]),
        "home_score": float(r["team_a_avg_score"]),
        "away_score": float(r["team_b_avg_score"]),
    }
