"""
Model parameters for the update (bayes_update.py) and the simulation (simulation.py).

Game-to-game noise sigma, one value per four factor, chosen separately for the two:

  "fitted"  model_params.json["sigma"], estimated by code/fit_sigma.py from the training
            data only (eFG 0.0637, TOV 0.0364, ORB 0.0719, FTR 0.0854).
  "legacy"  the values before Phase 6: the update used 0.05 for every factor, the
            simulation 0.05 / 0.02 / 0.05 / 0.02 (eFG / TOV / ORB / FTR).

Defaults: fitted for the UPDATE, legacy for the SIMULATION. That split scored best on
2026-27 (Brier 0.2057). With the fitted sigma, the simulation draws the four factors
independently and spreads final margins too wide (SD 19.9 pts vs 15.5 real), so its
win probabilities are too timid. Legacy for both reproduces log_exact.csv exactly.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO_ROOT / "model_params.json"

FACTORS = ("eFG", "TOV", "ORB", "FTR")
SIGMA_MODES = ("fitted", "legacy")
DEFAULT_UPDATE_SIGMA_MODE = "fitted"
DEFAULT_SIM_SIGMA_MODE = "legacy"

LEGACY_UPDATE_SIGMA = {"eFG": 0.05, "TOV": 0.05, "ORB": 0.05, "FTR": 0.05}
LEGACY_SIM_SIGMA = {"eFG": 0.05, "TOV": 0.02, "ORB": 0.05, "FTR": 0.02}

_cache = {}


def load_params(path=None) -> dict:
    """model_params.json (re-read only when the file changes)."""
    path = Path(path or PARAMS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run python code/fit_sigma.py to create it")
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _cache:
        _cache.clear()
        _cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[key]


def check_sigma(sigma: dict) -> dict:
    if set(sigma) != set(FACTORS) or not all(isinstance(v, (int, float)) and v > 0 for v in sigma.values()):
        raise ValueError(f"sigma must have a positive value for each of {FACTORS}, got {sigma}")
    return {f: float(sigma[f]) for f in FACTORS}


def fitted_sigma() -> dict:
    return check_sigma(load_params()["sigma"])


def _check_mode(mode: str):
    if mode not in SIGMA_MODES:
        raise ValueError(f"unknown sigma mode {mode!r}; use one of {SIGMA_MODES}")


def update_sigma(mode: str = DEFAULT_UPDATE_SIGMA_MODE) -> dict:
    """Observation noise used by the Bayesian update, per factor (default: fitted)."""
    _check_mode(mode)
    return fitted_sigma() if mode == "fitted" else dict(LEGACY_UPDATE_SIGMA)


def sim_sigma(mode: str = DEFAULT_SIM_SIGMA_MODE) -> dict:
    """Game noise used by the Monte Carlo simulation, per factor (default: legacy)."""
    _check_mode(mode)
    return fitted_sigma() if mode == "fitted" else dict(LEGACY_SIM_SIGMA)
