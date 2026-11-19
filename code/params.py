"""
Model parameters for the update (bayes_update.py) and the simulation (simulation.py).

Game-to-game noise sigma, one value per four factor, chosen separately for the two:

  "fitted"  model_params.json["sigma"], estimated by code/fit_sigma.py from the training
            data only (eFG 0.0637, TOV 0.0364, ORB 0.0719, FTR 0.0854).
  "legacy"  the values before Phase 6: the update used 0.05 for every factor, the
            simulation 0.05 / 0.02 / 0.05 / 0.02 (eFG / TOV / ORB / FTR).

Defaults: fitted for both, the live model since the pace-home switch (see below). With the
OLD points formula (pace off), the fitted sigma spreads final margins too wide (SD 19.9 pts
vs 15.5 real); legacy sim sigma was the better match there (Phase 6, log_sigma-update.csv).
With pace on, the corrected formula gets the scale right and the fitted sigma calibrates.
Legacy for both + every switch off reproduces log_exact.csv exactly.

LIVE DEFAULTS (pace-home): exact engine, sigma update fitted, sigma sim fitted, pace on,
sim_corr off, home on. Picked on the training data (train_calibration.py: Brier 0.2264 vs
0.2346 for the old live simulation, 5.5 paired SE); log_pace-home.csv is its 2026-27 replay.
The old live model (Phase 6) is --sigma-sim legacy --pace off --home off.

Simulation switches (Phase 7), each "on" or "off" (off = the old simulation):

  sim_corr  draw the 8 game factors (both teams' eFG/TOV/ORB/FTR) jointly with the
            correlation in model_params.json["factor_corr"] instead of independently
  pace      possessions from the two teams' pace curves (priors[team]["pace"]) plus game
            noise model_params.json["pace"]["sigma"], and the corrected points formula
            (model_params.json["points"]); off = 100 possessions and the old formula

Defaults: sim_corr off, pace on. Both off with legacy sim sigma and home off reproduces
log_sigma-update.csv exactly.
code/fit_sim.py estimates factor_corr, points and pace from the training data only.

Home court (Phase 8), "on" or "off", default "on":

  home      add model_params.json["home"] to the expected value of each line in the
            update and the simulation: the home team's offensive line gets
            home["offense"][f], the away team's offensive line (what the home defense
            allows) gets home["defense"][f]. Neutral-site games get nothing.
            code/fit_home.py estimates it from the training data only.
"""
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO_ROOT / "model_params.json"

FACTORS = ("eFG", "TOV", "ORB", "FTR")
SIGMA_MODES = ("fitted", "legacy")
DEFAULT_UPDATE_SIGMA_MODE = "fitted"
DEFAULT_SIM_SIGMA_MODE = "fitted"

LEGACY_UPDATE_SIGMA = {"eFG": 0.05, "TOV": 0.05, "ORB": 0.05, "FTR": 0.05}
LEGACY_SIM_SIGMA = {"eFG": 0.05, "TOV": 0.02, "ORB": 0.05, "FTR": 0.02}

SWITCH_MODES = ("on", "off")
DEFAULT_SIM_CORR_MODE = "off"
DEFAULT_PACE_MODE = "on"
DEFAULT_HOME_MODE = "on"
# rows/columns of the factor correlation: the team's offensive line, then the opponent's
CORR_LABELS = tuple(f"team {f}" for f in FACTORS) + tuple(f"opponent {f}" for f in FACTORS)

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
    """Game noise used by the Monte Carlo simulation, per factor (default: fitted)."""
    _check_mode(mode)
    return fitted_sigma() if mode == "fitted" else dict(LEGACY_SIM_SIGMA)


# ---------------------------------------------------------------------------
# Phase 7: correlated factors, pace, points formula
# ---------------------------------------------------------------------------

def is_on(mode: str, what: str = "switch") -> bool:
    """True for "on", False for "off"; anything else is an error."""
    if mode not in SWITCH_MODES:
        raise ValueError(f"unknown {what} setting {mode!r}; use one of {SWITCH_MODES}")
    return mode == "on"


FITTED_BY = {"factor_corr": "fit_sim.py", "points": "fit_sim.py", "pace": "fit_sim.py", "home": "fit_home.py"}


def section(name: str) -> dict:
    data = load_params()
    if name not in data:
        raise KeyError(f"{PARAMS_PATH.name} has no {name!r} section; run python code/{FITTED_BY.get(name, 'fit_sim.py')}")
    return data[name]


def check_corr(matrix) -> list:
    """An 8x8 correlation matrix (CORR_LABELS order): symmetric, unit diagonal, entries in [-1, 1]."""
    n = len(CORR_LABELS)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"factor correlation must be {n}x{n}")
    for i in range(n):
        if abs(matrix[i][i] - 1) > 1e-9:
            raise ValueError("factor correlation must have 1 on the diagonal")
        for j in range(n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-9 or not -1 <= matrix[i][j] <= 1:
                raise ValueError("factor correlation must be symmetric with entries in [-1, 1]")
    return [[float(v) for v in row] for row in matrix]


def factor_corr() -> list:
    """The fitted 8x8 correlation of the game factors' noise (model_params.json["factor_corr"])."""
    return check_corr(section("factor_corr")["matrix"])


def sim_corr(mode: str = DEFAULT_SIM_CORR_MODE):
    """The correlation matrix the simulation draws with: None when off (independent draws)."""
    return factor_corr() if is_on(mode, "sim_corr") else None


def points_params() -> dict:
    """Constants of the corrected points formula: {"ft_pct", "possession_offset", "rebound_chances"}."""
    p = section("points")
    return {k: float(p[k]) for k in ("ft_pct", "possession_offset", "rebound_chances")}


def pace_params() -> dict:
    """{"sigma": game noise in possessions, "start_mean", "start_std": training's flat start}."""
    p = section("pace")
    out = {k: float(p[k]) for k in ("sigma", "start_mean", "start_std")}
    if not all(v > 0 for v in out.values()):
        raise ValueError(f"pace parameters must be positive, got {out}")
    return out


def pace_sigma() -> float:
    return pace_params()["sigma"]


def pace_scale() -> float:
    """
    Season reset / drift scale for pace, the same rule as the ratings in pace units:
    ratings use 0.1 = 2 x the legacy factor noise 0.05 (std 0.1/sqrt(N), tau 0.1/M), so
    pace uses 2 x its own game noise (std 2 sigma_pace / sqrt(N), tau 2 sigma_pace / M).
    """
    return 2 * pace_sigma()


def pace_season_std(n: int) -> float:
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    return pace_scale() / math.sqrt(n)


def pace_tau(m: int) -> float:
    if m <= 0:
        raise ValueError(f"m must be a positive integer, got {m}")
    return pace_scale() / m


# ---------------------------------------------------------------------------
# Phase 8: home court
# ---------------------------------------------------------------------------

def check_home(boost: dict) -> dict:
    """{"offense": {factor: shift}, "defense": {factor: shift}} with a number for every factor."""
    out = {}
    for side in ("offense", "defense"):
        values = boost.get(side, {})
        if set(values) != set(FACTORS) or not all(isinstance(v, (int, float)) and abs(v) < 0.2 for v in values.values()):
            raise ValueError(f"home boost {side} must have a small number for each of {FACTORS}, got {values}")
        out[side] = {f: float(values[f]) for f in FACTORS}
    return out


def home_boost() -> dict:
    """The fitted home boost (model_params.json["home"])."""
    return check_home(section("home"))


def home_offsets(mode: str = DEFAULT_HOME_MODE, neutral: bool = False, boost=None):
    """
    (offset of the home team's offensive line, offset of the away team's offensive line),
    each {factor: shift} to add to the expected value, or (None, None) when home is off or
    the game is at a neutral site. boost: default home_boost().
    """
    if not is_on(mode, "home") or neutral:
        return None, None
    b = check_home(boost) if boost is not None else home_boost()
    return b["offense"], b["defense"]
