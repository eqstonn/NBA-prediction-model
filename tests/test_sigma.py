"""
Tests for the per-factor game noise sigma (code/params.py, code/fit_sigma.py):

  - sigma is read from model_params.json
  - the update and the simulation take their sigma separately (--sigma-update / --sigma-sim),
    with default fitted for both everywhere (the live pace-home model; Phase 6 had legacy sim)
  - legacy for both (pace and home off) reproduces the pre-Phase-6 results exactly
The noise-scale spies use pace, home and rest off (the toy priors have no paces).
  - the estimator recovers a known sigma

Run from the repo root:  python -m pytest tests

The two "reproduces" tests read the real box-score cache (stats/boxscores/) and the
Kaggle CSV, which are not in git; they are skipped when those files are missing.
"""
import json
import math
import random

import numpy as np
import pytest

from fake_nba import REPO_ROOT, update

import bayes_update  # noqa: E402  (code/ is on sys.path via fake_nba)
import build_site  # noqa: E402
import dynamic_model  # noqa: E402
import fit_sigma  # noqa: E402
import params  # noqa: E402
import predict  # noqa: E402
import simulation  # noqa: E402
from state_manager import default_priors  # noqa: E402

FACTORS = params.FACTORS
MODEL_PARAMS = json.loads((REPO_ROOT / "model_params.json").read_text(encoding="utf-8"))
FITTED = MODEL_PARAMS["sigma"]
LEGACY_SIM = [0.05, 0.02, 0.05, 0.02] * 2   # eFG TOV ORB FTR, for each team


def toy_priors(std=0.02):
    """Two teams, every factor N(mean, std^2): HOME off 0.50 / def 0.56, AWAY off 0.52 / def 0.54."""
    def team(off, dfn):
        return {"offense": {f: {"mean": off, "std": std} for f in FACTORS},
                "defense": {f: {"mean": dfn, "std": std} for f in FACTORS}}
    return {"HOME": team(0.50, 0.56), "AWAY": team(0.52, 0.54)}


def expected_after_one_fit(sigma_f, s=0.02, m_o=0.50, m_d=0.54, y=0.60):
    """Hand formula for HOME offense after HOME (0.50) vs AWAY defense (0.54) with y = 0.60."""
    v = (s ** 2 + s ** 2) / 4 + sigma_f ** 2
    return m_o + (s ** 2 / 2) / v * (y - (m_o + m_d) / 2), math.sqrt(s ** 2 - (s ** 2 / 2) ** 2 / v)


def update_uses(expected_sigma, **kwargs):
    """Run one toy fit (default sigma unless kwargs pass one) and check it matches the hand formula with expected_sigma."""
    priors = toy_priors()
    bayes_update.update_game(priors, {"team": "HOME", "opponent": "AWAY", **{f: 0.60 for f in FACTORS}}, **kwargs)
    return all(priors["HOME"]["offense"][f]["mean"] == pytest.approx(expected_after_one_fit(expected_sigma[f])[0], abs=1e-15)
               and priors["HOME"]["offense"][f]["std"] == pytest.approx(expected_after_one_fit(expected_sigma[f])[1], abs=1e-15)
               for f in FACTORS)


@pytest.fixture
def sim_scales(monkeypatch):
    """Records the `scale` of every game-noise draw (loc = an array of matchup expectations) the simulation makes."""
    scales = []
    real_rng = simulation.np.random.default_rng

    class Spy:
        def __init__(self, rng):
            self.rng = rng

        def normal(self, loc, scale, size):
            if isinstance(loc, np.ndarray):
                scales.append(scale)
            return self.rng.normal(loc=loc, scale=scale, size=size)

    monkeypatch.setattr(simulation.np.random, "default_rng", lambda seed=None: Spy(real_rng(seed)))
    return scales


@pytest.fixture
def custom_params(tmp_path, monkeypatch):
    custom = {"eFG": 0.031, "TOV": 0.042, "ORB": 0.063, "FTR": 0.074}
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps({"sigma": custom}), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    return custom


# ---------------------------------------------------------------------------
# model_params.json
# ---------------------------------------------------------------------------

def test_model_params_file_holds_the_fit():
    assert set(FITTED) == set(FACTORS) and all(0.005 < v < 0.2 for v in FITTED.values())
    assert params.fitted_sigma() == FITTED
    data = MODEL_PARAMS["sigma_fit"]["data"]
    assert data["before"] <= dynamic_model.TRAINING_CUTOFF   # no 2026-27 game was used
    assert data["last"] < "2026-10-03"
    # at the maximum-likelihood sigma the standardized residuals have variance ~1
    assert all(0.95 < z < 1.05 for z in MODEL_PARAMS["sigma_fit"]["mean_squared_standardized_residual"].values())
    assert all(MODEL_PARAMS["sigma_fit"]["log_likelihood"][f] > MODEL_PARAMS["sigma_fit"]["log_likelihood_at_0.05"][f]
               for f in FACTORS)


def test_fitted_sigma_is_read_from_model_params(custom_params, sim_scales):
    assert params.update_sigma("fitted") == params.sim_sigma("fitted") == custom_params
    assert update_uses(custom_params)                                   # update default = fitted
    simulation.simulate_factors("HOME", "AWAY", toy_priors(), seed=1, sigma=params.sim_sigma("fitted"), pace="off",
                                home="off", rest="off")
    assert sim_scales == [custom_params[f] for f in FACTORS] * 2


def test_legacy_values_are_the_old_model(sim_scales):
    assert params.update_sigma("legacy") == {f: 0.05 for f in FACTORS}
    assert params.sim_sigma("legacy") == {"eFG": 0.05, "TOV": 0.02, "ORB": 0.05, "FTR": 0.02}
    assert update_uses({f: 0.05 for f in FACTORS}, sigma=params.update_sigma("legacy"))
    simulation.simulate_factors("HOME", "AWAY", toy_priors(), seed=1, sigma=params.sim_sigma("legacy"), pace="off",
                                home="off", rest="off")
    assert sim_scales == LEGACY_SIM


# ---------------------------------------------------------------------------
# defaults: fitted for the update and the simulation, everywhere
# ---------------------------------------------------------------------------

def test_default_modes():
    assert params.DEFAULT_UPDATE_SIGMA_MODE == "fitted"
    assert params.DEFAULT_SIM_SIGMA_MODE == "fitted"
    assert params.update_sigma() == FITTED
    assert params.sim_sigma() == FITTED


def test_update_defaults_to_fitted():
    assert update_uses(FITTED)                                           # bayes_update.update_game


def with_paces(priors):
    for team in priors.values():
        team["pace"] = {"mean": 100.0, "std": 2.0}
    return priors


def test_simulation_predict_and_site_default_to_fitted(sim_scales):
    """Default game noise: fitted per factor for both lines, then the pace noise (pace is on by default)."""
    fitted_then_pace = [FITTED[f] for f in FACTORS] * 2 + [params.pace_sigma()]
    simulation.simulate_factors("HOME", "AWAY", with_paces(toy_priors()), seed=1, rest_a="b2b", rest_b="2+")
    assert sim_scales == fitted_then_pace
    sim_scales.clear()
    predict.predict_game("HOME", "AWAY", with_paces(toy_priors()), seed=1, rest_home="b2b", rest_away="2+")
    assert sim_scales == fitted_then_pace
    sim_scales.clear()
    build_site.build_predictions([{"date": "2026-11-20", "home": {"id": "HOME", "abbr": "H", "score": None},
                                   "away": {"id": "AWAY", "abbr": "A", "score": None}, "status": "scheduled"}],
                                 with_paces(toy_priors()), rest_lookup=lambda team, day: {"rest": "2+"})
    assert sim_scales == fitted_then_pace


def test_command_line_defaults():
    u = update.make_parser().parse_args([])
    assert (u.sigma_update, u.sigma_sim) == ("fitted", "fitted")
    assert dynamic_model.make_parser().parse_args(["--out", "x.json", "--before", "2026-10-03"]).sigma_update == "fitted"
    assert build_site.make_parser().parse_args([]).sigma_sim == "fitted"


# ---------------------------------------------------------------------------
# the update and the simulation can differ
# ---------------------------------------------------------------------------

@pytest.fixture
def seen_sigma(env, monkeypatch):
    """Record the sigma the update fit and the prediction simulation receive during a run."""
    seen = {"update": [], "sim": []}
    fake_fit, real_predict = bayes_update._fit, update.predict_game

    def fit(prior, observed, random_seed=None, engine=None, sigma=None):
        seen["update"].append(sigma)
        return fake_fit(prior, observed, random_seed, engine, sigma)

    def pred(home_id, away_id, priors, seed=None, sigma=None, **kwargs):
        seen["sim"].append(sigma)
        return real_predict(home_id, away_id, priors, seed=seed, sigma=sigma, **kwargs)

    monkeypatch.setattr(bayes_update, "_fit", fit)
    monkeypatch.setattr(update, "predict_game", pred)
    return seen


@pytest.mark.parametrize("upd, sim", [("fitted", "legacy"), ("legacy", "fitted"), ("fitted", "fitted"),
                                      ("legacy", "legacy")])
def test_update_and_simulation_sigma_are_set_separately(env, seen_sigma, upd, sim):
    result = update.run_update(sigma_update=upd, sigma_sim=sim)
    assert result["applied"]
    assert all(s == params.update_sigma(upd) for s in seen_sigma["update"])
    assert all(s == params.sim_sigma(sim) for s in seen_sigma["sim"])
    assert len(seen_sigma["update"]) == 2 * len(result["applied"]) and len(seen_sigma["sim"]) == len(result["applied"])
    assert {(e["sigma_update"], e["sigma_sim"]) for e in env.applied().values()} == {(upd, sim)}


def test_command_line_sets_them_separately(env, seen_sigma, tmp_path):
    update.main(["--update", "--sigma-update", "legacy", "--sigma-sim", "fitted"])
    assert seen_sigma["update"][0] == params.LEGACY_UPDATE_SIGMA and seen_sigma["sim"][0] == FITTED
    seen_sigma["update"].clear(), seen_sigma["sim"].clear()
    update.main(["--replay", "--from", str(update.START_PRIORS_PATH), "--out", str(tmp_path / "p.json"),
                 "--log", str(tmp_path / "l.csv"), "--sigma-sim", "legacy"])
    assert seen_sigma["update"][0] == FITTED and seen_sigma["sim"][0] == params.LEGACY_SIM_SIGMA


def test_update_run_records_engine_and_both_sigmas(env):
    update.run_update()
    assert {(e["engine"], e["sigma_update"], e["sigma_sim"]) for e in env.applied().values()} == \
           {("exact", "fitted", "fitted")}


def test_bad_sigma_settings_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        params.update_sigma("nope")
    with pytest.raises(ValueError):
        params.sim_sigma("nope")
    with pytest.raises(ValueError):
        params.check_sigma({"eFG": 0.05, "TOV": 0.05, "ORB": 0.05})          # a factor missing
    with pytest.raises(ValueError):
        params.check_sigma({"eFG": 0.05, "TOV": -0.01, "ORB": 0.05, "FTR": 0.05})
    for argv in (["--update", "--sigma-update", "nope"], ["--update", "--sigma-sim", "nope"], ["--update", "--sigma", "legacy"]):
        with pytest.raises(SystemExit):
            update.main(argv)
    with pytest.raises(SystemExit):
        dynamic_model.main(["--out", str(tmp_path / "x.json"), "--before", "2026-10-03", "--sigma-update", "nope"])
    with pytest.raises(SystemExit):
        build_site.make_parser().parse_args(["--sigma-sim", "nope"])
    monkeypatch.setattr(params, "PARAMS_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="fit_sigma"):
        params.update_sigma()


# ---------------------------------------------------------------------------
# legacy for both reproduces Phase 5 exactly
# ---------------------------------------------------------------------------

def box_cache_complete():
    applied = json.loads((REPO_ROOT / "applied_games.json").read_text(encoding="utf-8"))
    return all((REPO_ROOT / "stats" / "boxscores" / f"{gid}.json").exists() for gid in applied)


@pytest.mark.skipif(not box_cache_complete(), reason="needs the box-score cache in stats/boxscores/")
def test_legacy_replay_reproduces_log_exact(tmp_path):
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_exact.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--sigma-update", "legacy", "--sigma-sim", "legacy",
                 "--pace", "off", "--home", "off", "--rest", "off"])
    assert log.read_text(encoding="utf-8").splitlines() == \
        (REPO_ROOT / "log_exact.csv").read_text(encoding="utf-8").splitlines()
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_2026-27_exact.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not box_cache_complete(), reason="needs the box-score cache in stats/boxscores/")
def test_phase6_replay_reproduces_log_sigma_update(tmp_path):
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_sigma-update.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--sigma-sim", "legacy", "--pace", "off", "--home", "off", "--rest", "off"])
    assert log.read_text(encoding="utf-8").splitlines() == \
        (REPO_ROOT / "log_sigma-update.csv").read_text(encoding="utf-8").splitlines()


@pytest.mark.skipif(not dynamic_model.DEFAULT_CSV.exists(), reason="needs the Kaggle CSV in stats/")
def test_legacy_training_reproduces_priors_pre_exact(tmp_path):
    out = tmp_path / "trained.json"
    dynamic_model.main(["--out", str(out), "--before", "2026-10-03", "--sigma-update", "legacy", "--pace", "off",
                        "--home", "off", "--rest", "off"])
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_pre_2026-27_exact.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------

def test_fit_sigma_recovers_known_noise():
    """Games simulated from the model itself with known sigma: the estimator finds it again (within 10%)."""
    true_sigma = {"eFG": 0.045, "TOV": 0.025, "ORB": 0.065, "FTR": 0.080}
    rng = random.Random(7)
    teams = [f"T{i}" for i in range(12)]
    start = default_priors(teams)[teams[0]]["offense"]
    strength = {t: {side: {f: rng.gauss(start[f]["mean"], 0.02) for f in FACTORS} for side in ("offense", "defense")}
                for t in teams}
    games = []
    for _ in range(1000):
        home, away = rng.sample(teams, 2)
        for team, opp in ((home, away), (away, home)):
            games.append({"team": team, "opponent": opp,
                          **{f: (strength[team]["offense"][f] + strength[opp]["defense"][f]) / 2
                                + rng.gauss(0, true_sigma[f]) for f in FACTORS}})
    fit = fit_sigma.estimate_sigma(games, teams)
    for f in FACTORS:
        assert fit["sigma"][f] == pytest.approx(true_sigma[f], rel=0.10), f
        assert fit["mean_z2"][f] == pytest.approx(1.0, abs=0.05), f


def test_fit_sigma_refuses_2026_27_data():
    with pytest.raises(SystemExit):
        fit_sigma.main(["--before", "2026-10-20", "--out", "unused.json"])
