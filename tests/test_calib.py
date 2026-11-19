"""
Tests for the win-probability calibration (Phase 10b): params.calibrate_win_prob, the --calib switch in the
simulation / prediction / update / backtest, and code/fit_calib.py's fit, evaluation and diagnosis helpers.
Nothing here touches the real priors.json or applied_games.json (the env fixture uses a temp folder).
"""
import csv
import json
import math
from statistics import NormalDist

import numpy as np
import pytest

import backtest
import fit_calib
import params
import predict
import predlog
import simulation
import update
from fake_nba import EXPECTED, ID, REPO_ROOT

PHI = NormalDist()
PRE = REPO_ROOT / "priors_pre_2026-27_pace-home-rest.json"
BOS, NYK = ID["BOS"], ID["NYK"]


def season_priors():
    return update.reset_priors(json.loads(PRE.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------

def test_off_is_the_identity_and_on_is_phi_a_phi_inverse():
    assert params.calibrate_win_prob(0.63, "off") == 0.63
    assert params.calibrate_win_prob(0.63, "on", temperature=1.0) == pytest.approx(0.63, abs=1e-12)
    p = PHI.cdf(0.5)                                                  # 0.6915
    assert params.calibrate_win_prob(p, "on", temperature=2.0) == pytest.approx(PHI.cdf(1.0), abs=1e-12)
    assert params.calibrate_win_prob(0.5, "on", temperature=1.36) == pytest.approx(0.5, abs=1e-12)


def test_transform_is_symmetric_monotone_and_never_changes_the_pick():
    ps = np.linspace(0.01, 0.99, 99)
    out = [params.calibrate_win_prob(p, "on", temperature=1.36) for p in ps]
    assert all(b > a for a, b in zip(out, out[1:]))                                 # monotone
    for p, q in zip(ps, out):
        assert params.calibrate_win_prob(1 - p, "on", temperature=1.36) == pytest.approx(1 - q, abs=1e-12)
        assert (q > 0.5) == (p > 0.5)                                               # same pick
        if p != 0.5:
            assert abs(q - 0.5) > abs(p - 0.5)                                      # a > 1 sharpens


def test_extreme_probabilities_are_clipped_not_infinite():
    top = params.calibrate_win_prob(1.0, "on", temperature=1.36)
    assert 0.99 < top < 1 and params.calibrate_win_prob(0.0, "on", temperature=1.36) == pytest.approx(1 - top)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        params.calibrate_win_prob(0.6, "maybe")
    with pytest.raises(ValueError):
        params.calibrate_win_prob(0.6, "on", temperature=0.0)
    with pytest.raises(ValueError):
        params.check_temperature(True)


def test_temperature_comes_from_model_params(tmp_path, monkeypatch):
    a = params.calib_temperature()
    assert a == json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))["calib"]["temperature"]
    assert params.calibrate_win_prob(0.6, "on") == pytest.approx(PHI.cdf(a * PHI.inv_cdf(0.6)))
    stripped = json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))
    del stripped["calib"]
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps(stripped), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    with pytest.raises(KeyError, match="fit_calib.py"):
        params.calibrate_win_prob(0.6, "on")
    assert params.calibrate_win_prob(0.6, "off") == 0.6        # off never needs the section


def test_fitted_section_records_its_data():
    c = json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))["calib"]
    assert "2024-25 only" in c["fit"]["data"]
    assert c["fit"]["simulation"] == {"sigma_sim": "fitted", "sim_corr": "off", "pace": "on", "home": "on", "rest": "on"}
    assert c["fit"]["loglik_after"] > c["fit"]["loglik_before"]
    assert 1 < c["temperature"] < 2 and c["temperature_se"] > 0


# ---------------------------------------------------------------------------
# the switch in the simulation and the prediction
# ---------------------------------------------------------------------------

def test_simulation_with_calib_changes_only_the_win_pct():
    priors = season_priors()
    kw = {"seed": 7, "rest_a": "2+", "rest_b": "b2b"}
    off = simulation.simulate_factors(BOS, NYK, priors, **kw)
    on = simulation.simulate_factors(BOS, NYK, priors, calib="on", **kw)
    for key in ("team_a_avg_score", "team_b_avg_score", "margin_sd", "possessions", "team_a_win_pct_raw"):
        assert on[key] == off[key], key
    assert off["team_a_win_pct"] == off["team_a_win_pct_raw"]
    assert on["team_a_win_pct"] == pytest.approx(100 * params.calibrate_win_prob(off["team_a_win_pct"] / 100, "on"))
    assert on["team_a_win_pct"] + on["team_b_win_pct"] == pytest.approx(100)
    assert (on["team_a_win_pct"] > 50) == (off["team_a_win_pct"] > 50)


def test_predict_game_passes_calib_through():
    priors = season_priors()
    kw = {"seed": predlog.PREDICT_SEED, "rest_home": "2+", "rest_away": "2+"}
    off = predict.predict_game(BOS, NYK, priors, **kw)
    on = predict.predict_game(BOS, NYK, priors, calib="on", **kw)
    assert (on["home_score"], on["away_score"]) == (off["home_score"], off["away_score"])
    assert on["home_win_pct"] == pytest.approx(100 * params.calibrate_win_prob(off["home_win_pct"] / 100, "on"))


def test_pace_sigma_override_only_for_diagnosis():
    priors = season_priors()
    kw = {"seed": 3, "rest_a": "2+", "rest_b": "2+"}
    default = simulation.simulate_factors(BOS, NYK, priors, **kw)
    assert simulation.simulate_factors(BOS, NYK, priors, pace_sigma=params.pace_sigma(), **kw) == default
    quiet = simulation.simulate_factors(BOS, NYK, priors, pace_sigma=1e-12, **kw)
    assert quiet["margin_sd"] != default["margin_sd"]


# ---------------------------------------------------------------------------
# the switch in --update, --replay and the backtest
# ---------------------------------------------------------------------------

def read_log(path):
    with open(path, encoding="utf-8") as f:
        return {r["game_id"]: r for r in csv.DictReader(f)}


def test_update_with_calib_logs_calibrated_win_pct_and_the_same_priors(env, tmp_path):
    update.run_update()
    off_priors, off_log = env.priors_text(), read_log(update.LOG_PATH)
    assert all(e["calib"] == "off" for e in env.applied().values())
    env.use(tmp_path / "calibrated")
    update.run_update(calib="on")
    on_log = read_log(update.LOG_PATH)
    assert env.priors_text() == off_priors                     # the update never sees the calibration
    assert all(e["calib"] == "on" for e in env.applied().values()) and list(env.applied()) == EXPECTED
    for gid in EXPECTED:
        off, on = off_log[gid], on_log[gid]
        assert (on["predicted_winner"], on["proj_home"], on["proj_away"]) == \
               (off["predicted_winner"], off["proj_home"], off["proj_away"])
        expected = 100 * params.calibrate_win_prob(float(off["win_pct"]) / 100, "on")
        assert float(on["win_pct"]) == pytest.approx(expected, abs=0.006)   # the log keeps 2 decimals
        assert float(on["win_pct"]) >= float(off["win_pct"])                # a > 1: never less sure


def test_update_cli_takes_calib(env):
    update.main(["--update", "--calib", "on"])
    assert {e["calib"] for e in env.applied().values()} == {"on"}
    with pytest.raises(SystemExit):
        update.main(["--update", "--calib", "maybe"])


def test_backtest_season_with_calib(monkeypatch):
    from test_backtest import GAMES, start_priors
    start = start_priors()
    off, on = backtest.run_season(start, GAMES), backtest.run_season(start, GAMES, calib="on")
    for a, b in zip(off, on):
        assert {k: v for k, v in a.items() if k != "p_model"} == {k: v for k, v in b.items() if k != "p_model"}
        assert b["p_model"] == pytest.approx(params.calibrate_win_prob(a["p_model"], "on"))


# ---------------------------------------------------------------------------
# fit_calib.py
# ---------------------------------------------------------------------------

def synthetic(a_true, n=6000, seed=11):
    """Raw probabilities whose outcomes follow Phi(a_true * Phi^-1(p))."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.45, n)
    p_raw = np.array([PHI.cdf(x) for x in z])
    y = rng.random(n) < np.array([PHI.cdf(a_true * x) for x in z])
    return p_raw, y


@pytest.mark.parametrize("a_true", [1.0, 1.36])
def test_fit_recovers_a_known_temperature(a_true):
    p, y = synthetic(a_true)
    fit = fit_calib.fit_temperature(p, y)
    assert abs(fit["temperature"] - a_true) < 3 * fit["se"]
    assert 0 < fit["se"] < 0.2
    assert fit["loglik_after"] >= fit["loglik_before"] - 1e-9
    with pytest.raises(ValueError):
        fit_calib.fit_temperature([0.6], [True])


def test_vector_and_scalar_transforms_agree():
    p = np.array([0.0, 0.2, 0.5, 0.71, 1.0])
    assert fit_calib.calibrated(p, 1.36) == pytest.approx(
        [params.calibrate_win_prob(x, "on", temperature=1.36) for x in p], abs=1e-12)


def test_compare_keeps_accuracy_and_pairs_the_rest():
    p_old = np.array([0.6, 0.3, 0.55, 0.8])
    p_new = fit_calib.calibrated(p_old, 1.36)
    won, proj, margin = [True, True, False, True], [3, -4, 1, 8], [5, 2, -6, 10]
    c = fit_calib.compare(p_old, p_new, won, proj, margin)
    assert c["accuracy"]["diff"] == 0 and c["margin_mae"]["diff"] == 0
    assert c["brier"]["diff"] == pytest.approx(np.mean((p_new - np.array(won)) ** 2 - (p_old - np.array(won)) ** 2))
    assert "new - old" in fit_calib.format_compare("x", c)


def write_log(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(predlog.COLUMNS)
        w.writerows(rows)
    return path


def test_compare_logs_needs_the_same_projections(tmp_path):
    base = ["2026-11-01", "0022600100", "BOS", "NYK", "BOS", "60.00", "112.0", "108.0", "110", "100"]
    old = write_log(tmp_path / "old.csv", [base])
    new = write_log(tmp_path / "new.csv", [base[:5] + ["64.10"] + base[6:]])
    c = fit_calib.compare_logs(old, new)
    assert c["n"] == 1 and c["new"]["brier"] < c["old"]["brier"]
    other = write_log(tmp_path / "other.csv", [base[:6] + ["115.0"] + base[7:]])
    with pytest.raises(ValueError, match="projected scores differ"):
        fit_calib.compare_logs(old, other)


def test_summarize_diagnosis_shares_and_ratios():
    rows = [{"margin": m, "proj": pr, "p": 0.6, "full": 16.0, "no_rating": 15.5, "no_game": 4.0, "no_pace": 16.0}
            for m, pr in ((10, 5), (-8, -4), (6, 3), (-2, -1))]
    sigma = params.sim_sigma()
    lines = [{f: (0.0 if f != "eFG" else 0.06, 0.0001) for f in params.FACTORS}] * 4
    d = fit_calib.summarize_diagnosis(rows, lines)
    assert d["share"]["factor game noise"] == pytest.approx(1 - 16 / 256)
    assert d["share"]["rating uncertainty"] == pytest.approx(1 - 15.5 ** 2 / 256)
    assert d["share"]["pace noise"] == pytest.approx(0)
    assert d["slope"] == pytest.approx(2.0)
    assert d["factors"]["eFG"]["ratio"] == pytest.approx(0.0036 / (0.0001 + sigma["eFG"] ** 2))
    assert "factor level" in fit_calib.format_diagnosis(d)


def test_noise_split_on_real_priors():
    priors = season_priors()
    game = {"home": {"id": BOS}, "away": {"id": NYK}, "neutral": False, "rest": {"home": "2+", "away": "2+"}}
    s = fit_calib.noise_split(priors, game)
    assert s["no_game"] < s["no_rating"] < s["full"]           # game noise dominates, rating uncertainty adds a bit
    assert s["no_pace"] == pytest.approx(s["full"], rel=0.02)
    assert priors == season_priors()                           # the copy without rating uncertainty is a copy


def test_fit_cli_refuses_to_replace_without_overwrite(tmp_path, capsys):
    path = tmp_path / "model_params.json"
    path.write_text(params.PARAMS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit):
        fit_calib.main(["--params", str(path)])
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        fit_calib.main(["--old-log", "a.csv", "--new-log", "b.csv"])      # logs only with --evaluate
