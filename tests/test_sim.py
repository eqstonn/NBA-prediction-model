"""
Tests for the Phase 7 simulation (code/simulation.py, code/fit_sim.py, pace in
bayes_update.py / update.py / dynamic_model.py / boxscores.py):

  - the factor-correlation estimate recovers a known matrix from simulated games, and
    --sim-corr on draws the game factors with that matrix
  - the corrected points formula, and that it removes the old formula's totals bias
  - pace: the exact update (hand-computed), the season reset, --update / --replay with --pace on
  - both switches off (+ legacy sim sigma, home off) reproduce log_sigma-update.csv exactly
The Phase 7 tests pass home="off" and rest="off" (their model_params.json has neither section); the live
defaults are tested in test_live.py.
  - none of this writes priors.json, applied_games.json or any existing priors/log file

Run from the repo root:  python -m pytest tests

The "reproduces" tests read the real box-score caches (stats/boxscores/,
stats/boxscores_advanced/) and the Kaggle CSV, which are not in git; they are skipped when
those files are missing.
"""
import copy
import hashlib
import json
import math

import numpy as np
import pytest

from fake_nba import EXPECTED, FAKE_SEASON, FakeNBA, ID, REPO_ROOT, update

import bayes_update  # noqa: E402  (code/ is on sys.path via fake_nba)
import boxscores  # noqa: E402
import build_site  # noqa: E402
import dynamic_model  # noqa: E402
import fit_sigma  # noqa: E402
import fit_sim  # noqa: E402
import params  # noqa: E402
import predlog  # noqa: E402
import simulation  # noqa: E402
from schedule import NBA_TEAMS  # noqa: E402
from state_manager import default_priors  # noqa: E402

FACTORS = params.FACTORS
SIGMA = {"eFG": 0.06, "TOV": 0.035, "ORB": 0.07, "FTR": 0.085}
HOME, AWAY = "1610612765", "1610612738"   # DET, BOS

# a known exchangeable 8x8 correlation: within one team's line, and between the two lines
WITHIN = [[1.0, 0.1, -0.2, -0.1],
          [0.1, 1.0, 0.3, 0.1],
          [-0.2, 0.3, 1.0, 0.0],
          [-0.1, 0.1, 0.0, 1.0]]
ACROSS = [[0.25, -0.1, 0.0, 0.0],
          [-0.1, 0.1, 0.0, -0.05],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, -0.05, 0.0, 0.2]]
KNOWN = np.block([[np.array(WITHIN), np.array(ACROSS)], [np.array(ACROSS), np.array(WITHIN)]])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_data_hashes():
    """Every priors / log / params file in the repo root."""
    return {p.name: sha(p) for p in sorted(REPO_ROOT.glob("*.json")) + sorted(REPO_ROOT.glob("*.csv"))}


LEAGUE = {"eFG": 0.54, "TOV": 0.13, "ORB": 0.26, "FTR": 0.21}


def toy_priors(std=0.02, pace=True):
    """DET (a bit better) and BOS around league-average factors, every curve N(mean, std^2), plus pace curves."""
    def team(off_shift, def_shift, p):
        t = {"offense": {f: {"mean": LEAGUE[f] + off_shift, "std": std} for f in FACTORS},
             "defense": {f: {"mean": LEAGUE[f] + def_shift, "std": std} for f in FACTORS}}
        if pace:
            t["pace"] = {"mean": p, "std": 2.0}
        return t
    return {HOME: team(0.01, -0.005, 98.0), AWAY: team(-0.01, 0.005, 102.0)}


@pytest.fixture
def phase7_params(tmp_path, monkeypatch):
    """A model_params.json with known Phase 7 sections, in a temp folder."""
    custom = {"sigma": SIGMA,
              "factor_corr": {"labels": list(params.CORR_LABELS), "matrix": KNOWN.tolist()},
              "points": {"ft_pct": 0.78, "possession_offset": 2.0, "rebound_chances": 0.92},
              "pace": {"sigma": 4.8, "start_mean": 100.0, "start_std": 5.0}}
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    return custom


# ---------------------------------------------------------------------------
# factor correlation
# ---------------------------------------------------------------------------

def test_known_matrix_is_a_valid_correlation():
    np.linalg.cholesky(KNOWN)
    assert params.check_corr(KNOWN.tolist()) == KNOWN.tolist()


def test_corr_estimate_recovers_known_matrix():
    """Games simulated from the model with known team strengths and noise correlation: the estimator finds it."""
    rng = np.random.default_rng(11)
    teams = [f"T{i}" for i in range(12)]
    start = default_priors(teams)[teams[0]]["offense"]
    strength = {t: {side: {f: start[f]["mean"] + rng.normal(0, 0.02) for f in FACTORS}
                    for side in ("offense", "defense")} for t in teams}
    chol = np.linalg.cholesky(KNOWN)
    games = []
    for k in range(6000):
        home, away = (str(t) for t in rng.choice(teams, 2, replace=False))
        z = chol @ rng.standard_normal(8)
        for team, opp, zz in ((home, away, z[:4]), (away, home, z[4:])):
            games.append({"game_id": f"G{k}", "team": team, "opponent": opp,
                          **{f: float((strength[team]["offense"][f] + strength[opp]["defense"][f]) / 2
                                      + SIGMA[f] * zz[i]) for i, f in enumerate(FACTORS)}})
    fit = fit_sim.estimate_corr(games, teams, SIGMA, burn_in={f"G{k}" for k in range(500)})
    assert fit["games"] == 5500 and fit["burn_in_games"] == 500
    assert np.abs(fit["matrix"] - KNOWN).max() < 0.06
    assert np.allclose(fit["matrix"], fit["matrix"].T) and np.allclose(np.diag(fit["matrix"]), 1)


def test_correlated_draws_follow_the_matrix():
    priors = toy_priors(std=0.0)
    a_game, b_game = simulation.correlated_factors(np.random.default_rng(3), priors[HOME], priors[AWAY], SIGMA,
                                                   KNOWN.tolist(), n=40000)
    mean = lambda team, side, f: priors[team][side][f]["mean"]   # noqa: E731
    expected_a = {f: (mean(HOME, "offense", f) + mean(AWAY, "defense", f)) / 2 for f in FACTORS}   # DET's line
    expected_b = {f: (mean(AWAY, "offense", f) + mean(HOME, "defense", f)) / 2 for f in FACTORS}   # BOS's line
    z = np.array([(a_game[f] - expected_a[f]) / SIGMA[f] for f in FACTORS] +
                 [(b_game[f] - expected_b[f]) / SIGMA[f] for f in FACTORS])
    assert np.abs(np.corrcoef(z) - KNOWN).max() < 0.03
    assert np.allclose(z.std(axis=1), 1, atol=0.03)
    assert np.abs(z.mean(axis=1)).max() < 0.03


def test_sim_corr_switch_picks_the_draws(phase7_params, monkeypatch):
    calls = []
    real_ind, real_corr = simulation.independent_factors, simulation.correlated_factors
    monkeypatch.setattr(simulation, "independent_factors", lambda *a, **k: calls.append("off") or real_ind(*a, **k))
    monkeypatch.setattr(simulation, "correlated_factors",
                        lambda rng, a, b, s, corr, *rest: calls.append(corr) or real_corr(rng, a, b, s, corr, *rest))
    simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=1, home="off", rest="off")
    simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=1, sim_corr="off", home="off", rest="off")
    simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=1, sim_corr="on", home="off", rest="off")
    assert calls == ["off", "off", KNOWN.tolist()]


def test_shared_noise_between_the_teams_narrows_margins(phase7_params, tmp_path, monkeypatch):
    """With both teams' eFG noise strongly correlated, simulated margins spread less than with independent draws."""
    independent = simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=5, sigma=SIGMA, home="off", rest="off")
    strong = np.eye(8)
    strong[0, 4] = strong[4, 0] = 0.8
    custom = dict(phase7_params, factor_corr={"labels": list(params.CORR_LABELS), "matrix": strong.tolist()})
    path = tmp_path / "strong.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    joint = simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=5, sigma=SIGMA, sim_corr="on", home="off", rest="off")
    assert joint["margin_sd"] < 0.9 * independent["margin_sd"]


def test_real_model_params_have_valid_phase7_sections():
    model_params = json.loads((REPO_ROOT / "model_params.json").read_text(encoding="utf-8"))
    corr = np.array(model_params["factor_corr"]["matrix"])
    np.linalg.cholesky(corr)
    assert model_params["factor_corr"]["labels"] == list(params.CORR_LABELS)
    assert params.factor_corr() == corr.tolist()
    for section in ("factor_corr", "points", "pace"):
        data = model_params[section]["fit"]["data"]
        assert data["before"] <= dynamic_model.TRAINING_CUTOFF and data["last"] < "2026-10-03"   # no 2026-27 game
    check = model_params["points"]["fit"]["check_on_actual_factors_and_possessions"]
    assert abs(check["game_total_bias"]) < 1 and check["legacy_bias_actual_possessions"] > 15
    assert 2 < model_params["pace"]["sigma"] < 10 and 95 < model_params["pace"]["start_mean"] < 105
    assert 0.95 < model_params["pace"]["fit"]["mean_squared_standardized_residual"] < 1.05


# ---------------------------------------------------------------------------
# points formula
# ---------------------------------------------------------------------------

def test_points_formula_recovers_a_box_score():
    """A team-game built from NBA.com's definitions: the formula gets its points back from the four factors."""
    fga, fta, ftm, tov, orb, fgm, tpm = 88.0, 22.0, 17.6, 14.0, 10.0, 42.0, 12.8
    efg = (fgm + 0.5 * tpm) / fga
    raw = fga + 0.44 * fta + tov - orb
    offset, chances_factor = 2.0, 0.9
    factors = {"eFG": efg, "TOV": tov / raw, "FTR": fta / fga, "ORB": orb / (chances_factor * fga * (1 - efg))}
    constants = {"possession_offset": offset, "rebound_chances": chances_factor, "ft_pct": ftm / fta}
    assert simulation.points(factors, raw - offset, constants) == pytest.approx(2 * fgm + tpm + ftm, abs=1e-9)
    # the old formula overshoots the same game by ~10 points (it ignores free-throw possessions)
    assert simulation.legacy_points(factors, raw - offset) - (2 * fgm + tpm + ftm) > 8


def test_legacy_points_is_the_original_formula():
    g = {"eFG": 0.54, "TOV": 0.13, "ORB": 0.27, "FTR": 0.24}
    assert simulation.legacy_points(g) == 100 * (1 - 0.13) * (1 + (0.27 * (1 - 0.54))) * (2 * 0.54 + 0.24 * 0.77)


def test_pace_on_simulates_realistic_totals_and_uses_team_paces(phase7_params):
    off = simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=2, sigma=SIGMA, pace="off", home="off", rest="off")
    on = simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=2, sigma=SIGMA, pace="on", home="off", rest="off")
    assert on["possessions"] == pytest.approx(100.0, abs=0.2)   # (98 + 102) / 2
    assert off["possessions"] == 100
    # the old formula's +~17 per game at league-average factors is gone
    assert on["team_a_avg_score"] + on["team_b_avg_score"] < off["team_a_avg_score"] + off["team_b_avg_score"] - 12
    fast = toy_priors()
    fast[HOME]["pace"]["mean"] = fast[AWAY]["pace"]["mean"] = 110.0
    quick = simulation.simulate_factors(HOME, AWAY, fast, seed=2, sigma=SIGMA, pace="on", home="off", rest="off")
    assert quick["possessions"] == pytest.approx(110.0, abs=0.2)
    assert quick["team_a_avg_score"] > on["team_a_avg_score"] * 1.07


def test_pace_on_needs_pace_curves(phase7_params):
    with pytest.raises(KeyError, match="pace"):
        simulation.simulate_factors(HOME, AWAY, toy_priors(pace=False), seed=1, pace="on", home="off", rest="off")


# ---------------------------------------------------------------------------
# pace update
# ---------------------------------------------------------------------------

def test_pace_update_hand_computed():
    priors = {"A": {"pace": {"mean": 98.0, "std": 2.0}}, "B": {"pace": {"mean": 102.0, "std": 1.5}},
              "C": {"pace": {"mean": 97.0, "std": 1.0}}}
    bayes_update.update_pace(priors, "A", "B", 106.0, tau=0.3, sigma=4.8)
    s_a2, s_b2 = 2.0 ** 2 + 0.3 ** 2, 1.5 ** 2 + 0.3 ** 2
    v = (s_a2 + s_b2) / 4 + 4.8 ** 2
    surprise = 106.0 - (98.0 + 102.0) / 2
    assert priors["A"]["pace"]["mean"] == pytest.approx(98.0 + (s_a2 / 2) / v * surprise, abs=1e-12)
    assert priors["A"]["pace"]["std"] == pytest.approx(math.sqrt(s_a2 - (s_a2 / 2) ** 2 / v), abs=1e-12)
    assert priors["B"]["pace"]["mean"] == pytest.approx(102.0 + (s_b2 / 2) / v * surprise, abs=1e-12)
    assert priors["B"]["pace"]["std"] == pytest.approx(math.sqrt(s_b2 - (s_b2 / 2) ** 2 / v), abs=1e-12)
    assert priors["C"]["pace"] == {"mean": 97.0, "std": 1.0}
    # a faster game than expected speeds both teams up, and each fit narrows the curves
    assert priors["A"]["pace"]["mean"] > 98.0 and priors["B"]["pace"]["mean"] > 102.0


def test_pace_update_is_the_factor_formula():
    """Pace uses exactly the offense/defense formula (conjugate_pair)."""
    a, b = bayes_update.conjugate_pair(0.50, 0.03, 0.54, 0.02, 0.60, 0.05)
    priors = {"A": {"pace": {"mean": 0.50, "std": 0.03}}, "B": {"pace": {"mean": 0.54, "std": 0.02}}}
    bayes_update.update_pace(priors, "A", "B", 0.60, sigma=0.05)
    assert (priors["A"]["pace"]["mean"], priors["A"]["pace"]["std"]) == a
    assert (priors["B"]["pace"]["mean"], priors["B"]["pace"]["std"]) == b


def test_apply_game_updates_pace_only_when_on(phase7_params):
    game = {"game_id": "X", "home": {"id": HOME, **{f: LEAGUE[f] + 0.02 for f in FACTORS}},
            "away": {"id": AWAY, **{f: LEAGUE[f] - 0.01 for f in FACTORS}}, "possessions": 104.0}
    on = update.apply_game(toy_priors(), game, m=30, pace="on", home="off", rest="off")
    off = update.apply_game(toy_priors(), game, m=30, pace="off", home="off", rest="off")
    assert off[HOME]["pace"] == toy_priors()[HOME]["pace"] and off[AWAY]["pace"] == toy_priors()[AWAY]["pace"]
    for team in (HOME, AWAY):   # the factors don't depend on pace
        assert {k: v for k, v in on[team].items() if k != "pace"} == {k: v for k, v in off[team].items() if k != "pace"}
    expected = toy_priors()
    bayes_update.update_pace(expected, HOME, AWAY, 104.0, tau=2 * 4.8 / 30, sigma=4.8)
    assert on[HOME]["pace"] == expected[HOME]["pace"] and on[AWAY]["pace"] == expected[AWAY]["pace"]
    with pytest.raises(KeyError, match="possessions"):
        update.apply_game(toy_priors(), {k: v for k, v in game.items() if k != "possessions"}, pace="on", home="off", rest="off")


def test_season_reset_sets_pace_std(phase7_params):
    frozen = json.loads((REPO_ROOT / "priors_pre_2026-27_pace.json").read_text(encoding="utf-8"))
    reset = update.reset_priors(frozen, 20)
    for team in frozen:
        if team in NBA_TEAMS:
            assert reset[team]["pace"]["mean"] == frozen[team]["pace"]["mean"]
            assert reset[team]["pace"]["std"] == pytest.approx(2 * 4.8 / math.sqrt(20))
        else:
            assert reset[team] == frozen[team]
    no_pace = update.reset_priors(json.loads(update.FROZEN_PRIORS_PATH.read_text(encoding="utf-8")), 20)
    assert not any("pace" in team for team in no_pace.values())


def with_paces(env):
    priors = json.loads(update.PRIORS_PATH.read_text(encoding="utf-8"))
    for team in NBA_TEAMS:
        priors[team]["pace"] = {"mean": 100.0, "std": 2.0}
    update.write_json(update.PRIORS_PATH, priors)
    return priors


def test_update_run_with_pace(env):
    start = with_paces(env)
    result = update.run_update(pace="on")
    assert result["applied"] == EXPECTED
    assert {(e["sim_corr"], e["pace"]) for e in env.applied().values()} == {("off", "on")}
    expected = copy.deepcopy(start)
    teams = {g[0]: (ID[g[3]], ID[g[2]]) for g in FAKE_SEASON}   # game -> (home, away)
    for gid in EXPECTED:
        bayes_update.update_pace(expected, *teams[gid], FakeNBA.game_possessions(gid), tau=params.pace_tau(30))
    after = json.loads(update.PRIORS_PATH.read_text(encoding="utf-8"))
    assert all(after[t]["pace"] == expected[t]["pace"] for t in NBA_TEAMS)
    played = {t for gid in EXPECTED for t in teams[gid]}
    assert all(after[t]["pace"] == start[t]["pace"] for t in NBA_TEAMS if t not in played)
    assert sorted(p.stem for p in boxscores.ADVANCED_CACHE_DIR.glob("*.json")) == sorted(EXPECTED)


def test_game_without_possessions_is_skipped_and_not_recorded(env):
    with_paces(env)
    env.nba.no_possessions.add("0022600003")
    result = update.run_update(pace="on")
    assert "0022600003" not in env.applied() and "0022600003" not in result["applied"]
    assert any(gid == "0022600003" and "possessions" in why for gid, why in result["skipped"])
    assert "0022600003" not in {r["game_id"] for r in predlog.read_log(update.LOG_PATH)}


def test_pace_off_never_downloads_possessions(env):
    update.run_update(pace="off")
    assert env.nba.advanced_requests == []
    assert {(e["sim_corr"], e["pace"]) for e in env.applied().values()} == {("off", "off")}


def test_pace_on_needs_team_paces(env, tmp_path):
    no_paces = json.loads(update.FROZEN_PRIORS_PATH.read_text(encoding="utf-8"))
    update.write_json(update.PRIORS_PATH, update.reset_priors(no_paces))
    with pytest.raises(ValueError, match="pace"):
        update.run_update(pace="on")
    with pytest.raises(SystemExit):
        update.main(["--update", "--pace", "on"])
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(update.FROZEN_PRIORS_PATH), "--out", str(tmp_path / "p.json"),
                     "--log", str(tmp_path / "l.csv"), "--pace", "on"])
    assert not (tmp_path / "p.json").exists() and not (tmp_path / "l.csv").exists()


def test_replay_with_pace_and_corr(env, tmp_path):
    update.run_update()
    start = json.loads(update.FROZEN_PRIORS_PATH.read_text(encoding="utf-8"))
    for team in start:
        start[team]["pace"] = {"mean": 100.0, "std": 0.5}
    src = tmp_path / "pre_pace.json"
    src.write_text(json.dumps(start), encoding="utf-8")
    out, log = tmp_path / "out.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(src), "--out", str(out), "--log", str(log),
                 "--sim-corr", "on", "--pace", "on", "--sigma-sim", "fitted"])
    replayed = json.loads(out.read_text(encoding="utf-8"))
    assert all(replayed[t]["pace"]["mean"] != 100.0 for gid in EXPECTED for t in (ID[g[3]] for g in FAKE_SEASON if g[0] == gid))
    assert [r["game_id"] for r in predlog.read_log(log)] == EXPECTED


# ---------------------------------------------------------------------------
# defaults, settings
# ---------------------------------------------------------------------------

def test_switch_defaults_are_sim_corr_off_and_pace_on_everywhere(phase7_params):
    assert (params.DEFAULT_SIM_CORR_MODE, params.DEFAULT_PACE_MODE) == ("off", "on")
    u = update.make_parser().parse_args([])
    assert (u.sim_corr, u.pace) == ("off", "on")
    assert dynamic_model.make_parser().parse_args(["--out", "x.json", "--before", "2026-10-03"]).pace == "on"
    b = build_site.make_parser().parse_args([])
    assert (b.sim_corr, b.pace) == ("off", "on")
    priors = toy_priors()
    assert simulation.simulate_factors(HOME, AWAY, priors, seed=9, home="off", rest="off") == \
        simulation.simulate_factors(HOME, AWAY, priors, seed=9, sim_corr="off", pace="on", home="off", rest="off")


def test_bad_switch_settings_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        params.is_on("yes")
    with pytest.raises(ValueError):
        simulation.simulate_factors(HOME, AWAY, toy_priors(), seed=1, sim_corr="maybe")
    with pytest.raises(ValueError):
        params.check_corr(np.eye(7).tolist())
    asym = np.eye(8)
    asym[0, 1] = 0.3
    with pytest.raises(ValueError):
        params.check_corr(asym.tolist())
    for argv in (["--update", "--sim-corr", "yes"], ["--update", "--pace", "1"]):
        with pytest.raises(SystemExit):
            update.main(argv)
    with pytest.raises(SystemExit):
        build_site.make_parser().parse_args(["--pace", "maybe"])
    monkeypatch.setattr(params, "PARAMS_PATH", tmp_path / "only_sigma.json")
    (tmp_path / "only_sigma.json").write_text(json.dumps({"sigma": SIGMA}), encoding="utf-8")
    with pytest.raises(KeyError, match="fit_sim"):
        params.sim_corr("on")


def test_fit_sim_refuses_2026_27_data_and_existing_sections(tmp_path):
    with pytest.raises(SystemExit):
        fit_sim.main(["--before", "2026-10-20", "--params", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit):   # model_params.json already has the sections
        fit_sim.main([])


def test_fit_sigma_keeps_the_phase7_sections(tmp_path, monkeypatch):
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps({"sigma": SIGMA, "factor_corr": {"matrix": "kept"}, "pace": {"sigma": 4.8}}),
                    encoding="utf-8")
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("x\n", encoding="utf-8")
    import pandas as pd
    rows = pd.DataFrame([{"teamId": 1, "opponentTeamId": 2, "effectiveFieldGoalPercentage": 0.5,
                          "teamTurnoverPercentage": 0.1, "offensiveReboundPercentage": 0.3, "freeThrowAttemptRate": 0.2}])
    monkeypatch.setattr(dynamic_model, "load_training_rows",
                        lambda *a, **k: {"rows": rows, "team_ids": ["1", "2"], "games": 1, "first": "2024-01-01",
                                         "last": "2024-01-01"})
    monkeypatch.setattr(fit_sigma, "estimate_sigma",
                        lambda g, t: {"sigma": dict(SIGMA), "log_likelihood": {f: 1.0 for f in FACTORS},
                                      "mean_z2": {f: 1.0 for f in FACTORS}})
    fit_sigma.main(["--csv", str(fake_csv), "--out", str(path), "--overwrite"])
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["factor_corr"] == {"matrix": "kept"} and out["pace"] == {"sigma": 4.8} and out["sigma"] == SIGMA


# ---------------------------------------------------------------------------
# reproducibility on the real season, and protected files
# ---------------------------------------------------------------------------

def caches_complete(advanced=False):
    applied = json.loads((REPO_ROOT / "applied_games.json").read_text(encoding="utf-8"))
    folder = "boxscores_advanced" if advanced else "boxscores"
    return all((REPO_ROOT / "stats" / folder / f"{gid}.json").exists() for gid in applied)


@pytest.mark.skipif(not caches_complete(), reason="needs the box-score cache in stats/boxscores/")
def test_switches_off_reproduce_log_sigma_update(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_sigma-update.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--sim-corr", "off", "--pace", "off",
                 "--sigma-sim", "legacy", "--home", "off", "--rest", "off"])
    assert sha(log) == sha(REPO_ROOT / "log_sigma-update.csv")
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_2026-27_sigma-update.json").read_text(encoding="utf-8"))
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (caches_complete() and caches_complete(advanced=True)),
                    reason="needs the box-score caches in stats/boxscores/ and stats/boxscores_advanced/")
def test_combined_replay_reproduces_log_corr_pace(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_pace.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--sim-corr", "on", "--pace", "on",
                 "--sigma-sim", "fitted", "--home", "off", "--rest", "off"])
    assert sha(log) == sha(REPO_ROOT / "log_corr-pace.csv")
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_2026-27_corr-pace.json").read_text(encoding="utf-8"))
    assert repo_data_hashes() == before


@pytest.mark.skipif(not dynamic_model.DEFAULT_CSV.exists(), reason="needs the Kaggle CSV in stats/")
def test_training_with_pace_keeps_the_factors(tmp_path):
    before = repo_data_hashes()
    out = tmp_path / "trained.json"
    dynamic_model.main(["--out", str(out), "--before", "2026-10-03", "--pace", "on", "--home", "off", "--rest", "off"])
    trained = json.loads(out.read_text(encoding="utf-8"))
    assert trained == json.loads((REPO_ROOT / "priors_pre_2026-27_pace.json").read_text(encoding="utf-8"))
    factors_only = {t: {k: v for k, v in c.items() if k != "pace"} for t, c in trained.items()}
    assert factors_only == json.loads((REPO_ROOT / "priors_pre_2026-27_sigma-update.json").read_text(encoding="utf-8"))
    assert all("pace" in c for c in trained.values())
    assert repo_data_hashes() == before


@pytest.mark.skipif(not dynamic_model.DEFAULT_CSV.exists(), reason="needs the Kaggle CSV in stats/")
def test_fit_sim_reproduces_model_params_and_keeps_sigma(tmp_path):
    before = repo_data_hashes()
    path = tmp_path / "model_params.json"
    path.write_bytes((REPO_ROOT / "model_params.json").read_bytes())
    fit_sim.main(["--params", str(path), "--overwrite"])
    new, old = (json.loads(p.read_text(encoding="utf-8")) for p in (path, REPO_ROOT / "model_params.json"))

    def strip_dates(section):
        return {k: ({kk: vv for kk, vv in v.items() if kk != "fitted"} if k == "fit" else v) for k, v in section.items()}

    assert new["sigma"] == old["sigma"] and new["sigma_fit"] == old["sigma_fit"]
    for section in ("factor_corr", "points", "pace"):
        assert strip_dates(new[section]) == strip_dates(old[section]), section
    assert repo_data_hashes() == before


@pytest.mark.parametrize("target", ["priors.json", "applied_games.json", "model_params.json",
                                    "priors_pre_2026-27.json", "log_sigma-update.csv", "priors_2026-27_corr-pace.json"])
def test_phase7_replay_never_overwrites_existing_files(env, tmp_path, target):
    update.run_update()
    before = repo_data_hashes()
    real = REPO_ROOT / target
    other = tmp_path / ("x.csv" if real.suffix == ".json" else "x.json")
    out, log = (real, other) if real.suffix == ".json" else (other, real)
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_pace.json"), "--out", str(out),
                     "--log", str(log), "--sim-corr", "on", "--pace", "on", "--sigma-sim", "fitted"])
    assert repo_data_hashes() == before
