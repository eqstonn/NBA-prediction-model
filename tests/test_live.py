"""
Tests for the live defaults (the pace-home switch, then the rest switch): exact engine,
fitted sigma for the update AND the simulation, pace on, sim_corr off, home on, rest on,
everywhere:

  - every command line, every function signature and params.py agree on them
  - the default simulation / prediction is the explicit live setting, draw for draw, and it
    needs both teams' rest
  - a season run with no flags (start-season, --update, --trial) uses all of them and
    records them in applied_games.json; the season reset reads the live preseason file
  - train_calibration.py's variants spell out every switch ("live" = these defaults)
  - with no flags, training reproduces priors_pre_2026-27_pace-home-rest.json and the replay
    reproduces priors_2026-27_pace-home-rest.json + log_pace-home-rest.csv; priors.json and
    applied_games.json are that replay (the switch)
  - neither the live files nor the previous live model's (pace-home) are ever overwritten

Run from the repo root:  python -m pytest tests

The "reproduces" tests read the box-score caches, the cached NBA.com schedules and the
Kaggle CSV, which are not in git; they are skipped when missing.
"""
import copy
import csv
import hashlib
import inspect
import json

import pytest

from fake_nba import EXPECTED, FAKE_SEASON, FakeNBA, ID, REPO_ROOT, update

import bayes_update  # noqa: E402  (code/ is on sys.path via fake_nba)
import build_site  # noqa: E402
import dynamic_model  # noqa: E402
import params  # noqa: E402
import predict  # noqa: E402
import simulation  # noqa: E402
import train_calibration  # noqa: E402
from schedule import NBA_TEAMS  # noqa: E402

LIVE = {"engine": "exact", "sigma_update": "fitted", "sigma_sim": "fitted", "sim_corr": "off", "pace": "on",
        "home": "on", "rest": "on"}
OLD_LIVE = {"sigma_sim": "legacy", "sim_corr": "off", "pace": "off", "home": "off"}   # Phase 6
CALIB = "off"   # Phase 10b win-probability calibration: not live (the 210 recorded games predate it, so not in LIVE)
GAMES = json.loads((REPO_ROOT / "tests" / "fixtures" / "games_2026-27_first5.json").read_text(encoding="utf-8"))
PRE = REPO_ROOT / "priors_pre_2026-27_pace-home-rest.json"
REPLAY = REPO_ROOT / "priors_2026-27_pace-home-rest.json"
LOG = REPO_ROOT / "log_pace-home-rest.csv"
PREVIOUS = {"pre": REPO_ROOT / "priors_pre_2026-27_pace-home.json",          # the previous live model (rest off)
            "replay": REPO_ROOT / "priors_2026-27_pace-home.json", "log": REPO_ROOT / "log_pace-home.csv"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_data_hashes():
    return {p.name: sha(p) for p in sorted(REPO_ROOT.glob("*.json")) + sorted(REPO_ROOT.glob("*.csv"))}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def live_priors():
    """Start-of-season priors of the live pipeline (team paces included)."""
    return update.reset_priors(load(PRE))


# ---------------------------------------------------------------------------
# the defaults, everywhere
# ---------------------------------------------------------------------------

def test_params_defaults_are_the_live_model():
    assert bayes_update.DEFAULT_ENGINE == LIVE["engine"]
    assert params.DEFAULT_UPDATE_SIGMA_MODE == LIVE["sigma_update"]
    assert params.DEFAULT_SIM_SIGMA_MODE == LIVE["sigma_sim"]
    assert params.DEFAULT_SIM_CORR_MODE == LIVE["sim_corr"]
    assert params.DEFAULT_PACE_MODE == LIVE["pace"]
    assert params.DEFAULT_HOME_MODE == LIVE["home"]
    assert params.DEFAULT_REST_MODE == LIVE["rest"]
    assert params.DEFAULT_CALIB_MODE == CALIB
    assert params.calibrate_win_prob(0.6) == 0.6
    assert params.update_sigma() == params.sim_sigma() == params.fitted_sigma()
    assert params.sim_corr() is None
    assert params.home_offsets() == (params.home_boost()["offense"], params.home_boost()["defense"])
    effects = params.rest_effects()
    assert params.rest_offset(LIVE["rest"], "b2b", "2+") == effects["offense"]["b2b"]
    assert (update.DEFAULT_N, update.DEFAULT_M) == (20, 30)


def test_command_line_defaults_are_the_live_model():
    u = update.make_parser().parse_args([])
    assert {k: getattr(u, k) for k in LIVE} == LIVE
    assert (u.n, u.m, u.calib) == (20, 30, CALIB)
    d = dynamic_model.make_parser().parse_args(["--out", "x.json", "--before", "2026-10-03"])
    assert (d.engine, d.sigma_update, d.pace, d.home, d.rest) == ("exact", "fitted", "on", "on", "on")
    b = build_site.make_parser().parse_args([])
    assert (b.sigma_sim, b.sim_corr, b.pace, b.home, b.rest, b.calib) == ("fitted", "off", "on", "on", "on", CALIB)


# every function that takes a setting, and the keyword it takes it by
SIGNATURES = [
    (update.apply_game, {"engine": "exact", "pace": "on", "home": "on", "rest": "on"}),
    (update.record_game, {"engine": "exact", "sigma_update": "fitted", "sigma_sim": "fitted", "sim_corr": "off",
                          "pace": "on", "home_mode": "on", "rest_mode": "on", "calib": CALIB}),
    (update.log_prediction, {"sim_corr": "off", "pace": "on", "home": "on", "rest": "on", "calib": CALIB}),
    (update.run_update, {"engine": "exact", "sigma_update": "fitted", "sigma_sim": "fitted", "sim_corr": "off",
                         "pace": "on", "home": "on", "rest": "on", "m": 30, "calib": CALIB}),
    (update.replay, {"engine": "exact", "sigma_update": "fitted", "sigma_sim": "fitted", "sim_corr": "off",
                     "pace": "on", "home": "on", "rest": "on", "n": 20, "m": 30, "calib": CALIB}),
    (update.trial, {"engine": "exact", "sigma_update": "fitted", "pace": "on", "home": "on", "rest": "on"}),
    (update.start_season, {"pace": "on", "n": 20}),
    (dynamic_model.run_training, {"engine": "exact", "sigma_update": "fitted", "pace": "on", "home": "on",
                                  "rest": "on"}),
    (predict.predict_game, {"sim_corr": "off", "pace": "on", "home": "on", "rest": "on", "calib": CALIB}),
    (simulation.simulate_factors, {"sim_corr": "off", "pace": "on", "home": "on", "rest": "on", "calib": CALIB}),
    (build_site.build_predictions, {"sim_corr": "off", "pace": "on", "home": "on", "rest": "on", "calib": CALIB}),
    (train_calibration.run_pass, {"rest": "on", "calib": CALIB}),
    (params.update_sigma, {"mode": "fitted"}),
    (params.sim_sigma, {"mode": "fitted"}),
]


@pytest.mark.parametrize("fn, expected", SIGNATURES, ids=[f.__module__ + "." + f.__name__ for f, _ in SIGNATURES])
def test_function_defaults_are_the_live_model(fn, expected):
    sig = inspect.signature(fn).parameters
    assert {k: sig[k].default for k in expected} == expected


def test_default_simulation_is_the_explicit_live_setting():
    priors = live_priors()
    h, a = GAMES[0]["home"]["id"], GAMES[0]["away"]["id"]
    default = simulation.simulate_factors(h, a, priors, seed=7, rest_a="2+", rest_b="b2b")
    explicit = simulation.simulate_factors(h, a, priors, seed=7, sigma=params.sim_sigma("fitted"), sim_corr="off",
                                           pace="on", home="on", neutral=False, rest="on", rest_a="2+", rest_b="b2b")
    assert default == explicit
    no_rest = simulation.simulate_factors(h, a, priors, seed=7, rest="off")      # the previous live simulation
    assert default != no_rest
    assert default["team_a_avg_score"] - default["team_b_avg_score"] > \
        no_rest["team_a_avg_score"] - no_rest["team_b_avg_score"] + 2           # the away team is on a b2b
    old = simulation.simulate_factors(h, a, priors, seed=7, sigma=params.sim_sigma("legacy"), sim_corr="off",
                                      pace="off", home="off", rest="off")
    assert default != old
    assert default["possessions"] != simulation.LEGACY_PACE                # simulated from the teams' paces
    assert predict.predict_game(h, a, priors, seed=7, rest_home="2+", rest_away="b2b") == {
        "home_win_pct": default["team_a_win_pct"], "away_win_pct": default["team_b_win_pct"],
        "home_score": default["team_a_avg_score"], "away_score": default["team_b_avg_score"]}


def test_default_simulation_needs_both_teams_rest_and_team_paces():
    h, a = GAMES[0]["home"]["id"], GAMES[0]["away"]["id"]
    with pytest.raises(ValueError, match="rest"):
        simulation.simulate_factors(h, a, live_priors(), seed=1)
    with pytest.raises(ValueError, match="rest"):
        predict.predict_game(h, a, live_priors(), seed=1, rest_home="2+")
    no_paces = update.reset_priors(load(update.FROZEN_PRIORS_PATH))
    with pytest.raises(KeyError, match="pace"):
        simulation.simulate_factors(h, a, no_paces, seed=1, rest_a="2+", rest_b="2+")


def test_default_apply_game_needs_possessions_the_neutral_flag_and_rest():
    game = GAMES[0]
    with pytest.raises(KeyError, match="possessions"):
        update.apply_game(live_priors(), game)
    with pytest.raises(KeyError, match="neutral"):
        update.apply_game(live_priors(), {**game, "possessions": 100.0})
    with pytest.raises(KeyError, match="rest"):
        update.apply_game(live_priors(), {**game, "possessions": 100.0, "neutral": False})
    full = {**game, "possessions": 100.0, "neutral": False, "rest": {"home": "b2b", "away": "1"}}
    priors = update.apply_game(live_priors(), full)
    expected = update.apply_game(live_priors(), full, engine="exact", sigma=params.update_sigma("fitted"), pace="on",
                                 home="on", rest="on")
    assert priors == expected
    assert priors != update.apply_game(live_priors(), full, rest="off")
    start = live_priors()
    assert priors[game["home"]["id"]]["pace"] != start[game["home"]["id"]]["pace"]


# ---------------------------------------------------------------------------
# a season with no flags (fake NBA.com)
# ---------------------------------------------------------------------------

def test_default_season_run_uses_and_records_the_live_settings(env, monkeypatch):
    """env starts the season with update.start_season() (the live preseason file); then --update with no flags."""
    env.nba.neutral.add("0022600015")
    start = json.loads(env.priors_text())
    assert all("pace" in start[t] for t in NBA_TEAMS)
    seen = {}
    real = update.apply_game
    monkeypatch.setattr(update, "apply_game", lambda priors, game, *a, **kw: seen.setdefault(game["game_id"], game)
                        and real(priors, game, *a, **kw))
    update.main(["--update"])
    applied = env.applied()
    assert list(applied) == EXPECTED
    assert {tuple(e[k] for k in LIVE) for e in applied.values()} == {tuple(LIVE.values())}
    assert {gid for gid, e in applied.items() if e["neutral"]} == {"0022600015"}
    assert all(set(seen[g]["rest"]) == {"home", "away"} for g in EXPECTED)   # every game carries both teams' rest
    assert sorted(env.nba.advanced_requests) == sorted(EXPECTED)      # possessions for every game
    after = json.loads(env.priors_text())
    teams = {g[0]: (ID[g[3]], ID[g[2]]) for g in FAKE_SEASON}
    expected_pace = copy.deepcopy(start)
    for gid in EXPECTED:
        bayes_update.update_pace(expected_pace, *teams[gid], FakeNBA.game_possessions(gid), tau=params.pace_tau(30))
    assert all(after[t]["pace"] == expected_pace[t]["pace"] for t in NBA_TEAMS)


def test_default_run_equals_the_explicit_live_run(env, tmp_path):
    update.run_update()
    default = env.priors_text()
    env.use(tmp_path / "explicit")
    update.run_update(engine="exact", sigma_update="fitted", sigma_sim="fitted", sim_corr="off", pace="on", home="on",
                      rest="on")
    assert env.priors_text() == default


def test_update_counts_a_game_the_schedule_leaves_out(env, monkeypatch):
    """A finished game NBA.com's schedule doesn't list still gets its rest, from the combined game list."""
    env.nba.sched_missing.add("0022600015")
    seen = {}
    real = update.apply_game
    monkeypatch.setattr(update, "apply_game", lambda priors, game, *a, **kw: seen.setdefault(game["game_id"], game)
                        and real(priors, game, *a, **kw))
    result = update.run_update()
    assert result["applied"] == EXPECTED
    assert seen["0022600015"]["rest"] == {"home": "2+", "away": "2+"}   # LAL and GSW's first games


def test_trial_defaults_to_pace_home_and_rest_and_writes_nothing(env, monkeypatch):
    env.nba.neutral.add("0022600002")
    seen = []
    real = update.apply_game
    monkeypatch.setattr(update, "apply_game", lambda priors, game, *a, **kw: seen.append((game, kw)) or
                        real(priors, game, *a, **kw))
    before = (update.PRIORS_PATH.read_bytes(), update.APPLIED_GAMES_PATH.read_bytes())
    update.main(["--trial", "3"])
    assert [g["game_id"] for g, _ in seen] == EXPECTED[:3]
    assert all((kw["pace"], kw["home"], kw["rest"]) == ("on", "on", "on") for _, kw in seen)
    assert [g["neutral"] for g, _ in seen] == [False, True, False]
    assert all(g["possessions"] == FakeNBA.game_possessions(g["game_id"]) for g, _ in seen)
    assert all(set(g["rest"]) == {"home", "away"} for g, _ in seen)
    assert (update.PRIORS_PATH.read_bytes(), update.APPLIED_GAMES_PATH.read_bytes()) == before


def test_start_season_reads_the_live_preseason_file(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "PRIORS_PATH", tmp_path / "priors.json")
    monkeypatch.setattr(update, "APPLIED_GAMES_PATH", tmp_path / "applied_games.json")
    assert update.START_PRIORS_PATH == PRE
    update.main(["--start-season"])
    assert load(tmp_path / "priors.json") == live_priors()
    for team in NBA_TEAMS:
        assert load(tmp_path / "priors.json")[team]["pace"]["std"] == pytest.approx(params.pace_season_std(20))


@pytest.mark.parametrize("path", [PRE, PREVIOUS["pre"]])
def test_live_and_previous_preseason_files_are_never_written(path):
    before = sha(path)
    with pytest.raises(PermissionError):
        update.write_json(path, {})
    with pytest.raises(SystemExit):
        dynamic_model.main(["--out", str(path), "--before", "2026-10-03", "--overwrite"])
    assert sha(path) == before


# ---------------------------------------------------------------------------
# train_calibration.py
# ---------------------------------------------------------------------------

def test_calibration_variants_spell_out_every_switch():
    for name, v in train_calibration.VARIANTS.items():
        assert set(v) == {"sigma", "sim_corr", "pace"}, name
    assert train_calibration.VARIANTS["live"] == {"sigma": params.DEFAULT_SIM_SIGMA_MODE,
                                                  "sim_corr": params.DEFAULT_SIM_CORR_MODE,
                                                  "pace": params.DEFAULT_PACE_MODE}
    assert train_calibration.VARIANTS["old-live"] == {"sigma": OLD_LIVE["sigma_sim"], "sim_corr": OLD_LIVE["sim_corr"],
                                                      "pace": OLD_LIVE["pace"]}


# ---------------------------------------------------------------------------
# the real season: reproduction, and the switch
# ---------------------------------------------------------------------------

def applied_games():
    return load(REPO_ROOT / "applied_games.json")


def caches(*folders):
    return all((REPO_ROOT / "stats" / folder / f"{gid}.json").exists() for folder in folders for gid in applied_games())


SCHEDULES = REPO_ROOT / "stats" / "schedules"
TRAINING_SCHEDULES = all((SCHEDULES / f"{s}.json").exists() for s in ("2023-24", "2024-25", "2025-26"))


@pytest.mark.skipif(not (caches("boxscores", "boxscores_advanced") and (SCHEDULES / "2026-27.json").exists()),
                    reason="needs stats/boxscores/, stats/boxscores_advanced/ and the cached 2026-27 schedule")
def test_default_replay_reproduces_log_pace_home_rest(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(PRE), "--out", str(out), "--log", str(log)])
    assert sha(log) == sha(LOG)
    assert load(out) == load(REPLAY)
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_default_training_reproduces_priors_pre_pace_home_rest(tmp_path):
    before = repo_data_hashes()
    out = tmp_path / "trained.json"
    dynamic_model.main(["--out", str(out), "--before", "2026-10-03"])
    trained = load(out)
    assert trained == load(PRE)
    pace = load(REPO_ROOT / "priors_pre_2026-27_pace.json")
    assert all(trained[t]["pace"] == pace[t]["pace"] for t in trained)   # the Phase 7 paces (no pace rest effect)
    assert trained != load(PREVIOUS["pre"])                              # rest moves the factor curves
    assert repo_data_hashes() == before


def test_live_files_are_the_rest_replay():
    """
    The switch: the replayed games are the first entries of applied_games.json, in the same
    (date, gameId) order, each recorded with the live settings, and priors.json is the replay
    for as long as no later game has been applied.
    """
    applied = applied_games()
    with open(LOG, encoding="utf-8") as f:
        logged = [r["game_id"] for r in csv.DictReader(f)]
    assert len(logged) == len(set(logged)) == 210
    assert list(applied)[:len(logged)] == logged == sorted(logged, key=lambda gid: (applied[gid]["date"], gid))
    if set(applied) == set(logged):
        assert load(REPO_ROOT / "priors.json") == load(REPLAY)
    assert all({k: applied[gid][k] for k in LIVE} == LIVE for gid in logged)
    assert all(isinstance(e["neutral"], bool) and "pending" not in e for e in applied.values())
    assert {gid for gid, e in applied.items() if e["neutral"]} <= {"0022600189", "0062600001", "0022600075",
                                                                  "0022600076"}


@pytest.mark.parametrize("target", ["priors_pre_2026-27_pace-home-rest.json", "priors_2026-27_pace-home-rest.json",
                                    "log_pace-home-rest.csv", "priors_pre_2026-27_pace-home.json",
                                    "priors_2026-27_pace-home.json", "log_pace-home.csv", "priors.json",
                                    "applied_games.json"])
def test_live_and_previous_files_are_never_overwritten(env, tmp_path, target):
    update.run_update()
    before = repo_data_hashes()
    real = REPO_ROOT / target
    other = tmp_path / ("x.csv" if real.suffix == ".json" else "x.json")
    out, log = (real, other) if real.suffix == ".json" else (other, real)
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(PRE), "--out", str(out), "--log", str(log)])
    assert repo_data_hashes() == before
