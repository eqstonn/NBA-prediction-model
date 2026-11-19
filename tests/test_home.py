"""
Tests for the home-court boost (Phase 8: code/fit_home.py, the "home" switch in
bayes_update.py / update.py / dynamic_model.py / simulation.py, neutral sites in boxscores.py)
and code/train_calibration.py:

  - the fit recovers a known home gap from simulated games
  - the boost is added to the expected value in the update and the simulation, and
    neutral-site games get none (training, --update, --replay, the simulation, the site)
  - the neutral-site rule on NBA.com's schedule
  - --home off reproduces the Phase 6 log exactly; --home on reproduces log_home.csv
These tests use priors without paces, so they pass pace="off", rest="off"; the live defaults (pace and
home on) are tested in test_live.py.
  - nothing writes priors.json, applied_games.json or an existing priors/log file

Run from the repo root:  python -m pytest tests

The "reproduces" tests read the box-score caches, the cached NBA.com schedules
(stats/schedules/) and the Kaggle CSV, which are not in git; they are skipped when missing.
"""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fake_nba import EXPECTED, FAKE_SEASON, FakeNBA, ID, REPO_ROOT, update

import bayes_update  # noqa: E402  (code/ is on sys.path via fake_nba)
import boxscores  # noqa: E402
import build_site  # noqa: E402
import dynamic_model  # noqa: E402
import fit_home  # noqa: E402
import params  # noqa: E402
import predlog  # noqa: E402
import simulation  # noqa: E402
import train_calibration  # noqa: E402
from state_manager import default_priors  # noqa: E402

FACTORS = params.FACTORS
GAMES = json.loads((REPO_ROOT / "tests" / "fixtures" / "games_2026-27_first5.json").read_text(encoding="utf-8"))
SIGMA = {"eFG": 0.06, "TOV": 0.035, "ORB": 0.07, "FTR": 0.085}
BOOST = {"offense": {"eFG": 0.005, "TOV": -0.001, "ORB": 0.002, "FTR": 0.003},
         "defense": {"eFG": -0.005, "TOV": 0.001, "ORB": -0.002, "FTR": -0.003}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_data_hashes():
    return {p.name: sha(p) for p in sorted(REPO_ROOT.glob("*.json")) + sorted(REPO_ROOT.glob("*.csv"))}


def season_priors():
    return update.reset_priors(json.loads(update.FROZEN_PRIORS_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def home_params(tmp_path, monkeypatch):
    """A model_params.json with a known home boost, in a temp folder."""
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps({"sigma": SIGMA, "home": BOOST}), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    return BOOST


def schedule(games):
    """A ScheduleLeagueV2-shaped response: games = [(gameId, home teamId, away teamId, arenaState, isNeutral)]."""
    return {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [
        {"gameId": gid, "homeTeam": {"teamId": home}, "awayTeam": {"teamId": away}, "arenaState": state,
         "isNeutral": neutral} for gid, home, away, state, neutral in games]}]}}


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def test_home_gap_recovered_from_simulated_games():
    """Games simulated with a known home gap (and some neutral games without it): the estimator finds it."""
    rng = np.random.default_rng(21)
    teams = [f"T{i}" for i in range(10)]
    true_gap = {"eFG": 0.012, "TOV": -0.006, "ORB": 0.008, "FTR": 0.010}
    sigma = {"eFG": 0.04, "TOV": 0.03, "ORB": 0.05, "FTR": 0.05}
    start = default_priors(teams)[teams[0]]["offense"]
    strength = {t: {side: {f: start[f]["mean"] + rng.normal(0, 0.02) for f in FACTORS}
                    for side in ("offense", "defense")} for t in teams}
    games = []
    for k in range(1500):
        home, away = (str(t) for t in rng.choice(teams, 2, replace=False))
        neutral = k % 10 == 0
        for team, opp, side in ((home, away, "home"), (away, home, "away")):
            shift = 0.0 if neutral else (0.5 if side == "home" else -0.5)
            games.append({"team": team, "opponent": opp, "side": None if neutral else side,
                          **{f: float((strength[team]["offense"][f] + strength[opp]["defense"][f]) / 2
                                      + shift * true_gap[f] + rng.normal(0, sigma[f])) for f in FACTORS}})
    fit = fit_home.estimate_home(games, teams, sigma)
    for f in FACTORS:
        assert fit["se"][f] < 0.004, f
        assert abs(fit["delta"][f] - true_gap[f]) < 3.5 * fit["se"][f], (f, fit["delta"][f], true_gap[f])
        assert fit["log_likelihood"][f] > fit["log_likelihood_without_home"][f]


def test_split_is_even():
    split = fit_home.split({"eFG": 0.01, "TOV": -0.002, "ORB": 0.0, "FTR": 0.004})
    assert split["offense"] == {"eFG": 0.005, "TOV": -0.001, "ORB": 0.0, "FTR": 0.002}
    assert split["defense"] == {"eFG": -0.005, "TOV": 0.001, "ORB": -0.0, "FTR": -0.002}


def test_real_home_section_is_from_training_data_only():
    home = json.loads((REPO_ROOT / "model_params.json").read_text(encoding="utf-8"))["home"]
    assert params.check_home(home) == {side: home[side] for side in ("offense", "defense")}
    assert home["fit"]["data"]["before"] <= dynamic_model.TRAINING_CUTOFF and home["fit"]["data"]["last"] < "2026-10-03"
    for f in FACTORS:   # an even split of the gap, and the eFG edge is clear (> 3 SE)
        assert home["offense"][f] == pytest.approx(-home["defense"][f])
        assert home["offense"][f] - home["defense"][f] == pytest.approx(home["fit"]["delta"][f], abs=1e-5)
    assert home["fit"]["delta"]["eFG"] > 3 * home["fit"]["delta_se"]["eFG"]
    assert all(gid[:3] in ("002", "006") and boxscores.season_of(gid) < "2026-27"
               for gid in home["fit"]["neutral_site_games"])


# ---------------------------------------------------------------------------
# neutral sites
# ---------------------------------------------------------------------------

def test_neutral_site_rule():
    sas, ind = "1610612759", "1610612754"
    raw = schedule([
        ("0022300001", sas, ind, "TX", False),    # San Antonio: home
        ("0022300002", sas, ind, "TX", False),
        ("0022300003", sas, ind, "TX", False),    # Austin (same state): still a home game, as NBA.com has it
        ("0022300004", sas, ind, "MX", False),    # Mexico City, 2023-24 style (no isNeutral): neutral by state
        ("0022300005", sas, ind, "FR", True),     # Paris, flagged
        ("0022300006", ind, sas, "IN", False),    # Indianapolis: home
        ("0022300007", ind, sas, "IN", False),
        ("0062300001", ind, sas, "IN", True),     # Cup final at Hinkle Fieldhouse, flagged: neutral in the home city
        ("0062300002", None, None, "NV", True),   # a final whose teams aren't known yet
    ])
    flags = boxscores.neutral_flags(raw)
    assert {gid for gid, n in flags.items() if n} == {"0022300004", "0022300005", "0062300001", "0062300002"}


def test_season_and_game_keys():
    assert boxscores.game_key(22300172) == "0022300172"
    assert boxscores.season_of("0022300172") == "2023-24"
    assert boxscores.season_of(62500001) == "2025-26"


def test_neutral_games_downloads_a_season_once_and_refreshes_for_new_games(tmp_path, monkeypatch):
    monkeypatch.setattr(boxscores, "SCHEDULE_CACHE_DIR", tmp_path / "schedules")
    monkeypatch.setattr(boxscores, "PAUSE", 0)
    calls = []
    listed = [("0022600001", "1", "2", "IN", False)]

    def download(season):
        calls.append(season)
        return schedule(listed)

    monkeypatch.setattr(boxscores, "_download_schedule", download)
    assert boxscores.neutral_games(["0022600001"]) == {"0022600001": False}
    assert boxscores.neutral_games(["0022600001"]) == {"0022600001": False}
    assert calls == ["2026-27"]                                   # cached after the first call
    listed.append(("0022600189", "1", "2", "MX", True))
    assert boxscores.neutral_games(["0022600189"]) == {"0022600189": True}
    assert calls == ["2026-27", "2026-27"]                        # a game not in the cache -> one refresh
    with pytest.raises(KeyError):
        boxscores.neutral_games(["0022699999"])


SCHEDULE_2026_27 = REPO_ROOT / "stats" / "schedules" / "2026-27.json"


@pytest.mark.skipif(not SCHEDULE_2026_27.exists(), reason="needs the cached 2026-27 schedule in stats/schedules/")
def test_2026_27_neutral_games():
    flags = boxscores.neutral_flags(json.loads(SCHEDULE_2026_27.read_text(encoding="utf-8")))
    season = {gid for gid, n in flags.items() if n and gid[:3] in ("002", "006")}
    # Mexico City (Nov 7), the NBA Cup final (Hinkle Fieldhouse), Paris, Manchester
    assert season == {"0022600189", "0062600001", "0022600075", "0022600076"}
    applied = json.loads((REPO_ROOT / "applied_games.json").read_text(encoding="utf-8"))
    assert {gid for gid in applied if flags[gid]} == {"0022600189"}
    assert flags["0022600009"] is False                           # HOU @ SAS in Austin is a Spurs home game


# ---------------------------------------------------------------------------
# the boost in the update and the simulation
# ---------------------------------------------------------------------------

def test_home_offsets(home_params):
    assert params.home_offsets("off") == (None, None)
    assert params.home_offsets("on", neutral=True) == (None, None)
    assert params.home_offsets("on") == (BOOST["offense"], BOOST["defense"])


def test_update_adds_the_offset_to_the_expected_value():
    priors = season_priors()
    home, away = GAMES[0]["home"], GAMES[0]["away"]
    row = {"team": home["id"], "opponent": away["id"], **{f: home[f] for f in FACTORS}}
    before = copy.deepcopy(priors)
    bayes_update.update_game(priors, row, sigma=SIGMA, offset=BOOST["offense"])
    for f in FACTORS:
        o, d = before[home["id"]]["offense"][f], before[away["id"]]["defense"][f]
        expected = bayes_update.conjugate_pair(o["mean"], o["std"], d["mean"], d["std"],
                                               home[f] - BOOST["offense"][f], SIGMA[f])
        assert (priors[home["id"]]["offense"][f]["mean"], priors[home["id"]]["offense"][f]["std"]) == expected[0]
        assert (priors[away["id"]]["defense"][f]["mean"], priors[away["id"]]["defense"][f]["std"]) == expected[1]
    no_offset, zero = season_priors(), season_priors()
    bayes_update.update_game(no_offset, row, sigma=SIGMA)
    bayes_update.update_game(zero, row, sigma=SIGMA, offset={f: 0.0 for f in FACTORS})
    assert no_offset == zero


def test_apply_game_boosts_both_lines_but_not_at_neutral_sites(home_params):
    game = {**GAMES[0], "neutral": False}
    off = update.apply_game(season_priors(), game, pace="off", rest="off", home="off")
    on = update.apply_game(season_priors(), game, pace="off", rest="off", home="on")
    neutral = update.apply_game(season_priors(), {**game, "neutral": True}, pace="off", rest="off", home="on")
    assert neutral == off and on != off
    expected = season_priors()
    tau = update.season_tau(update.DEFAULT_M)
    for side, other, offset in (("home", "away", BOOST["offense"]), ("away", "home", BOOST["defense"])):
        row = {"team": game[side]["id"], "opponent": game[other]["id"], **{f: game[side][f] for f in FACTORS}}
        bayes_update.update_game(expected, row, tau=tau, offset=offset)
    assert on == expected
    with pytest.raises(KeyError, match="neutral"):
        update.apply_game(season_priors(), GAMES[0], pace="off", rest="off", home="on")   # home on needs the flag


def test_simulation_boosts_the_home_team_but_not_at_neutral_sites(home_params):
    priors = season_priors()
    h, a = GAMES[0]["home"]["id"], GAMES[0]["away"]["id"]
    off = simulation.simulate_factors(h, a, priors, seed=3, pace="off", rest="off", home="off")
    on = simulation.simulate_factors(h, a, priors, seed=3, pace="off", rest="off", home="on")
    neutral = simulation.simulate_factors(h, a, priors, seed=3, pace="off", rest="off", home="on", neutral=True)
    assert neutral == off
    margin = lambda r: r["team_a_avg_score"] - r["team_b_avg_score"]   # noqa: E731
    assert margin(on) > margin(off) + 1.5
    assert on["team_a_win_pct"] > off["team_a_win_pct"]


@pytest.mark.parametrize("draws", ["independent", "correlated"])
def test_both_draw_methods_shift_each_line_by_its_offset(draws):
    """With exact curves (std 0), the mean of each simulated line is (o + d)/2 + its offset."""
    team = lambda off, dfn: {"offense": {f: {"mean": off, "std": 0.0} for f in FACTORS},   # noqa: E731
                             "defense": {f: {"mean": dfn, "std": 0.0} for f in FACTORS}}
    a, b = team(0.50, 0.54), team(0.52, 0.56)
    rng = np.random.default_rng(8)
    args = (rng, a, b, SIGMA) + ((np.eye(8).tolist(),) if draws == "correlated" else ())
    method = simulation.correlated_factors if draws == "correlated" else simulation.independent_factors
    a_game, b_game = method(*args, 40000, BOOST["offense"], BOOST["defense"])
    for f in FACTORS:
        assert a_game[f].mean() == pytest.approx((0.50 + 0.56) / 2 + BOOST["offense"][f], abs=4 * SIGMA[f] / 200)
        assert b_game[f].mean() == pytest.approx((0.52 + 0.54) / 2 + BOOST["defense"][f], abs=4 * SIGMA[f] / 200)


def test_site_passes_the_neutral_flag(home_params, monkeypatch):
    priors = season_priors()
    h, a = GAMES[0]["home"], GAMES[0]["away"]
    game = {"home": {"id": h["id"], "abbr": h["abbr"], "score": None},
            "away": {"id": a["id"], "abbr": a["abbr"], "score": None}, "status": "scheduled"}
    seen = []
    real = build_site.predict_game
    monkeypatch.setattr(build_site, "predict_game", lambda *args, **kw: seen.append(kw) or real(*args, **kw))
    build_site.build_predictions([{**game, "neutral": True}, {**game, "neutral": False}], priors, pace="off", rest="off",
                                 home="on")
    assert [(kw["home"], kw["neutral"]) for kw in seen] == [("on", True), ("on", False)]


# ---------------------------------------------------------------------------
# training, --update and --replay with --home on
# ---------------------------------------------------------------------------

def write_training_csv(path, dated_games):
    rows = []
    for day, g in dated_games:
        for side, other in (("home", "away"), ("away", "home")):
            rows.append({"gameId": int(g["game_id"]), "gameType": "Regular Season", "gameDateTimeEst": f"{day} 19:30:00",
                         "teamId": int(g[side]["id"]), "teamName": g[side]["abbr"],
                         "opponentTeamId": int(g[other]["id"]), "opponentTeamName": g[other]["abbr"],
                         "home": int(side == "home"),
                         "effectiveFieldGoalPercentage": g[side]["eFG"], "teamTurnoverPercentage": g[side]["TOV"],
                         "offensiveReboundPercentage": g[side]["ORB"], "freeThrowAttemptRate": g[side]["FTR"]})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_training_adds_the_boost_except_at_neutral_sites(home_params, tmp_path, monkeypatch):
    dated = [("2025-01-10", GAMES[0]), ("2025-02-10", GAMES[1]), ("2025-03-10", GAMES[2]), ("2025-04-10", GAMES[3])]
    neutral_id = GAMES[1]["game_id"]
    csv, out_on, out_off = tmp_path / "games.csv", tmp_path / "on.json", tmp_path / "off.json"
    write_training_csv(csv, dated)
    monkeypatch.setattr(boxscores, "SCHEDULE_CACHE_DIR", tmp_path / "schedules")
    monkeypatch.setattr(boxscores, "PAUSE", 0)
    monkeypatch.setattr(boxscores, "_download_schedule", lambda season: schedule(
        [(g["game_id"], g["home"]["id"], g["away"]["id"], "XX", g["game_id"] == neutral_id) for _, g in dated]))
    dynamic_model.main(["--csv", str(csv), "--out", str(out_on), "--before", "2026-10-20", "--home", "on",
                        "--pace", "off", "--rest", "off"])
    dynamic_model.main(["--csv", str(csv), "--out", str(out_off), "--before", "2026-10-20", "--home", "off",
                        "--pace", "off", "--rest", "off"])

    teams = sorted({g[side]["id"] for _, g in dated for side in ("home", "away")})
    expected = default_priors(teams)
    for _, g in dated:
        for side, other, offset in (("home", "away", BOOST["offense"]), ("away", "home", BOOST["defense"])):
            row = {"team": g[side]["id"], "opponent": g[other]["id"], **{f: g[side][f] for f in FACTORS}}
            bayes_update.update_game(expected, row, offset=None if g["game_id"] == neutral_id else offset)
    trained = json.loads(out_on.read_text(encoding="utf-8"))
    for team in teams:
        for side in ("offense", "defense"):
            for f in FACTORS:
                assert trained[team][side][f]["mean"] == pytest.approx(expected[team][side][f]["mean"], abs=1e-12)
                assert trained[team][side][f]["std"] == pytest.approx(expected[team][side][f]["std"], abs=1e-12)
    assert trained != json.loads(out_off.read_text(encoding="utf-8"))


@pytest.fixture
def fits(env, monkeypatch):
    """Record what the (fake) fit sees: the observation after the home offset."""
    seen = []
    fake = bayes_update._fit

    def spy(prior, observed, random_seed=None, engine=None, sigma=None):
        seen.append(dict(observed))
        return fake(prior, observed, random_seed, engine, sigma)

    monkeypatch.setattr(bayes_update, "_fit", spy)
    return seen


def raw_factors(game_id):
    return boxscores.parse_factors(FakeNBA().box_score(game_id))


def test_update_run_home_on_skips_the_boost_at_neutral_sites(env, fits, home_params):
    env.nba.neutral.add("0022600015")
    result = update.run_update(home="on", pace="off", rest="off")
    assert result["applied"] == EXPECTED
    applied = env.applied()
    assert {gid for gid, e in applied.items() if e["neutral"]} == {"0022600015"}
    assert {e["home"] for e in applied.values()} == {"on"}
    for i, gid in enumerate(EXPECTED):
        raw = raw_factors(gid)
        for side, offset, observed in (("home", BOOST["offense"], fits[2 * i]), ("away", BOOST["defense"], fits[2 * i + 1])):
            shift = {f: 0.0 for f in FACTORS} if gid == "0022600015" else offset
            assert observed == pytest.approx({f: raw[side][f] - shift[f] for f in FACTORS}, abs=1e-15), (gid, side)


def test_update_run_home_off_uses_no_offset(env, fits):
    update.run_update(home="off")
    assert all(observed == {f: raw_factors(gid)[side][f] for f in FACTORS}
               for (gid, side), observed in zip([(g, s) for g in EXPECTED for s in ("home", "away")], fits))
    assert {(e["home"], e["neutral"]) for e in env.applied().values()} == {("off", False)}


def test_replay_home_on_reads_cached_schedule_and_home_off_never_does(env, home_params, tmp_path, monkeypatch):
    env.nba.neutral.add("0022600015")
    update.run_update(pace="off", rest="off")                        # caches the schedule in the temp folder
    monkeypatch.setattr(boxscores, "_download_schedule", lambda season: pytest.fail("no download expected"))
    update.main(["--replay", "--from", str(update.FROZEN_PRIORS_PATH), "--out", str(tmp_path / "on.json"),
                 "--log", str(tmp_path / "on.csv"), "--home", "on", "--pace", "off", "--rest", "off"])
    assert [r["game_id"] for r in predlog.read_log(tmp_path / "on.csv")] == EXPECTED
    for p in boxscores.SCHEDULE_CACHE_DIR.glob("*.json"):
        p.unlink()
    update.main(["--replay", "--from", str(update.FROZEN_PRIORS_PATH), "--out", str(tmp_path / "off.json"),
                 "--log", str(tmp_path / "off.csv"), "--home", "off", "--pace", "off", "--rest", "off"])
    assert json.loads((tmp_path / "on.json").read_text(encoding="utf-8")) != \
        json.loads((tmp_path / "off.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# defaults, settings, train_calibration.py
# ---------------------------------------------------------------------------

def test_home_defaults_on_everywhere():
    assert params.DEFAULT_HOME_MODE == "on"
    assert update.make_parser().parse_args([]).home == "on"
    assert dynamic_model.make_parser().parse_args(["--out", "x.json", "--before", "2026-10-03"]).home == "on"
    assert build_site.make_parser().parse_args([]).home == "on"
    priors = season_priors()
    h, a = GAMES[0]["home"]["id"], GAMES[0]["away"]["id"]
    default = simulation.simulate_factors(h, a, priors, seed=4, pace="off", rest="off")
    assert default == simulation.simulate_factors(h, a, priors, seed=4, pace="off", rest="off", home="on")
    assert default != simulation.simulate_factors(h, a, priors, seed=4, pace="off", rest="off", home="off")


def test_bad_home_settings_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        params.home_offsets("yes")
    with pytest.raises(ValueError):
        params.check_home({"offense": {"eFG": 0.01}, "defense": BOOST["defense"]})
    for argv in (["--update", "--home", "yes"], ["--replay", "--home", "1"]):
        with pytest.raises(SystemExit):
            update.main(argv)
    with pytest.raises(SystemExit):
        build_site.make_parser().parse_args(["--home", "maybe"])
    (tmp_path / "p.json").write_text(json.dumps({"sigma": SIGMA}), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", tmp_path / "p.json")
    with pytest.raises(KeyError, match="fit_home"):
        params.home_offsets("on")


def test_fit_home_refuses_2026_27_data_and_an_existing_section(tmp_path):
    with pytest.raises(SystemExit):
        fit_home.main(["--before", "2026-10-20", "--params", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit):   # model_params.json already has a home section
        fit_home.main([])


def test_train_calibration_summary_and_cutoff():
    #       p_home, home margin, projected margin, projected total, actual total, sim margin SD
    records = [(0.7, 5, 4.0, 220.0, 225, 14.0), (0.4, 3, -2.0, 230.0, 228, 15.0), (0.6, -8, 2.0, 225.0, 210, 13.0)]
    s = train_calibration.summarize(records)
    # favored 0.7 (home, won), 0.6 (away, lost), 0.6 (home, lost)
    assert s["brier_each"].tolist() == pytest.approx([0.09, 0.36, 0.36])
    assert s["accuracy"] == pytest.approx(1 / 3) and s["confidence"] == pytest.approx((0.7 + 0.6 + 0.6) / 3)
    assert s["proj_home"] == pytest.approx(4 / 3) and s["actual_home"] == pytest.approx(0.0)
    assert s["proj_total"] == pytest.approx(225.0) and s["sim_sd"] == pytest.approx(14.0)
    with pytest.raises(SystemExit):
        train_calibration.main(["--before", "2026-10-20"])


# ---------------------------------------------------------------------------
# reproducibility on the real season, and protected files
# ---------------------------------------------------------------------------

def caches(*folders):
    applied = json.loads((REPO_ROOT / "applied_games.json").read_text(encoding="utf-8"))
    return all((REPO_ROOT / "stats" / folder / f"{gid}.json").exists() for folder in folders for gid in applied)


TRAINING_SCHEDULES = all((REPO_ROOT / "stats" / "schedules" / f"{s}.json").exists()
                         for s in ("2023-24", "2024-25", "2025-26"))


@pytest.mark.skipif(not caches("boxscores"), reason="needs the box-score cache in stats/boxscores/")
def test_home_off_reproduces_the_phase6_log(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_sigma-update.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--home", "off", "--pace", "off", "--rest", "off", "--sigma-sim", "legacy"])
    assert sha(log) == sha(REPO_ROOT / "log_sigma-update.csv")
    assert sha(out) == sha(REPO_ROOT / "priors_2026-27_sigma-update.json")
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (caches("boxscores") and SCHEDULE_2026_27.exists()),
                    reason="needs stats/boxscores/ and the cached 2026-27 schedule")
def test_home_on_replay_reproduces_log_home(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_home.json"), "--out", str(out),
                 "--log", str(log), "--n", "20", "--m", "30", "--home", "on", "--pace", "off", "--rest", "off", "--sigma-sim", "legacy"])
    assert sha(log) == sha(REPO_ROOT / "log_home.csv")
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_2026-27_home.json").read_text(encoding="utf-8"))
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_home_training_reproduces_priors_pre_home(tmp_path):
    before = repo_data_hashes()
    out = tmp_path / "trained.json"
    dynamic_model.main(["--out", str(out), "--before", "2026-10-03", "--home", "on", "--pace", "off", "--rest", "off"])
    assert json.loads(out.read_text(encoding="utf-8")) == \
        json.loads((REPO_ROOT / "priors_pre_2026-27_home.json").read_text(encoding="utf-8"))
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_fit_home_reproduces_model_params_and_keeps_the_rest(tmp_path):
    before = repo_data_hashes()
    path = tmp_path / "model_params.json"
    path.write_bytes((REPO_ROOT / "model_params.json").read_bytes())
    fit_home.main(["--params", str(path), "--overwrite"])
    new, old = (json.loads(p.read_text(encoding="utf-8")) for p in (path, REPO_ROOT / "model_params.json"))
    assert {k: v for k, v in new.items() if k != "home"} == {k: v for k, v in old.items() if k != "home"}
    strip = lambda h: {**h, "fit": {k: v for k, v in h["fit"].items() if k != "fitted"}}   # noqa: E731
    assert strip(new["home"]) == strip(old["home"])
    assert repo_data_hashes() == before


@pytest.mark.parametrize("target", ["priors.json", "applied_games.json", "model_params.json",
                                    "priors_pre_2026-27_home.json", "log_sigma-update.csv", "log_home.csv"])
def test_home_replay_never_overwrites_existing_files(env, tmp_path, target):
    update.run_update()
    before = repo_data_hashes()
    real = REPO_ROOT / target
    other = tmp_path / ("x.csv" if real.suffix == ".json" else "x.json")
    out, log = (real, other) if real.suffix == ".json" else (other, real)
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(REPO_ROOT / "priors_pre_2026-27_home.json"), "--out", str(out),
                     "--log", str(log), "--home", "on", "--pace", "off", "--rest", "off"])
    assert repo_data_hashes() == before


@pytest.mark.parametrize("name", ["priors.json", "applied_games.json", "model_params.json", "priors_pre_2026-27.json"])
def test_home_training_refuses_protected_outputs(name):
    before = repo_data_hashes()
    with pytest.raises(SystemExit):
        dynamic_model.main(["--out", str(REPO_ROOT / name), "--before", "2026-10-03", "--home", "on"])
    assert repo_data_hashes() == before
