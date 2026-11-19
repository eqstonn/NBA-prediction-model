"""
Tests for rest days and back-to-backs (Phase 9: code/rest.py, code/fit_rest.py, the "rest"
switch in params.py / update.py / dynamic_model.py / simulation.py / predict.py /
build_site.py / train_calibration.py):

  - rest days from a schedule: back-to-backs, 1 day, 2+, travel, season openers, cross-season
    gaps, games that don't count (preseason, All-Star, postponed, teams not known yet)
  - on the real data: the schedules' rest agrees with the Kaggle dates for every training row
  - the fit recovers a known rest effect from simulated games, and its linear shortcut is exact
  - the effect enters the expected value in the update and the simulation, only with rest on
  - --rest off reproduces the pace-home model (log_pace-home.csv) exactly; --rest on (the
    default since the rest switch) reproduces the Phase 9 training, replay and log
  - nothing writes priors.json, applied_games.json or an existing priors/log file

Run from the repo root:  python -m pytest tests

The "reproduces" tests read the box-score caches, the cached NBA.com schedules and the
Kaggle CSV, which are not in git; they are skipped when missing.
"""
import copy
import hashlib
import json
import math

import numpy as np
import pytest

import fake_nba
from fake_nba import EXPECTED, ID, REPO_ROOT, update

import bayes_update  # noqa: E402  (code/ is on sys.path via fake_nba)
import boxscores  # noqa: E402
import build_site  # noqa: E402
import dynamic_model  # noqa: E402
import fit_rest  # noqa: E402
import params  # noqa: E402
import predict  # noqa: E402
import predlog  # noqa: E402
import rest  # noqa: E402
import simulation  # noqa: E402
import train_calibration  # noqa: E402
from schedule import NBA_TEAMS  # noqa: E402
from state_manager import default_priors  # noqa: E402

FACTORS = params.FACTORS
GAMES = json.loads((REPO_ROOT / "tests" / "fixtures" / "games_2026-27_first5.json").read_text(encoding="utf-8"))
EFFECTS = {"offense": {"b2b": {"eFG": -0.01, "TOV": 0.002, "ORB": -0.004, "FTR": -0.001}},
           "defense": {"b2b": {"eFG": 0.01, "TOV": -0.002, "ORB": 0.004, "FTR": 0.001}}}
PRE = REPO_ROOT / "priors_pre_2026-27_pace-home.json"
PRE_REST = REPO_ROOT / "priors_pre_2026-27_pace-home-rest.json"
REPLAY_REST = REPO_ROOT / "priors_2026-27_pace-home-rest.json"
LOG_REST = REPO_ROOT / "log_pace-home-rest.csv"
A, B, C = "1610612738", "1610612752", "1610612765"   # BOS, NYK, DET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_data_hashes():
    return {p.name: sha(p) for p in sorted(REPO_ROOT.glob("*.json")) + sorted(REPO_ROOT.glob("*.csv"))}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def schedule(games):
    """ScheduleLeagueV2 shape: games = [(gameId, date, home, away, arena)] (+ optional postponedStatus)."""
    days = {}
    for gid, day, home, away, arena, *postponed in games:
        days.setdefault(day, []).append({
            "gameId": gid, "gameDateEst": f"{day}T00:00:00Z", "arenaName": arena,
            "postponedStatus": postponed[0] if postponed else "N",
            "homeTeam": {"teamId": int(home) if home else 0}, "awayTeam": {"teamId": int(away) if away else 0}})
    return {"leagueSchedule": {"gameDates": [{"gameDate": d, "games": g} for d, g in sorted(days.items())]}}


# BOS's 2026-27 season, as a schedule
SEASON = schedule([
    ("0012600010", "2026-10-10", A, B, "TD Garden"),             # preseason: doesn't count
    ("0022600001", "2026-10-20", C, A, "Little Caesars Arena"),  # opener -> 2+
    ("0022600002", "2026-10-21", B, A, "Madison Square Garden"), # b2b, traveled
    ("0022600003", "2026-10-22", B, A, "Madison Square Garden"), # b2b again, same arena (NYK's 2nd b2b too)
    ("0022600004", "2026-10-24", A, C, "TD Garden"),             # 1 day
    ("0022600005", "2026-10-28", A, B, "TD Garden"),             # 3 days -> 2+
    ("0022600006", "2026-10-29", C, A, "Little Caesars Arena", "Y"),   # postponed: doesn't count
    ("0032600001", "2026-10-29", A, B, "All-Star"),              # All-Star: doesn't count
    ("0022600007", "2026-10-30", None, None, ""),                # teams not known yet (Cup knockout)
    ("0022600008", "2026-10-30", A, C, "TD Garden"),             # 1 day (after 10-28; the 10-29 games don't count)
])


# ---------------------------------------------------------------------------
# rest days
# ---------------------------------------------------------------------------

def test_categories():
    assert [rest.category(d) for d in (None, 0, 1, 2, 7)] == ["2+", "b2b", "1", "2+", "2+"]
    with pytest.raises(ValueError):
        rest.category(-1)


def test_rest_days_back_to_backs_travel_and_openers():
    table = rest.rest_flags(SEASON)
    bos = {gid: info[A] for gid, info in table.items() if A in info}
    assert sorted(bos) == ["0022600001", "0022600002", "0022600003", "0022600004", "0022600005", "0022600008"]
    assert bos["0022600001"] == {"days": None, "rest": "2+", "traveled": None, "opener": True}
    assert bos["0022600002"] == {"days": 0, "rest": "b2b", "traveled": True, "opener": False}
    assert bos["0022600003"] == {"days": 0, "rest": "b2b", "traveled": False, "opener": False}
    assert bos["0022600004"] == {"days": 1, "rest": "1", "traveled": None, "opener": False}
    assert bos["0022600005"] == {"days": 3, "rest": "2+", "traveled": None, "opener": False}
    assert bos["0022600008"]["days"] == 1                              # postponed + All-Star games skipped
    assert table["0022600002"][B]["opener"] and table["0022600003"][B]["rest"] == "b2b"
    assert table["0022600003"][B]["traveled"] is False                 # NYK at home both nights
    assert "0022600006" not in table and "0032600001" not in table and "0022600007" not in table


def test_a_season_opener_is_2_plus_even_right_after_last_season():
    """The previous season's last game never counts, however close (here: the night before)."""
    last_season = schedule([("0022500999", "2026-10-19", A, B, "TD Garden")])
    games = rest.schedule_games(last_season) + rest.schedule_games(SEASON)
    table = rest.rest_table(games)
    assert table["0022600001"][A] == {"days": None, "rest": "2+", "traveled": None, "opener": True}
    assert table["0022500999"][A]["opener"]
    # within one season a long gap is just 2+
    summer = rest.rest_table(rest.schedule_games(schedule([("0022600001", "2026-10-20", A, B, "x"),
                                                           ("0042600001", "2027-04-20", A, B, "x")])))
    assert summer["0042600001"][A] == {"days": 181, "rest": "2+", "traveled": None, "opener": False}


def test_two_games_on_one_day_is_an_error():
    with pytest.raises(ValueError, match="two games"):
        rest.rest_flags(schedule([("0022600001", "2026-10-20", A, B, "x"), ("0022600002", "2026-10-20", A, C, "y")]))


def test_rest_games_reads_the_cache_and_refreshes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(boxscores, "SCHEDULE_CACHE_DIR", tmp_path / "schedules")
    monkeypatch.setattr(boxscores, "PAUSE", 0)
    calls = []
    monkeypatch.setattr(boxscores, "_download_schedule", lambda season: calls.append(season) or SEASON)
    assert rest.rest_games(["0022600002"])["0022600002"][A]["rest"] == "b2b"
    assert rest.rest_games(["0022600004"])["0022600004"][A]["rest"] == "1"
    assert calls == ["2026-27"]
    with pytest.raises(KeyError):
        rest.rest_games(["0022600999"])
    assert calls == ["2026-27", "2026-27"]


def test_website_lookup_by_team_and_date():
    lookup = rest.rest_lookup(SEASON)
    assert lookup(A, "2026-10-21") == rest.rest_flags(SEASON)["0022600002"][A]       # a game in the schedule
    assert lookup(A, "2026-10-25") == {"days": 0, "rest": "b2b", "traveled": None, "opener": False}
    assert lookup(A, "2026-10-01")["opener"] and lookup("1610612737", "2026-10-25")["rest"] == "2+"


SCHEDULES = REPO_ROOT / "stats" / "schedules"
TRAINING_SCHEDULES = all((SCHEDULES / f"{s}.json").exists() for s in ("2023-24", "2024-25", "2025-26"))


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_schedule_rest_agrees_with_kaggle_dates_for_every_training_row():
    data = dynamic_model.load_training_rows(dynamic_model.DEFAULT_CSV, "2026-10-03", 3)
    rows = dynamic_model.row_rest(data)
    cleaned = data["cleaned"]
    played = {}
    for gid, team, opp, when in zip(cleaned["gameId"], cleaned["teamId"], cleaned["opponentTeamId"],
                                    cleaned["gameDateTimeEst"]):
        if str(int(team)) not in NBA_TEAMS or str(int(opp)) not in NBA_TEAMS:
            continue   # All-Star tournament teams play several games a day; they never get a rest effect
        key = boxscores.game_key(gid)
        played.setdefault((str(int(team)), boxscores.season_of(key)), []).append((str(when)[:10], key))
    kaggle = {}
    for (team, _), games in played.items():
        games.sort()
        for i, (day, gid) in enumerate(games):
            prev = games[i - 1][0] if i else None
            days = None if prev is None else (np.datetime64(day) - np.datetime64(prev)).astype(int) - 1
            kaggle[(gid, team)] = (rest.category(days), prev, day)
    # NBA.com has 6 games the Kaggle CSV lacks altogether (the 2023 and 2024 NBA Cup finals and 4
    # regular-season games); they were played, so they count for rest. Every disagreement must be one
    # of those games falling between the Kaggle "previous game" and this one.
    in_kaggle = {k for k, _ in kaggle}
    schedule_games = [g for s in ("2023-24", "2024-25", "2025-26")
                      for g in rest.schedule_games(load(SCHEDULES / f"{s}.json"))]
    missing = [g for g in schedule_games if g["game_id"] not in in_kaggle and "2023-10-03" <= g["date"] < "2026-10-03"]
    assert sorted(g["game_id"] for g in missing) == ["0022400532", "0022400537", "0022400538", "0022400617",
                                                     "0062300001", "0062400001"]
    checked, explained = 0, 0
    for gid, team, r in zip(cleaned["gameId"], cleaned["teamId"], rows):
        if r is None:
            continue
        category, prev, day = kaggle[(boxscores.game_key(gid), str(int(team)))]
        checked += 1
        if r[0]["rest"] != category:
            assert any(str(int(team)) in (g["home"], g["away"]) and prev < g["date"] < day for g in missing)
            explained += 1
    assert checked == 7912 and explained == 10
    counts = {c: sum(r is not None and r[0]["rest"] == c for r in rows) for c in rest.CATEGORIES}
    assert counts == {"b2b": 1309, "1": 4899, "2+": 1704}


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def simulated_games(beta, sigma, n_games=2500, seed=5):
    """Games from the model with a known back-to-back difference effect; returns (games, teams, row categories)."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    start = default_priors(teams)[teams[0]]["offense"]
    strength = {t: {side: {f: start[f]["mean"] + rng.normal(0, 0.02) for f in FACTORS}
                    for side in ("offense", "defense")} for t in teams}
    cats = ["b2bT", "b2bS", "1", "2+"]
    games, row_cats = [], []
    for _ in range(n_games):
        home, away = (str(t) for t in rng.choice(teams, 2, replace=False))
        rest_of = {home: rng.choice(cats, p=[0.12, 0.06, 0.6, 0.22]), away: rng.choice(cats, p=[0.12, 0.06, 0.6, 0.22])}
        for team, opp in ((home, away), (away, home)):
            tired = float(rest_of[team].startswith("b2b")) - float(rest_of[opp].startswith("b2b"))
            games.append({"team": team, "opponent": opp,
                          **{f: float((strength[team]["offense"][f] + strength[opp]["defense"][f]) / 2
                                      + beta[f] * tired + rng.normal(0, sigma[f])) for f in FACTORS}})
            row_cats.append((str(rest_of[team]), str(rest_of[opp])))
    return games, teams, row_cats


BETA = {"eFG": -0.012, "TOV": 0.004, "ORB": -0.008, "FTR": -0.006}
SIGMA = {"eFG": 0.04, "TOV": 0.03, "ORB": 0.05, "FTR": 0.05}


@pytest.fixture(scope="module")
def simulated_design():
    games, teams, row_cats = simulated_games(BETA, SIGMA)
    e0, v, z = fit_rest.design(games, teams, SIGMA, [None] * len(games), row_cats)
    return games, teams, row_cats, e0, v, z


def test_fit_recovers_a_known_rest_effect(simulated_design):
    *_, e0, v, z = simulated_design
    diff = fit_rest.fit_form(e0, v, z, fit_rest.FORMS["difference, b2b"][0])
    both = fit_rest.fit_form(e0, v, z, fit_rest.FORMS["both, b2b"][0])
    one_day = fit_rest.fit_form(e0, v, z, fit_rest.FORMS["difference, b2b + 1 day"][0])
    for f in FACTORS:
        theta, se = diff[f]["theta"][0], diff[f]["se"][0]
        assert se < 0.003 and abs(theta - BETA[f]) < 3.5 * se, (f, theta, BETA[f], se)
        assert diff[f]["gain"] > 0
        off, dfn = both[f]["theta"]                               # the both-teams form: offense = beta, defense = -beta
        assert abs(off - BETA[f]) < 3.5 * both[f]["se"][0] and abs(dfn + BETA[f]) < 3.5 * both[f]["se"][1]
        assert abs(one_day[f]["theta"][1]) < 3.5 * one_day[f]["se"][1]   # no 1-day effect in the data


def test_the_linear_shortcut_is_exact(simulated_design):
    """e(theta) = e0 - Z theta matches a direct training pass with the shifts, to rounding."""
    games, teams, row_cats, e0, v, z = simulated_design
    theta = dict(zip(fit_rest.BASIS, [-0.01, -0.02, 0.003, 0.012, 0.004, -0.002]))
    offsets = []
    for c in row_cats:
        shift = sum(t for name, t in theta.items()
                    if c[0 if name.startswith("off") else 1] == name.split("_")[1])
        offsets.append({f: shift for f in FACTORS} if shift else None)
    direct, v_direct = fit_rest.residuals(games, teams, SIGMA, offsets)
    predicted = e0 - sum(z[name] * t for name, t in theta.items())
    assert np.abs(direct - predicted).max() < 1e-12 and np.allclose(v_direct, v)


def test_chi2_tail():
    for x, k in ((3.841459, 1), (5.991465, 2), (9.487729, 4), (15.507313, 8)):
        assert fit_rest.chi2_sf(x, k) == pytest.approx(0.05, abs=1e-4)
    assert fit_rest.chi2_sf(0, 3) == 1.0


def test_margin_in_points():
    line, poss = {"eFG": 0.54, "TOV": 0.13, "ORB": 0.26, "FTR": 0.21}, 99.0
    tired = fit_rest.margin_points(EFFECTS, "b2b", "2+", line, poss)
    assert tired < -2 and fit_rest.margin_points(EFFECTS, "2+", "b2b", line, poss) == pytest.approx(-tired)
    assert fit_rest.margin_points(EFFECTS, "b2b", "b2b", line, poss) == pytest.approx(0, abs=1e-9)
    grad = fit_rest.margin_gradient(line, poss)
    first_order = sum(grad[f] * (EFFECTS["offense"]["b2b"][f] - EFFECTS["defense"]["b2b"][f]) for f in FACTORS)
    assert first_order == pytest.approx(tired, rel=0.02)


def test_real_rest_section_is_from_training_data_only():
    section = load(REPO_ROOT / "model_params.json")["rest"]
    effects = params.check_rest(section)
    assert set(effects["offense"]) == set(effects["defense"]) == {"b2b"}
    for f in FACTORS:   # the difference form
        assert effects["defense"]["b2b"][f] == -effects["offense"]["b2b"][f]
    fit = section["fit"]
    assert fit["form"] == "difference, b2b"
    assert fit["data"]["before"] <= dynamic_model.TRAINING_CUTOFF and fit["data"]["last"] < "2026-10-03"
    est = fit["estimate"]["eFG"]["b2b"]
    assert est["value"] < -3 * est["se"]                         # a tired team shoots worse, > 3 SE
    assert -4 < fit["margin_points"]["b2b vs 2+"] < -1 and not fit["pace"]["used"]


def test_fit_rest_refuses_2026_27_data_and_an_existing_section(tmp_path):
    with pytest.raises(SystemExit):
        fit_rest.main(["--before", "2026-10-20", "--params", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit):   # model_params.json already has a rest section
        fit_rest.main([])


# ---------------------------------------------------------------------------
# the switch: update and simulation
# ---------------------------------------------------------------------------

def test_rest_offsets():
    assert params.rest_offsets("off", "b2b", "2+", EFFECTS) == (None, None)
    a, b = params.rest_offsets("on", "b2b", "2+", EFFECTS)
    assert a == EFFECTS["offense"]["b2b"] and b == EFFECTS["defense"]["b2b"]
    assert params.rest_offset("on", "b2b", "b2b", EFFECTS) == {f: 0.0 for f in FACTORS}
    assert params.rest_offset("on", "1", "2+", EFFECTS) == {f: 0.0 for f in FACTORS}
    with pytest.raises(ValueError):
        params.rest_offset("on", "tired", "2+", EFFECTS)
    with pytest.raises(ValueError):
        params.rest_offsets("maybe", "b2b", "2+", EFFECTS)
    with pytest.raises(ValueError):
        params.check_rest({"offense": {"3": EFFECTS["offense"]["b2b"]}, "defense": {}})
    assert params.combine_offsets(None, None) is None
    home = {f: 0.01 for f in FACTORS}
    assert params.combine_offsets(home, None) is home
    assert params.combine_offsets(home, a) == {f: 0.01 + a[f] for f in FACTORS}


@pytest.fixture
def rest_params(tmp_path, monkeypatch):
    """The real model_params.json with a known rest section, in a temp folder."""
    data = load(REPO_ROOT / "model_params.json")
    data["rest"] = EFFECTS
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)
    return data


def live_priors():
    return update.reset_priors(load(PRE))


def test_apply_game_adds_the_rest_shift_to_the_home_boost(rest_params):
    game = {**GAMES[0], "possessions": 100.0, "neutral": False, "rest": {"home": "2+", "away": "b2b"}}
    on = update.apply_game(live_priors(), game, rest="on")
    off = update.apply_game(live_priors(), game, rest="off")
    assert on != off
    expected = live_priors()
    home_line, away_line = params.home_offsets("on")
    rest_home, rest_away = params.rest_offsets("on", "2+", "b2b")
    tau = update.season_tau(update.DEFAULT_M)
    for side, other, offset in (("home", "away", params.combine_offsets(home_line, rest_home)),
                                ("away", "home", params.combine_offsets(away_line, rest_away))):
        row = {"team": game[side]["id"], "opponent": game[other]["id"], **{f: game[side][f] for f in FACTORS}}
        bayes_update.update_game(expected, row, tau=tau, offset=offset)
    bayes_update.update_pace(expected, game["home"]["id"], game["away"]["id"], 100.0, tau=params.pace_tau(30))
    assert on == expected
    # equally rested teams: the difference form adds nothing
    level = {**game, "rest": {"home": "b2b", "away": "b2b"}}
    assert update.apply_game(live_priors(), level, rest="on") == update.apply_game(live_priors(), level)
    with pytest.raises(KeyError, match="rest"):
        update.apply_game(live_priors(), {k: v for k, v in game.items() if k != "rest"}, rest="on")


def test_simulation_moves_the_margin_toward_the_rested_team(rest_params):
    priors = live_priors()
    h, a = GAMES[0]["home"]["id"], GAMES[0]["away"]["id"]
    margin = lambda r: r["team_a_avg_score"] - r["team_b_avg_score"]   # noqa: E731
    off = simulation.simulate_factors(h, a, priors, seed=3, rest="off")
    tired_away = simulation.simulate_factors(h, a, priors, seed=3, rest="on", rest_a="2+", rest_b="b2b")
    tired_home = simulation.simulate_factors(h, a, priors, seed=3, rest="on", rest_a="b2b", rest_b="2+")
    assert margin(tired_away) > margin(off) + 2 and margin(tired_home) < margin(off) - 2
    assert tired_away["team_a_win_pct"] > off["team_a_win_pct"] > tired_home["team_a_win_pct"]
    # equal rest = the same draws as rest off
    assert simulation.simulate_factors(h, a, priors, seed=3, rest="on", rest_a="1", rest_b="1") == off
    with pytest.raises(ValueError, match="rest"):
        simulation.simulate_factors(h, a, priors, seed=3, rest="on")
    p = predict.predict_game(h, a, priors, seed=3, rest="on", rest_home="2+", rest_away="b2b")
    assert p["home_score"] == tired_away["team_a_avg_score"]


def test_site_uses_the_schedule_lookup(rest_params):
    priors = live_priors()
    h, a = GAMES[0]["home"], GAMES[0]["away"]
    game = {"date": "2026-11-20", "home": {"id": h["id"], "abbr": h["abbr"], "score": None},
            "away": {"id": a["id"], "abbr": a["abbr"], "score": None}, "status": "scheduled"}
    tired = {h["id"]: "2+", a["id"]: "b2b"}
    rows = build_site.build_predictions([game], priors, rest="on", rest_lookup=lambda t, d: {"rest": tired[t]})
    assert rows[0]["rest"] == {"home": "2+", "away": "b2b"}
    plain = build_site.build_predictions([game], priors, rest="off")
    assert "rest" not in plain[0]
    with pytest.raises(ValueError):
        build_site.build_predictions([game], priors)              # rest is on by default: it needs the lookup


def test_rest_defaults_on_everywhere():
    assert params.DEFAULT_REST_MODE == "on"
    assert update.make_parser().parse_args([]).rest == "on"
    assert dynamic_model.make_parser().parse_args(["--out", "x.json", "--before", "2026-10-03"]).rest == "on"
    assert build_site.make_parser().parse_args([]).rest == "on"
    import inspect
    for fn in (update.apply_game, update.run_update, update.replay, update.trial, update.log_prediction,
               dynamic_model.run_training, simulation.simulate_factors, predict.predict_game,
               build_site.build_predictions, train_calibration.run_pass):
        assert inspect.signature(fn).parameters["rest"].default == "on", fn.__name__
    assert inspect.signature(update.record_game).parameters["rest_mode"].default == "on"
    for argv in (["--update", "--rest", "yes"], ["--replay", "--rest", "1"]):
        with pytest.raises(SystemExit):
            update.main(argv)


# ---------------------------------------------------------------------------
# --update / --replay with rest on (fake NBA.com)
# ---------------------------------------------------------------------------

B2B_GAME = ("0022600021", "2026-11-02", "BOS", "MIA", 100, 99, ["Regular Season"])   # BOS played DET at home 11-01


@pytest.fixture
def b2b_season(env, monkeypatch):
    monkeypatch.setattr(fake_nba, "FAKE_SEASON", fake_nba.FAKE_SEASON + [B2B_GAME])
    return EXPECTED[:5] + [B2B_GAME[0], EXPECTED[5]]


def test_update_run_with_rest(env, b2b_season, monkeypatch):
    seen = {}
    real = update.apply_game
    monkeypatch.setattr(update, "apply_game", lambda priors, game, *a, **kw: seen.setdefault(game["game_id"], (game, kw))
                        and real(priors, game, *a, **kw))
    result = update.run_update(rest="on")
    assert result["applied"] == b2b_season
    applied = env.applied()
    assert {e["rest"] for e in applied.values()} == {"on"}
    game, kw = seen["0022600021"]
    assert kw["rest"] == "on" and game["rest"] == {"home": "2+", "away": "b2b"}     # MIA rested, BOS on a b2b
    assert seen["0022600020"][0]["rest"] == {"home": "2+", "away": "2+"}
    assert all(seen[g][0]["rest"] == {"home": "2+", "away": "2+"} for g in ("0022600001", "0022600002"))   # openers


def test_rest_changes_only_back_to_back_games(env, b2b_season, tmp_path):
    """rest on vs off: the same priors until the b2b game, different after it."""
    update.run_update(rest="on", limit=5)
    before_b2b_on = env.priors_text()
    env.use(tmp_path / "off")
    update.run_update(rest="off", limit=5)
    assert env.priors_text() == before_b2b_on          # no team was on a b2b in the first 5 games
    update.run_update(rest="off")
    off = env.priors_text()
    env.use(tmp_path / "on")
    update.run_update(rest="on")
    assert env.priors_text() != off


def test_replay_with_rest_reads_only_the_cached_schedule(env, b2b_season, tmp_path, monkeypatch):
    update.run_update()                                    # caches the schedule
    monkeypatch.setattr(boxscores, "_download_schedule", lambda season: pytest.fail("no download expected"))
    out, log = tmp_path / "on.json", tmp_path / "on.csv"
    update.main(["--replay", "--from", str(PRE), "--out", str(out), "--log", str(log), "--rest", "on"])
    assert [r["game_id"] for r in predlog.read_log(log)] == b2b_season
    update.main(["--replay", "--from", str(PRE), "--out", str(tmp_path / "off.json"), "--log", str(tmp_path / "off.csv"),
                 "--rest", "off"])
    assert load(out) != load(tmp_path / "off.json")


# ---------------------------------------------------------------------------
# the real season: reproduction, and protected files
# ---------------------------------------------------------------------------

def caches(*folders):
    applied = load(REPO_ROOT / "applied_games.json")
    return all((REPO_ROOT / "stats" / folder / f"{gid}.json").exists() for folder in folders for gid in applied)


SEASON_CACHES = caches("boxscores", "boxscores_advanced") and (SCHEDULES / "2026-27.json").exists()


@pytest.mark.skipif(not SEASON_CACHES, reason="needs the box-score caches and the cached 2026-27 schedule")
def test_rest_off_reproduces_the_pace_home_model(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(PRE), "--out", str(out), "--log", str(log), "--n", "20", "--m", "30",
                 "--rest", "off"])
    assert sha(log) == sha(REPO_ROOT / "log_pace-home.csv")
    assert load(out) == load(REPO_ROOT / "priors_2026-27_pace-home.json")
    assert repo_data_hashes() == before


@pytest.mark.skipif(not SEASON_CACHES, reason="needs the box-score caches and the cached 2026-27 schedule")
def test_rest_on_replay_reproduces_log_pace_home_rest(tmp_path):
    before = repo_data_hashes()
    out, log = tmp_path / "priors.json", tmp_path / "log.csv"
    update.main(["--replay", "--from", str(PRE_REST), "--out", str(out), "--log", str(log), "--n", "20", "--m", "30",
                 "--rest", "on"])
    assert sha(log) == sha(LOG_REST)
    assert load(out) == load(REPLAY_REST)
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (SCHEDULES / "2026-27.json").exists(), reason="needs the cached 2026-27 schedule")
def test_2026_27_back_to_backs():
    with open(LOG_REST, encoding="utf-8") as f:
        ids = [line.split(",")[1] for line in f.read().splitlines()[1:]]
    table = rest.rest_games(ids)
    assert len(ids) == 210
    assert sum(any(i["rest"] == "b2b" for i in table[g].values()) for g in ids) == 61
    assert sum(all(i["rest"] == "b2b" for i in table[g].values()) for g in ids) == 13


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_rest_training_reproduces_priors_pre_pace_home_rest(tmp_path):
    before = repo_data_hashes()
    out = tmp_path / "trained.json"
    dynamic_model.main(["--out", str(out), "--before", "2026-10-03", "--rest", "on"])
    trained = load(out)
    assert trained == load(PRE_REST)
    assert trained != load(PRE)
    assert all(trained[t]["pace"] == load(PRE)[t]["pace"] for t in trained)    # no pace rest effect
    assert repo_data_hashes() == before


@pytest.mark.skipif(not (dynamic_model.DEFAULT_CSV.exists() and TRAINING_SCHEDULES),
                    reason="needs the Kaggle CSV and the training seasons' schedules in stats/")
def test_fit_rest_reproduces_model_params_and_keeps_the_rest(tmp_path):
    before = repo_data_hashes()
    path = tmp_path / "model_params.json"
    path.write_bytes((REPO_ROOT / "model_params.json").read_bytes())
    fit_rest.main(["--params", str(path), "--overwrite"])
    new, old = load(path), load(REPO_ROOT / "model_params.json")
    assert {k: v for k, v in new.items() if k != "rest"} == {k: v for k, v in old.items() if k != "rest"}
    strip = lambda r: {**r, "fit": {k: v for k, v in r["fit"].items() if k != "fitted"}}   # noqa: E731
    assert strip(new["rest"]) == strip(old["rest"])
    assert repo_data_hashes() == before


@pytest.mark.parametrize("target", ["priors.json", "applied_games.json", "model_params.json",
                                    "priors_pre_2026-27_pace-home-rest.json", "priors_2026-27_pace-home-rest.json",
                                    "log_pace-home-rest.csv", "log_pace-home.csv", "priors_2026-27_pace-home.json"])
def test_rest_replay_never_overwrites_existing_files(env, tmp_path, target):
    update.run_update()
    before = repo_data_hashes()
    real = REPO_ROOT / target
    other = tmp_path / ("x.csv" if real.suffix == ".json" else "x.json")
    out, log = (real, other) if real.suffix == ".json" else (other, real)
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(PRE_REST), "--out", str(out), "--log", str(log), "--rest", "on"])
    assert repo_data_hashes() == before


@pytest.mark.parametrize("name", ["priors.json", "applied_games.json", "model_params.json", "priors_pre_2026-27.json",
                                  "priors_pre_2026-27_pace-home.json", "priors_pre_2026-27_pace-home-rest.json"])
def test_rest_training_refuses_protected_outputs(name):
    before = repo_data_hashes()
    with pytest.raises(SystemExit):
        dynamic_model.main(["--out", str(REPO_ROOT / name), "--before", "2026-10-03", "--rest", "on"])
    assert repo_data_hashes() == before
