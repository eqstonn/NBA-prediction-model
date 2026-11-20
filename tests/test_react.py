"""
Tests for the rating reactivity switch (Phase 10c): params.react_settings / update.season_settings, the --react
switch in the season reset, --update, --replay, the backtest and the site, the react calibration temperature,
and code/fit_react.py's metrics, grid, diagnosis helpers and command line.
Nothing here touches the real priors.json or applied_games.json (the env fixture uses a temp folder).
"""
import copy
import json
import math

import numpy as np
import pytest

import backtest
import build_site
import fit_calib
import fit_react
import params
import predict
import predlog
import update
from fake_nba import EXPECTED, ID, REPO_ROOT
from test_backtest import GAMES, start_priors

START = update.START_PRIORS_PATH
BOS, NYK = ID["BOS"], ID["NYK"]


def react():
    return params.react_settings()


def without_section(tmp_path, monkeypatch, name):
    data = json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))
    del data[name]
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(params, "PARAMS_PATH", path)


# ---------------------------------------------------------------------------
# the settings
# ---------------------------------------------------------------------------

def test_react_section_is_valid_and_records_its_fit():
    section = json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))["react"]
    s = react()
    assert s == {"n": section["n"], "m": section["m"]}
    assert (s["n"], s["m"]) != (update.DEFAULT_N, update.DEFAULT_M)
    fit = section["fit"]
    assert "2024-25 only" in fit["data"]
    assert (fit["best"]["n"], fit["best"]["m"]) == (s["n"], s["m"])
    assert min(g["brier"] for g in fit["grid"]) == fit["best"]["brier"]          # the pick is the grid's best
    assert fit["live"]["brier"] > fit["best"]["brier"]
    assert {(g["n"], g["m"]) for g in fit["grid"]} == {(n, m) for n in fit["grid_n"] for m in fit["grid_m"]}
    assert 1 <= section["calib_temperature"] <= 2 and section["calib_temperature_se"] > 0


@pytest.mark.parametrize("bad", [{"n": 0, "m": 24}, {"n": 2}, {"n": 2.5, "m": 24}, {"n": True, "m": 24}, None])
def test_bad_react_sections_are_refused(bad):
    with pytest.raises(ValueError):
        params.check_react(bad)


def test_season_settings():
    s = react()
    assert update.season_settings() == (update.DEFAULT_N, update.DEFAULT_M) == (20, 30)
    assert update.season_settings("off", 7, None) == (7, 30)
    assert update.season_settings("on") == (s["n"], s["m"])
    with pytest.raises(ValueError, match="--n / --m"):
        update.season_settings("on", n=5)
    with pytest.raises(ValueError):
        update.season_settings("maybe")


def test_missing_section_only_matters_when_on(tmp_path, monkeypatch):
    without_section(tmp_path, monkeypatch, "react")
    assert update.season_settings("off") == (20, 30)
    with pytest.raises(KeyError, match="fit_react.py"):
        update.season_settings("on")
    with pytest.raises(KeyError, match="fit_react.py"):
        params.calib_temperature("on")


def test_calib_temperature_depends_on_react():
    data = json.loads(params.PARAMS_PATH.read_text(encoding="utf-8"))
    assert params.calib_temperature() == params.calib_temperature("off") == data["calib"]["temperature"]
    assert params.calib_temperature("on") == data["react"]["calib_temperature"]
    assert update.calib_temperature("off", "on") is None
    assert update.calib_temperature("on", "on") == data["react"]["calib_temperature"]
    with pytest.raises(ValueError):
        update.calib_temperature("off", "maybe")


def test_simulation_takes_an_explicit_temperature():
    priors = update.reset_priors(json.loads(START.read_text(encoding="utf-8")))
    kw = {"seed": predlog.PREDICT_SEED, "rest_home": "2+", "rest_away": "2+"}
    off = predict.predict_game(BOS, NYK, priors, **kw)
    on = predict.predict_game(BOS, NYK, priors, calib="on", calib_temperature=1.1, **kw)
    assert on["home_win_pct"] == pytest.approx(100 * params.calibrate_win_prob(off["home_win_pct"] / 100, "on", 1.1))
    assert (on["home_score"], on["away_score"]) == (off["home_score"], off["away_score"])


# ---------------------------------------------------------------------------
# the switch in the season procedure
# ---------------------------------------------------------------------------

def test_start_season_with_react_uses_its_n(env):
    priors = update.start_season(react="on")
    nba = next(iter(update.NBA_TEAMS))
    assert priors[nba]["offense"]["eFG"]["std"] == pytest.approx(update.season_std(react()["n"]))
    assert priors[nba]["pace"]["std"] == pytest.approx(params.pace_season_std(react()["n"]))
    with pytest.raises(ValueError):
        update.start_season(5, react="on")


def test_update_with_react_uses_its_m_and_records_it(env, tmp_path):
    update.run_update(m=react()["m"])
    plain = env.priors_text()
    assert {e["react"] for e in env.applied().values()} == {"off"}
    env.use(tmp_path / "react")
    update.run_update(react="on")
    assert env.priors_text() == plain                         # --update only takes M from the switch
    assert {e["react"] for e in env.applied().values()} == {"on"} and list(env.applied()) == EXPECTED


def test_replay_with_react_equals_the_explicit_n_and_m(env, tmp_path):
    update.run_update()
    s = react()
    out_a, log_a = tmp_path / "a.json", tmp_path / "a.csv"
    out_b, log_b = tmp_path / "b.json", tmp_path / "b.csv"
    update.replay(START, out_a, log_a, react="on")
    update.replay(START, out_b, log_b, n=s["n"], m=s["m"])
    assert out_a.read_bytes() == out_b.read_bytes() and log_a.read_bytes() == log_b.read_bytes()
    out_c, log_c = tmp_path / "c.json", tmp_path / "c.csv"
    update.replay(START, out_c, log_c)
    assert out_c.read_bytes() != out_a.read_bytes()            # and it is not the live N, M


def test_cli_react(env, tmp_path):
    args = update.make_parser().parse_args([])
    assert (args.n, args.m, args.react) == (None, None, "off")
    with pytest.raises(SystemExit):
        update.main(["--update", "--react", "on", "--m", "10"])
    with pytest.raises(SystemExit):
        update.main(["--update", "--react", "maybe"])
    update.main(["--update", "--react", "on"])
    assert {e["react"] for e in env.applied().values()} == {"on"}
    out, log = tmp_path / "r.json", tmp_path / "r.csv"
    update.main(["--replay", "--react", "on", "--from", str(START), "--out", str(out), "--log", str(log)])
    assert out.exists() and log.exists()


def test_backtest_with_react_equals_the_explicit_n_and_m():
    start, s = start_priors(), react()
    on = backtest.run_season(start, GAMES, react="on")
    assert on == backtest.run_season(start, GAMES, n=s["n"], m=s["m"])
    assert on != backtest.run_season(start, GAMES)
    with pytest.raises(ValueError):
        backtest.run_season(start, GAMES, n=5, react="on")


def test_backtest_with_react_and_calib_uses_the_react_temperature():
    start = start_priors()
    raw = backtest.run_season(start, GAMES, react="on")
    cal = backtest.run_season(start, GAMES, react="on", calib="on")
    a = params.calib_temperature("on")
    for r, c in zip(raw, cal):
        assert c["p_model"] == pytest.approx(params.calibrate_win_prob(r["p_model"], "on", a))
        assert c["proj_margin"] == r["proj_margin"]


def test_backtest_cli_refuses_n_with_react(capsys):
    with pytest.raises(SystemExit):
        backtest.main(["--react", "on", "--n", "5"])
    assert "--n / --m" in capsys.readouterr().err


def test_site_with_react_and_calib_uses_the_react_temperature(monkeypatch):
    priors = update.reset_priors(json.loads(START.read_text(encoding="utf-8")))
    games = [{"date": "2026-11-20", "status": "scheduled", "home": {"id": BOS, "abbr": "BOS", "score": None},
              "away": {"id": NYK, "abbr": "NYK", "score": None}}]
    lookup = lambda team, day: {"rest": "2+"}   # noqa: E731
    seen = []

    def spy(*args, **kwargs):
        seen.append((kwargs["calib"], kwargs["calib_temperature"]))
        return predict.predict_game(*args, **kwargs)

    monkeypatch.setattr(build_site, "predict_game", spy)
    for calib, r in (("on", "off"), ("on", "on"), ("off", "on")):
        rows = build_site.build_predictions(copy.deepcopy(games), priors, rest_lookup=lookup, calib=calib, react=r)
        assert rows[0]["pick"] in ("BOS", "NYK")
    assert seen == [("on", params.calib_temperature("off")), ("on", params.calib_temperature("on")), ("off", None)]
    assert build_site.make_parser().parse_args([]).react == "off"


# ---------------------------------------------------------------------------
# fit_react.py
# ---------------------------------------------------------------------------

def test_margin_slope_and_paired_slope():
    proj = np.array([-6.0, -2, 0, 3, 5, 9])
    s = fit_react.margin_slope(proj, 1.2 * proj + 1)
    assert s["slope"] == pytest.approx(1.2) and s["se"] == pytest.approx(0, abs=1e-9)
    rng = np.random.default_rng(1)
    proj = rng.normal(0, 6, 400)
    margin = 1.1 * proj + rng.normal(0, 12, 400)
    d = fit_react.paired_slope(proj, proj, margin, n_boot=50)
    assert d["diff"] == 0 and d["se"] == 0
    d = fit_react.paired_slope(proj, proj / 1.1, margin, n_boot=200)   # projections 1.1x too compressed
    assert d["diff"] == pytest.approx(0.1 * fit_react.margin_slope(proj, margin)["slope"], rel=1e-9)
    assert d["se"] > 0
    with pytest.raises(ValueError):
        fit_react.margin_slope([1, 2], [1, 2])


def rows_for(proj, p, margin, ids=None):
    return [{"game_id": ids[i] if ids else str(i), "date": "2024-11-01", "home_won": m > 0, "margin": m,
             "p_model": q, "proj_margin": x} for i, (x, q, m) in enumerate(zip(proj, p, margin))]


def test_compare_pairs_by_game_and_scores_each_with_its_own_projection():
    margin = [10, -8, 6, -2, 4, 12]
    old = rows_for([5, -4, 3, -1, 2, 6], [0.6, 0.4, 0.55, 0.48, 0.52, 0.66], margin)
    new = rows_for([10, -8, 6, -2, 4, 12], [0.7, 0.3, 0.62, 0.45, 0.56, 0.75], margin)
    c = fit_react.compare(old, list(reversed(new)), 1.3, 1.2)
    assert c["n"] == 6
    assert c["new"]["margin_mae"] == 0 and c["old"]["margin_mae"] > 0
    assert c["new-old"]["margin_mae"]["diff"] == pytest.approx(-c["old"]["margin_mae"])
    assert c["new"]["slope"] == pytest.approx(1.0) and c["old"]["slope"] == pytest.approx(2.0)
    assert c["new-old"]["brier"]["diff"] == pytest.approx(c["new"]["brier"] - c["old"]["brier"])
    p_old_c = fit_calib.calibrated(np.array([r["p_model"] for r in old]), 1.3)
    assert c["old_calib"]["brier"] == pytest.approx(np.mean((p_old_c - (np.array(margin) > 0)) ** 2))
    text = fit_react.format_compare("x", c)
    assert "new-old" in text and "new_calib-old_calib" in text and "slope" in text
    bad = rows_for([1] * 6, [0.5] * 6, [1, 2, 3, 4, 5, 7])
    with pytest.raises(ValueError, match="final score"):
        fit_react.compare(old, bad)


def test_best_point_and_grid_table():
    grid = [{"n": n, "m": m, "brier": 0.21 + 0.001 * abs(n - 3) + 0.0005 * abs(m - 24), "log_loss": 0.6}
            for n in (2, 3, 5) for m in (18, 24)]
    assert (fit_react.best_point(grid)["n"], fit_react.best_point(grid)["m"]) == (3, 24)
    table = fit_react.format_grid(grid, "brier")
    assert "N \\ M" in table and "0.2100" in table


def test_run_variant_defaults_are_the_backtest():
    start = start_priors()
    base = backtest.run_season(start, GAMES)
    variant = fit_react.run_variant(start, GAMES, extra=True)
    for b, v in zip(base, variant):
        assert v["p_model"] == pytest.approx(b["p_model"]) and v["proj_margin"] == pytest.approx(b["proj_margin"])
    assert variant[0]["proj_pre"] == pytest.approx(variant[0]["proj_margin"])     # nothing applied yet
    assert variant[3]["proj_pre"] != pytest.approx(variant[3]["proj_margin"])
    assert variant[0]["net_home"] == pytest.approx(0, abs=1e-9)                   # flat priors: everyone average


def test_sigma_scale_is_the_same_as_rescaling_n_and_m():
    games = GAMES * 5
    assert fit_react.sigma_equivalence(start_priors(), games, c=0.8) < 1e-12
    assert fit_react.sigma_equivalence(start_priors(), games, c=1.3) < 1e-12


def test_stretched_preseason_means():
    priors = update.reset_priors(json.loads(START.read_text(encoding="utf-8")))
    assert fit_react.stretched(priors, 1) == priors
    avg = fit_react.league_average(priors)
    double = fit_react.stretched(priors, 2)
    t = BOS
    for f in params.FACTORS:
        d0 = priors[t]["offense"][f]["mean"] - avg["offense"][f]
        assert double[t]["offense"][f]["mean"] - avg["offense"][f] == pytest.approx(2 * d0)
    assert fit_react.league_average(double)["defense"]["eFG"] == pytest.approx(avg["defense"]["eFG"])


def test_by_month_and_by_rating():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        proj = rng.normal(0, 6)
        rows.append({"date": "2024-11-05" if i < 200 else "2025-03-05", "proj_margin": proj, "proj_pre": proj / 2,
                     "margin": proj * (1 if i < 200 else 1.5) + rng.normal(0, 3), "net_home": proj / 2,
                     "net_away": -proj / 2})
    months = fit_react.by_month(rows)
    assert [m["month"] for m in months] == ["2024-11", "2025-03"]
    assert months[0]["slope"] == pytest.approx(1, abs=0.1) and months[1]["slope"] == pytest.approx(1.5, abs=0.1)
    assert months[0]["pre_slope"] == pytest.approx(2, abs=0.2)
    r = fit_react.by_rating(rows)
    assert sum(b["n"] for b in r["bins"]) == 800 and r["per_point"] > 0
    assert fit_react.months_slope(rows, fit_react.LATE) == pytest.approx(1.5, abs=0.1)


def test_grid_and_fit_on_a_tiny_season(tmp_path):
    loaded = {"2024-25": (start_priors(), GAMES * 3)}
    grid = fit_react.run_grid(None, ns=(2, 20), ms=(24, 30), loaded=loaded)
    assert {(g["n"], g["m"]) for g in grid} == {(2, 24), (2, 30), (20, 24), (20, 30)}
    live = next(g for g in grid if (g["n"], g["m"]) == (20, 30))
    rows = backtest.run_season(*loaded["2024-25"])
    assert live["brier"] == pytest.approx(np.mean([(r["p_model"] - r["home_won"]) ** 2 for r in rows]))
    section = fit_react.fit(None, loaded=loaded, ns=(2, 20), ms=(24, 30))
    assert (section["n"], section["m"]) == (section["fit"]["best"]["n"], section["fit"]["best"]["m"])
    params.check_react(section)
    params.check_temperature(section["calib_temperature"])
    path = tmp_path / "model_params.json"
    path.write_text(json.dumps({"sigma": 1}), encoding="utf-8")
    fit_react.write_section(path, section)
    assert json.loads(path.read_text(encoding="utf-8")) == {"sigma": 1, "react": json.loads(json.dumps(section))}


def test_fit_cli_refuses_to_replace_without_overwrite(tmp_path, capsys):
    path = tmp_path / "model_params.json"
    path.write_text(params.PARAMS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit):
        fit_react.main(["--params", str(path)])
    assert "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        fit_react.main(["--old-log", "a.csv", "--new-log", "b.csv"])      # logs only with --evaluate
    with pytest.raises(SystemExit):
        fit_react.main(["--workers", "0"])


def test_replay_files_are_the_react_replay():
    """priors_2026-27_react.json / log_react.csv: the live replay with N, M from model_params.json["react"]."""
    out, log = REPO_ROOT / "priors_2026-27_react.json", REPO_ROOT / "log_react.csv"
    if not (out.exists() and log.exists()):
        pytest.skip("replay files not made yet")
    rows = list(predlog.read_log(log))
    applied = json.loads(update.APPLIED_GAMES_PATH.read_text(encoding="utf-8"))
    assert len(rows) == len(applied)
    live = {r["game_id"]: r for r in predlog.read_log(REPO_ROOT / "log_pace-home-rest.csv")}
    assert [r["game_id"] for r in rows] == list(live)                       # same games, same order
    for r in rows:
        assert (r["actual_home"], r["actual_away"]) == (live[r["game_id"]]["actual_home"],
                                                         live[r["game_id"]]["actual_away"])
    # the first game is predicted from the reset priors: the live run's means, only wider stds
    first = rows[0]
    assert abs(float(first["proj_home"]) - float(live[first["game_id"]]["proj_home"])) < 0.3
    assert json.loads(out.read_text(encoding="utf-8")) != json.loads(update.PRIORS_PATH.read_text(encoding="utf-8"))
    assert not math.isnan(float(rows[-1]["win_pct"]))
