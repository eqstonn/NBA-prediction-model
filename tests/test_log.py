"""
Tests for the prediction log (code/predlog.py), its use in update.run_update, the replay
mode (update.py --replay) and code/compare.py.

Run from the repo root:  python -m pytest tests

Pipeline tests use the fake NBA.com and fake fit from tests/fake_nba.py (the `env`
fixture is in conftest.py); every file goes to a temp folder. Replays start from
update.START_PRIORS_PATH, the source --start-season (and so the env's season) uses.
"""
import hashlib
import json

import pytest

from fake_nba import EXPECTED, update

import compare  # noqa: E402  (code/ is on sys.path via fake_nba)
import predlog  # noqa: E402
from predict import predict_game  # noqa: E402

FROZEN = update.FROZEN_PRIORS_PATH
START = update.START_PRIORS_PATH


def row(gid, home, away, winner, pct, proj_home, proj_away, actual_home, actual_away):
    return {"date": "2026-10-20", "game_id": gid, "home": home, "away": away, "predicted_winner": winner,
            "win_pct": f"{pct:.2f}", "proj_home": f"{proj_home:.1f}", "proj_away": f"{proj_away:.1f}",
            "actual_home": str(actual_home), "actual_away": str(actual_away)}


# 3-game toy log. Home team won all three.
#   T1  BOS 70% to win, projected 110-100, actual 105-100  -> right, spread error |10 - 5|  = 5
#   T2  ORL 60% to win (away), projected 100-104, actual 101-99 -> wrong, spread error |-4 - 2| = 6
#   T3  LAL 55% to win, projected 108-106, actual 120-110  -> right, spread error |2 - 10| = 8
TOY = [
    row("T1", "BOS", "NYK", "BOS", 70, 110, 100, 105, 100),
    row("T2", "MIA", "ORL", "ORL", 60, 100, 104, 101, 99),
    row("T3", "LAL", "GSW", "LAL", 55, 108, 106, 120, 110),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# metrics + the CSV
# ---------------------------------------------------------------------------

def test_brier_hand_computed_on_3_game_toy_log(tmp_path):
    path = tmp_path / "toy.csv"
    predlog.append_rows(path, TOY)
    m = predlog.metrics(predlog.read_log(path))

    # Brier by hand: (0.70 - 1)^2 + (0.60 - 0)^2 + (0.55 - 1)^2 = 0.09 + 0.36 + 0.2025 = 0.6525; / 3 = 0.2175
    assert m["brier"] == pytest.approx(0.2175, abs=1e-12)
    # same thing in the home-win form: p_home = 0.70, 0.40, 0.55, home won every game
    assert m["brier"] == pytest.approx(((0.70 - 1) ** 2 + (0.40 - 1) ** 2 + (0.55 - 1) ** 2) / 3, abs=1e-12)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["confidence"] == pytest.approx((0.70 + 0.60 + 0.55) / 3)
    assert m["spread_mae"] == pytest.approx((5 + 6 + 8) / 3)
    assert m["games"] == 3
    # totals: projected 210, 204, 214; actual 205, 200, 230
    assert m["proj_total"] == pytest.approx((210 + 204 + 214) / 3)
    assert m["actual_total"] == pytest.approx((205 + 200 + 230) / 3)
    # margin errors (actual - projected): 5 - 10 = -5, 2 - (-4) = 6, 10 - 2 = 8 -> population SD
    errors = [-5, 6, 8]
    mean = sum(errors) / 3
    assert m["margin_sd"] == pytest.approx((sum((e - mean) ** 2 for e in errors) / 3) ** 0.5)
    # home margins: projected 10, -4, 2; actual 5, 2, 10
    assert m["proj_home_margin"] == pytest.approx((10 - 4 + 2) / 3)
    assert m["actual_home_margin"] == pytest.approx((5 + 2 + 10) / 3)


def test_metrics_of_empty_log():
    assert predlog.metrics([]) == {"games": 0, "accuracy": None, "confidence": None, "spread_mae": None, "brier": None,
                                   "proj_total": None, "actual_total": None, "margin_sd": None,
                                   "proj_home_margin": None, "actual_home_margin": None}


def test_append_rows_never_duplicates(tmp_path):
    path = tmp_path / "log.csv"
    assert predlog.append_rows(path, TOY) == 3
    assert predlog.append_rows(path, TOY) == 0
    assert predlog.append_rows(path, [{**TOY[0], "win_pct": "99.00"}]) == 0   # same game_id, different numbers
    assert predlog.append_rows(path, [TOY[0], TOY[0]] + [row("T4", "DEN", "IND", "DEN", 51, 101, 100, 114, 102)]) == 1
    rows = predlog.read_log(path)
    assert [r["game_id"] for r in rows] == ["T1", "T2", "T3", "T4"]
    assert rows[0]["win_pct"] == "70.00"
    assert list(rows[0]) == predlog.COLUMNS


def test_prediction_is_reproducible_with_fixed_seed():
    priors = update.reset_priors(json.loads(START.read_text(encoding="utf-8")))
    a = predict_game("1610612765", "1610612738", priors, seed=predlog.PREDICT_SEED)
    b = predict_game("1610612765", "1610612738", priors, seed=predlog.PREDICT_SEED)
    c = predict_game("1610612765", "1610612738", priors, seed=predlog.PREDICT_SEED + 1)
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# logging during --update
# ---------------------------------------------------------------------------

def test_update_logs_each_game_before_it_is_applied(env, monkeypatch):
    calls = []
    real_predict = update.predict_game

    def spy(home_id, away_id, priors, seed=None, sigma=None, **kwargs):
        on_disk = json.loads(update.PRIORS_PATH.read_text(encoding="utf-8"))
        calls.append({"same_as_saved": priors == on_disk, "applied": set(env.applied()), "seed": seed})
        return real_predict(home_id, away_id, priors, seed=seed, sigma=sigma, **kwargs)

    monkeypatch.setattr(update, "predict_game", spy)
    update.run_update()

    rows = predlog.read_log(update.LOG_PATH)
    assert [r["game_id"] for r in rows] == EXPECTED
    for call, gid in zip(calls, EXPECTED):
        assert call["same_as_saved"]          # predicted from the priors saved just before this game
        assert gid not in call["applied"]     # ...and before it was applied
        assert call["seed"] == predlog.PREDICT_SEED
    first = rows[0]
    assert (first["date"], first["home"], first["away"], first["actual_home"], first["actual_away"]) == \
           ("2026-10-20", "DET", "BOS", "121", "113")
    assert first["predicted_winner"] in ("DET", "BOS") and 50 <= float(first["win_pct"]) <= 100


def test_rerun_does_not_duplicate_rows(env):
    update.run_update()
    first = update.LOG_PATH.read_bytes()
    update.run_update()                       # nothing new
    assert update.LOG_PATH.read_bytes() == first
    update.start_season()                     # even if every game gets applied again
    update.run_update()
    assert [r["game_id"] for r in predlog.read_log(update.LOG_PATH)] == EXPECTED


def test_skipped_game_is_not_logged(env):
    env.nba.broken.add("0022600002")
    update.run_update()
    assert "0022600002" not in {r["game_id"] for r in predlog.read_log(update.LOG_PATH)}


def test_crash_after_logging_keeps_one_row_per_game(env, tmp_path, monkeypatch):
    update.run_update()
    reference_log = update.LOG_PATH.read_bytes()
    env.use(tmp_path / "crash")                # fresh folder, same fake season

    spy = update.apply_game

    def crash_on_third(priors, game, m=update.DEFAULT_M, random_seed=None, **kwargs):
        if game["game_id"] == EXPECTED[2]:
            raise RuntimeError("simulated crash after the prediction was logged")
        return spy(priors, game, m, random_seed, **kwargs)

    monkeypatch.setattr(update, "apply_game", crash_on_third)
    with pytest.raises(RuntimeError):
        update.run_update()
    assert [r["game_id"] for r in predlog.read_log(update.LOG_PATH)] == EXPECTED[:3]
    assert EXPECTED[2] not in env.applied()

    monkeypatch.setattr(update, "apply_game", spy)
    update.run_update()
    assert update.LOG_PATH.read_bytes() == reference_log   # same rows, none twice


def test_metrics_printed_after_run(env, capsys):
    update.run_update()
    out = capsys.readouterr().out
    assert "prediction_log.csv: 6 games: winner accuracy" in out and "spread MAE" in out and "Brier" in out


# ---------------------------------------------------------------------------
# --replay
# ---------------------------------------------------------------------------

def run_with_late_game(env):
    """Live run where 0022600003 (Oct 21) only shows up after the later games, like DEN @ IND."""
    env.nba.log_missing.add("0022600003")
    env.nba.sched_missing.add("0022600003")
    update.run_update()
    env.nba.log_missing.clear()
    env.nba.sched_missing.clear()
    update.run_update()
    assert list(env.applied()) == [g for g in EXPECTED if g != "0022600003"] + ["0022600003"]


def replay_args(tmp_path, name="base"):
    out, log = tmp_path / f"priors_{name}.json", tmp_path / f"log_{name}.csv"
    return ["--replay", "--from", str(START), "--out", str(out), "--log", str(log)], out, log


def test_replay_writes_only_out_and_log(env, tmp_path):
    run_with_late_game(env)
    watched = [update.PRIORS_PATH, update.APPLIED_GAMES_PATH, update.LOG_PATH, FROZEN, START]
    before = {p: sha(p) for p in watched}
    folder_before = sorted(p.name for p in update.PRIORS_PATH.parent.iterdir())

    args, out, log = replay_args(tmp_path)
    update.main(args)

    assert {p: sha(p) for p in watched} == before
    assert sorted(p.name for p in update.PRIORS_PATH.parent.iterdir()) == folder_before
    assert out.exists() and log.exists()


def test_replay_is_oldest_first_and_matches_a_clean_run(env, tmp_path):
    run_with_late_game(env)
    live_folder = update.PRIORS_PATH.parent

    env.use(tmp_path / "clean")                # the same season applied strictly in date order
    update.run_update()
    clean_priors = json.loads(update.PRIORS_PATH.read_text(encoding="utf-8"))
    clean_log = update.LOG_PATH.read_bytes()
    env.point_at(live_folder)

    env.applied_order.clear()
    args, out, log = replay_args(tmp_path)
    update.main(args)

    assert env.applied_order == EXPECTED                       # oldest first, not applied_games.json order
    assert json.loads(out.read_text(encoding="utf-8")) == clean_priors
    assert log.read_bytes() == clean_log
    assert json.loads(update.PRIORS_PATH.read_text(encoding="utf-8")) != clean_priors   # the live one differs


def test_replay_twice_gives_identical_files_and_no_duplicate_rows(env, tmp_path):
    update.run_update()
    args, out, log = replay_args(tmp_path)
    update.main(args)
    first = (out.read_bytes(), log.read_bytes())
    update.main(args + ["--overwrite"])
    assert (out.read_bytes(), log.read_bytes()) == first
    assert [r["game_id"] for r in predlog.read_log(log)] == EXPECTED


@pytest.mark.parametrize("existing", ["out", "log"])
def test_replay_never_overwrites_existing_results_without_flag(env, tmp_path, existing):
    update.run_update()
    args, out, log = replay_args(tmp_path)
    target = out if existing == "out" else log
    target.write_text("an earlier result", encoding="utf-8")
    with pytest.raises(SystemExit):
        update.main(args)
    assert target.read_text(encoding="utf-8") == "an earlier result"


def test_replay_needs_no_network(env, tmp_path, monkeypatch):
    update.run_update()                        # caches every box score

    def offline(*args):
        raise AssertionError("replay went to the network")

    for name in ("_download_game_log", "_download_schedule", "_download_box_score", "_download_advanced_box_score"):
        monkeypatch.setattr(update.boxscores, name, offline)
    args, out, log = replay_args(tmp_path)
    update.main(args)
    assert out.exists()


@pytest.mark.parametrize("target", ["out=priors", "out=applied", "log=applied", "log=live-log", "out=from", "out=log"])
def test_replay_refuses_to_overwrite_protected_files(env, tmp_path, target):
    update.run_update()
    paths = {"priors": update.PRIORS_PATH, "applied": update.APPLIED_GAMES_PATH,
             "live-log": update.LOG_PATH, "from": START}
    args, out, log = replay_args(tmp_path)
    flag, what = target.split("=")
    if what == "log":
        out = log
    else:
        out, log = (paths[what], log) if flag == "out" else (out, paths[what])
    before = {p: sha(p) for p in paths.values()}
    with pytest.raises(SystemExit):
        update.main(["--replay", "--from", str(START), "--out", str(out), "--log", str(log)])
    assert {p: sha(p) for p in paths.values()} == before


# ---------------------------------------------------------------------------
# compare.py
# ---------------------------------------------------------------------------

def test_compare_side_by_side(tmp_path, capsys):
    a, b = tmp_path / "log_a.csv", tmp_path / "log_b.csv"
    predlog.append_rows(a, TOY)
    better = [TOY[0], row("T2", "MIA", "ORL", "MIA", 60, 104, 100, 101, 99), TOY[2]]   # gets T2 right
    predlog.append_rows(b, better)
    compare.main([str(a), str(b)])
    out = capsys.readouterr().out
    lines = {line.split("  ")[0].strip(): line for line in out.splitlines() if line.strip()}
    assert "log_a" in out and "log_b" in out
    assert "66.7%" in lines["winner accuracy"] and "100.0%*" in lines["winner accuracy"]
    # B's Brier by hand: (0.7-1)^2 + (0.6-1)^2 + (0.55-1)^2 = 0.09 + 0.16 + 0.2025 = 0.4525; / 3 = 0.1508
    assert "0.2175" in lines["Brier score"] and "0.1508*" in lines["Brier score"]
    assert "games compared" in out


def test_compare_scores_only_common_games(tmp_path, capsys):
    a, b = tmp_path / "log_a.csv", tmp_path / "log_b.csv"
    predlog.append_rows(a, TOY)
    predlog.append_rows(b, TOY[:2])
    compare.main([str(a), str(b)])
    out = capsys.readouterr().out
    assert "Only the 2 games present in every log are scored." in out
    # over T1 + T2 only: Brier (0.09 + 0.36) / 2 = 0.2250 for both
    brier_line = next(line for line in out.splitlines() if line.startswith("Brier score"))
    assert brier_line.count("0.2250") == 2


def test_compare_needs_two_logs(tmp_path):
    a = tmp_path / "log_a.csv"
    predlog.append_rows(a, TOY)
    with pytest.raises(SystemExit):
        compare.main([str(a)])
