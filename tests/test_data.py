"""
Tests for the 2026-27 update run: which games are used (boxscores.py) and how they are
applied and recorded (update.run_update).

Run from the repo root:  python -m pytest tests

Uses the fake NBA.com and fake fit from tests/fake_nba.py (the `env` fixture is in
conftest.py). Everything is written to a temp folder; the real priors.json,
applied_games.json and stats/ are never touched.
"""
import json

import pytest

from fake_nba import EXPECTED, FAKE_SEASON, boxscores, reference_priors, update


# ---------------------------------------------------------------------------

def test_no_preseason_or_pre_opening_night_games(env):
    update.run_update()
    applied = env.applied()
    assert "Pre Season" not in env.nba.log_requests
    assert set(env.nba.log_requests) == {"Regular Season", "IST"}
    assert not [gid for gid in applied if gid.startswith("001")]
    assert all(entry["date"] >= boxscores.OPENING_NIGHT for entry in applied.values())
    assert "0022600900" not in applied   # 002 game dated before opening night


def test_nba_cup_games_included(env):
    update.run_update()
    applied = env.applied()
    assert "0022600015" in applied and applied["0022600015"]["nba_cup"] is True   # group game
    assert "0062600001" in applied and applied["0062600001"]["nba_cup"] is True   # Cup final (only in IST)
    assert applied["0022600001"]["nba_cup"] is False


def test_game_missing_one_team_is_skipped_and_not_recorded(env):
    env.nba.broken.add("0022600002")
    result = update.run_update()
    assert [gid for gid, _ in result["skipped"]] == ["0022600002"]
    assert "0022600002" not in env.applied()
    assert "0022600002" not in env.applied_order
    assert not (boxscores.CACHE_DIR / "0022600002.json").exists()   # incomplete box scores aren't cached
    assert [g for g in EXPECTED if g != "0022600002"] == list(env.applied())

    env.nba.broken.clear()   # NBA.com fills it in later: the next run picks it up
    assert update.run_update()["applied"] == ["0022600002"]


def test_second_run_applies_zero_games(env):
    first = update.run_update()
    priors_after_first = env.priors_text()
    second = update.run_update()
    assert first["applied"] == EXPECTED
    assert second["applied"] == [] and second["already"] == len(EXPECTED)
    assert env.priors_text() == priors_after_first


def test_no_duplicate_ids(env):
    update.run_update()
    update.run_update()

    def no_duplicate_keys(pairs):
        keys = [k for k, _ in pairs]
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
        return dict(pairs)

    applied = json.loads(update.APPLIED_GAMES_PATH.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    assert len(applied) == len(EXPECTED)
    assert len(env.applied_order) == len(set(env.applied_order))   # the Cup game is in two lists, applied once


def test_games_applied_in_date_order(env):
    update.run_update()
    assert env.applied_order == EXPECTED
    dates = [env.applied()[gid]["date"] for gid in env.applied_order]
    assert dates == sorted(dates)
    assert list(env.applied()) == env.applied_order


def test_through_and_limit(env):
    update.main(["--update", "--through", "2026-10-21"])
    assert list(env.applied()) == EXPECTED[:3]
    update.main(["--update", "--limit", "2"])
    assert list(env.applied()) == EXPECTED[:5]
    with pytest.raises(SystemExit):
        update.main(["--update", "--through", "Nov 1"])


@pytest.mark.parametrize("k", [1, 3])
def test_interrupted_run_resumes_at_next_game(env, tmp_path, monkeypatch, k):
    full = reference_priors(env, tmp_path)
    first_k = reference_priors(env, tmp_path / "first_k", limit=k)

    calls = {"n": 0}
    spy = update.apply_game

    def crash_after_k(priors, game, m=update.DEFAULT_M, random_seed=None):
        if calls["n"] == k:
            raise RuntimeError(f"simulated crash after game {k}")
        calls["n"] += 1
        return spy(priors, game, m, random_seed)

    monkeypatch.setattr(update, "apply_game", crash_after_k)
    with pytest.raises(RuntimeError):
        update.run_update()
    assert list(env.applied()) == EXPECTED[:k]
    assert env.priors_text() == first_k          # saved progress = exactly the first k games

    monkeypatch.setattr(update, "apply_game", spy)
    env.applied_order.clear()
    resumed = update.run_update()
    assert resumed["applied"] == EXPECTED[k:]    # starts at game k+1
    assert env.applied_order == EXPECTED[k:]     # nothing from the first run applied again
    assert list(env.applied()) == EXPECTED
    assert env.priors_text() == full             # same as never crashing


@pytest.mark.parametrize("crash_on_write", [8, 9], ids=["before-priors-saved", "after-priors-saved"])
def test_crash_mid_save_never_half_applies(env, tmp_path, monkeypatch, crash_on_write):
    """
    Each game is saved with 3 writes (applied: pending, priors.json, applied: done). Crash on
    game 3's 2nd write (priors.json not saved) or 3rd write (saved, still marked pending).
    """
    full = reference_priors(env, tmp_path)
    real_write = update.write_json
    writes = {"n": 0}

    def flaky_write(path, data):
        writes["n"] += 1
        if writes["n"] == crash_on_write:
            raise OSError("simulated crash while saving")
        return real_write(path, data)

    monkeypatch.setattr(update, "write_json", flaky_write)
    with pytest.raises(OSError):
        update.run_update()
    assert "pending" in env.applied()[EXPECTED[2]]

    monkeypatch.setattr(update, "write_json", real_write)
    env.applied_order.clear()
    update.run_update()
    if crash_on_write == 8:      # priors.json never got game 3: it is applied again, once
        assert env.applied_order == EXPECTED[2:]
    else:                        # priors.json already had game 3: it is not applied again
        assert env.applied_order == EXPECTED[3:]
    assert list(env.applied()) == EXPECTED
    assert not any("pending" in entry for entry in env.applied().values())
    assert env.priors_text() == full


def test_cached_box_scores_used_without_network(env):
    update.run_update()
    assert sorted(p.stem for p in boxscores.CACHE_DIR.glob("*.json")) == sorted(EXPECTED)
    downloads = len(env.nba.box_requests)

    def no_network(game_id):
        raise AssertionError(f"box score {game_id} was downloaded instead of read from the cache")

    env.mp.setattr(boxscores, "_download_box_score", no_network)
    update.start_season()                        # same cache folder, fresh priors
    assert update.run_update()["applied"] == EXPECTED
    assert len(env.nba.box_requests) == downloads


def test_failed_requests_retried_with_backoff(env):
    env.nba.failures["0022600003"] = 2           # fails twice, then works
    update.run_update()
    assert "0022600003" in env.applied()
    assert env.nba.box_requests.count("0022600003") == 3
    assert env.sleeps == [2.0, 4.0]


# ---------------------------------------------------------------------------
# game log + schedule
# ---------------------------------------------------------------------------

def test_lists_agree_no_warnings(env):
    """Default fake: both lists agree. The schedule's preseason and not-yet-played games are ignored."""
    result = update.run_update()
    assert env.nba.schedule_requests == 1
    assert result["disagreements"] == []
    assert list(env.applied()) == EXPECTED
    assert "0022600300" not in env.applied() and "0012600050" not in env.applied()


@pytest.mark.parametrize("gid, cup", [("0022600003", False), ("0022600015", True), ("0062600001", True)],
                         ids=["regular", "cup-group", "cup-final"])
def test_game_missing_from_log_is_taken_from_schedule(env, gid, cup):
    """The DEN @ IND case: a final game the game log leaves out is still applied, in date order, with a warning."""
    env.nba.log_missing.add(gid)
    result = update.run_update()
    assert env.applied_order == EXPECTED
    entry = env.applied()[gid]
    assert entry["nba_cup"] is cup
    spec = next(g for g in FAKE_SEASON if g[0] == gid)
    assert entry["matchup"] == f"{spec[2]} @ {spec[3]}" and entry["score"] == f"{spec[4]}-{spec[5]}"
    assert len(result["disagreements"]) == 1
    assert gid in result["disagreements"][0] and "missing from the game log" in result["disagreements"][0]


def test_game_missing_from_schedule_is_still_used_and_warned(env):
    env.nba.sched_missing.add("0022600015")
    result = update.run_update()
    assert env.applied_order == EXPECTED
    assert env.applied()["0022600015"]["nba_cup"] is True   # from the IST game log
    assert len(result["disagreements"]) == 1
    assert "0022600015" in result["disagreements"][0] and "not final in the schedule" in result["disagreements"][0]


def test_detail_mismatch_is_warned_and_game_log_wins(env):
    env.nba.sched_scores["0022600001"] = (113, 120)   # game log says 113-121
    result = update.run_update()
    assert env.applied()["0022600001"]["score"] == "113-121"
    assert len(result["disagreements"]) == 1
    assert "0022600001" in result["disagreements"][0] and "score 113-121 vs 113-120" in result["disagreements"][0]
    assert env.applied_order == EXPECTED


def test_missing_game_found_later_is_applied_once(env):
    """Run 1: the log drops a game AND the schedule hasn't caught up. Run 2: the schedule has it -> applied once."""
    env.nba.log_missing.add("0022600003")
    env.nba.sched_missing.add("0022600003")
    update.run_update()
    assert "0022600003" not in env.applied()
    env.nba.sched_missing.clear()
    second = update.run_update()
    assert second["applied"] == ["0022600003"]
    assert env.applied_order.count("0022600003") == 1


def test_repeated_failures_skip_game_without_recording(env):
    env.nba.failures["0022600003"] = -1          # never works
    result = update.run_update()
    assert [gid for gid, _ in result["skipped"]] == ["0022600003"]
    assert env.nba.box_requests.count("0022600003") == boxscores.RETRIES
    assert env.sleeps == [2.0, 4.0, 8.0]
    assert "0022600003" not in env.applied()
    assert list(env.applied()) == [g for g in EXPECTED if g != "0022600003"]
