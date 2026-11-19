"""
Tests for the season reset in code/update.py.

Run from the repo root:  python -m pytest tests

priors.json and applied_games.json are redirected to a temp folder, so the real
ones are never touched. The real preseason files (the live one, --start-season's default
source, and the frozen snapshot) are read, never written.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

import update  # noqa: E402
from schedule import NBA_TEAMS  # noqa: E402

FROZEN = REPO_ROOT / "priors_pre_2026-27.json"
SOURCE = REPO_ROOT / "priors_pre_2026-27_pace-home-rest.json"   # --start-season's default source
PACE_HOME = REPO_ROOT / "priors_pre_2026-27_pace-home.json"       # the previous one, also read-only


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def all_curves(priors, team_ids):
    for team_id in team_ids:
        for side in update.SIDES:
            for factor in update.FACTORS:
                yield team_id, side, factor, priors[team_id][side][factor]


@pytest.fixture
def frozen():
    return load(FROZEN)


@pytest.fixture
def source():
    return load(SOURCE)


def test_default_source_is_the_live_preseason_file():
    assert update.START_PRIORS_PATH == SOURCE


@pytest.fixture
def out(tmp_path, monkeypatch):
    """Point priors.json / applied_games.json at a temp folder, pre-filled with junk."""
    priors = tmp_path / "priors.json"
    applied = tmp_path / "applied_games.json"
    priors.write_text('{"stale": true}', encoding="utf-8")
    applied.write_text(json.dumps({"401810001": {"date": "2026-10-21"}}), encoding="utf-8")
    monkeypatch.setattr(update, "PRIORS_PATH", priors)
    monkeypatch.setattr(update, "APPLIED_GAMES_PATH", applied)
    return priors, applied


def test_means_unchanged(out, source):
    update.start_season()
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["mean"] == source[team_id][side][factor]["mean"], (team_id, side, factor)
    for team_id in NBA_TEAMS:
        assert priors[team_id]["pace"]["mean"] == source[team_id]["pace"]["mean"]


def test_std_default_n20(out):
    update.start_season()
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["std"] == pytest.approx(0.1 / math.sqrt(20), abs=1e-9), (team_id, side, factor)
        assert curve["std"] == pytest.approx(0.02236, abs=1e-5)


def test_non_nba_teams_identical_to_the_source(out, source):
    update.start_season()
    priors = load(out[0])
    non_nba = [team_id for team_id in source if team_id not in NBA_TEAMS]
    assert len(non_nba) == 7
    for team_id in non_nba:
        assert priors[team_id] == source[team_id], team_id
    assert set(priors) == set(source)


def test_reset_from_the_frozen_snapshot(out, frozen):
    """--from the frozen snapshot (no team paces) needs --pace off; the result keeps its means."""
    before = (out[0].read_bytes(), out[1].read_bytes())
    with pytest.raises(ValueError, match="pace"):
        update.start_season(from_path=FROZEN)
    with pytest.raises(SystemExit):
        update.main(["--start-season", "--from", str(FROZEN)])
    assert (out[0].read_bytes(), out[1].read_bytes()) == before      # refused before writing anything
    update.main(["--start-season", "--from", str(FROZEN), "--pace", "off"])
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["mean"] == frozen[team_id][side][factor]["mean"]
        assert curve["std"] == pytest.approx(0.1 / math.sqrt(20))
    assert load(out[1]) == {}


def test_source_files_hash_unchanged(out):
    before = (sha256(FROZEN), sha256(SOURCE), sha256(PACE_HOME))
    update.start_season()
    update.main(["--start-season", "--n", "10"])
    update.main(["--start-season", "--from", str(FROZEN), "--pace", "off"])
    update.main(["--start-season", "--from", str(PACE_HOME)])
    assert (sha256(FROZEN), sha256(SOURCE), sha256(PACE_HOME)) == before


@pytest.mark.parametrize("path", [FROZEN, SOURCE, PACE_HOME])
def test_writing_a_preseason_file_is_refused(path):
    before = sha256(path)
    with pytest.raises(PermissionError):
        update.write_json(path, {})
    assert sha256(path) == before


def test_from_needs_replay_or_start_season(out):
    with pytest.raises(SystemExit):
        update.main(["--update", "--from", str(SOURCE)])


def test_cli_n10(out):
    update.main(["--start-season", "--n", "10"])
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["std"] == pytest.approx(0.1 / math.sqrt(10), abs=1e-9), (team_id, side, factor)
        assert curve["std"] == pytest.approx(0.03162, abs=1e-5)


def test_cli_rejects_bad_n(out):
    with pytest.raises(SystemExit):
        update.main(["--start-season", "--n", "0"])


def test_reset_twice_matches_once(out):
    priors_path, applied_path = out
    update.start_season()
    once = (priors_path.read_bytes(), applied_path.read_bytes())
    update.start_season()
    twice = (priors_path.read_bytes(), applied_path.read_bytes())
    assert once == twice


def test_applied_games_empty_after(out):
    assert load(out[1]) != {}  # fixture pre-filled it
    update.start_season()
    assert load(out[1]) == {}
