"""
Tests for the season reset in code/update.py.

Run from the repo root:  python -m pytest tests

priors.json and applied_games.json are redirected to a temp folder, so the real
ones are never touched. The real frozen snapshot is read (never written).
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
def out(tmp_path, monkeypatch):
    """Point priors.json / applied_games.json at a temp folder, pre-filled with junk."""
    priors = tmp_path / "priors.json"
    applied = tmp_path / "applied_games.json"
    priors.write_text('{"stale": true}', encoding="utf-8")
    applied.write_text(json.dumps({"401810001": {"date": "2026-10-21"}}), encoding="utf-8")
    monkeypatch.setattr(update, "PRIORS_PATH", priors)
    monkeypatch.setattr(update, "APPLIED_GAMES_PATH", applied)
    return priors, applied


def test_means_unchanged(out, frozen):
    update.start_season()
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["mean"] == frozen[team_id][side][factor]["mean"], (team_id, side, factor)


def test_std_default_n20(out):
    update.start_season()
    priors = load(out[0])
    for team_id, side, factor, curve in all_curves(priors, NBA_TEAMS):
        assert curve["std"] == pytest.approx(0.1 / math.sqrt(20), abs=1e-9), (team_id, side, factor)
        assert curve["std"] == pytest.approx(0.02236, abs=1e-5)


def test_non_nba_teams_identical_to_frozen(out, frozen):
    update.start_season()
    priors = load(out[0])
    non_nba = [team_id for team_id in frozen if team_id not in NBA_TEAMS]
    assert len(non_nba) == 7
    for team_id in non_nba:
        assert priors[team_id] == frozen[team_id], team_id
    assert set(priors) == set(frozen)


def test_frozen_file_hash_unchanged(out):
    before = sha256(FROZEN)
    update.start_season()
    update.main(["--start-season", "--n", "10"])
    assert sha256(FROZEN) == before


def test_writing_frozen_file_is_refused(frozen):
    before = sha256(FROZEN)
    with pytest.raises(PermissionError):
        update.write_json(FROZEN, {})
    assert sha256(FROZEN) == before


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
