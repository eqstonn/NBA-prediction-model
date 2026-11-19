"""
Tests for the shared per-game update (code/bayes_update.py, exact and pymc engines),
training (code/dynamic_model.py) and the season updater's apply_game.

Run from the repo root:  python -m pytest tests

test_exact_and_pymc_agree_on_real_games runs 10 real PyMC fits (~25 s); everything
else uses the exact engine and is instant. tests/fixtures/games_2026-27_first5.json is
the first 5 games of 2026-27 as boxscores returned them, so the tests run offline.
"""
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import bayes_update  # noqa: E402  (sets PYTENSOR_FLAGS; pymc is only imported for the pymc engine)
import dynamic_model  # noqa: E402
import update  # noqa: E402
from state_manager import default_priors  # noqa: E402

FROZEN = REPO_ROOT / "priors_pre_2026-27.json"
GAMES = json.loads((REPO_ROOT / "tests" / "fixtures" / "games_2026-27_first5.json").read_text(encoding="utf-8"))
FACTORS = bayes_update.FACTORS
SIDES = ("offense", "defense")
SEED = 2026


def frozen_priors():
    return json.loads(FROZEN.read_text(encoding="utf-8"))


@pytest.fixture
def season_priors():
    """Start-of-season priors, as `update.py --start-season` builds them (N=20)."""
    return update.reset_priors(frozen_priors())


def curves(team_ids):
    for team_id in team_ids:
        for side in SIDES:
            for f in FACTORS:
                yield team_id, side, f


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# the exact formula, by hand
# ---------------------------------------------------------------------------

def toy_priors():
    """Two teams, every factor N(mean, 0.02^2): HOME off 0.50 / def 0.56, AWAY off 0.52 / def 0.54."""
    def team(off, dfn):
        return {"offense": {f: {"mean": off, "std": 0.02} for f in FACTORS},
                "defense": {f: {"mean": dfn, "std": 0.02} for f in FACTORS}}
    return {"HOME": team(0.50, 0.56), "AWAY": team(0.52, 0.54)}


def test_exact_formula_hand_computed_one_game():
    """
    One game, worked by hand (sigma = 0.05, tau = 0, the same numbers for every factor):

    HOME offense N(0.50, 0.02^2) vs AWAY defense N(0.54, 0.02^2); HOME shoots 0.60
      V        = (0.0004 + 0.0004) / 4 + 0.05^2 = 0.0027
      surprise = 0.60 - (0.50 + 0.54) / 2      = +0.08
      gain     = (0.0004 / 2) / 0.0027         = 2/27
      HOME offense mean = 0.50 + 0.08 * 2/27  = 0.505926
      AWAY defense mean = 0.54 + 0.08 * 2/27  = 0.545926
      both stds = sqrt(0.0004 - 0.0002^2 / 0.0027) = sqrt(0.000385185) = 0.019626

    AWAY offense N(0.52, 0.02^2) vs HOME defense N(0.56, 0.02^2); AWAY shoots 0.50
      surprise = 0.50 - 0.54 = -0.04 -> means 0.52 - 0.04 * 2/27 = 0.517037 and 0.557037, stds 0.019626
    """
    priors = toy_priors()
    bayes_update.update_game(priors, {"team": "HOME", "opponent": "AWAY", **{f: 0.60 for f in FACTORS}}, engine="exact")
    bayes_update.update_game(priors, {"team": "AWAY", "opponent": "HOME", **{f: 0.50 for f in FACTORS}}, engine="exact")

    for f in FACTORS:
        for (team, side), mean in {("HOME", "offense"): 0.505926, ("AWAY", "defense"): 0.545926,
                                   ("AWAY", "offense"): 0.517037, ("HOME", "defense"): 0.557037}.items():
            curve = priors[team][side][f]
            assert curve["mean"] == pytest.approx(mean, abs=1e-6), (team, side, f)
            assert curve["std"] == pytest.approx(0.019626, abs=1e-6), (team, side, f)
        # the same values from the fractions, to full precision
        assert priors["HOME"]["offense"][f]["mean"] == pytest.approx(0.50 + 0.08 * 2 / 27, abs=1e-15)
        assert priors["AWAY"]["offense"][f]["mean"] == pytest.approx(0.52 - 0.04 * 2 / 27, abs=1e-15)
        assert priors["HOME"]["offense"][f]["std"] == pytest.approx(math.sqrt(0.0004 - 0.0002 ** 2 / 0.0027), abs=1e-15)


def test_exact_formula_hand_computed_with_drift():
    """
    Same first fit with tau = 0.01 (each curve widened before the fit):
      s^2 = 0.0004 + 0.0001 = 0.0005 for both curves
      V   = (0.0005 + 0.0005) / 4 + 0.0025 = 0.00275;  gain = 0.00025 / 0.00275 = 1/11
      HOME offense mean = 0.50 + 0.08 / 11 = 0.507273, AWAY defense mean = 0.547273
      stds = sqrt(0.0005 - 0.00025^2 / 0.00275) = sqrt(0.000477273) = 0.021847
    Curves not in the fit (HOME defense, AWAY offense) don't change.
    """
    priors = toy_priors()
    bayes_update.update_game(priors, {"team": "HOME", "opponent": "AWAY", **{f: 0.60 for f in FACTORS}},
                             tau=0.01, engine="exact")
    for f in FACTORS:
        assert priors["HOME"]["offense"][f]["mean"] == pytest.approx(0.507273, abs=1e-6)
        assert priors["AWAY"]["defense"][f]["mean"] == pytest.approx(0.547273, abs=1e-6)
        assert priors["HOME"]["offense"][f]["std"] == pytest.approx(0.021847, abs=1e-6)
        assert priors["AWAY"]["defense"][f]["std"] == pytest.approx(0.021847, abs=1e-6)
        assert priors["HOME"]["defense"][f] == {"mean": 0.56, "std": 0.02}
        assert priors["AWAY"]["offense"][f] == {"mean": 0.52, "std": 0.02}


# ---------------------------------------------------------------------------
# exact vs pymc
# ---------------------------------------------------------------------------

# One PyMC fit estimates a posterior mean/std from 2,000 draws, about 0.0005 off on average;
# over a whole season (Phase 4 replay) the median gap was 0.0012 and the 95th percentile 0.0036.
MEAN_TOL = 0.003
STD_TOL = 0.002


def test_exact_and_pymc_agree_on_real_games(season_priors):
    """The first 5 real games of 2026-27 through both engines: same answer within sampling tolerance."""
    exact, sampled = copy.deepcopy(season_priors), copy.deepcopy(season_priors)
    for i, game in enumerate(GAMES):
        update.apply_game(exact, game, engine="exact")
        update.apply_game(sampled, game, random_seed=SEED + 10 * i, engine="pymc")

    played = {g[side]["id"] for g in GAMES for side in ("home", "away")}
    mean_gap = max(abs(exact[t][s][f]["mean"] - sampled[t][s][f]["mean"]) for t, s, f in curves(played))
    std_gap = max(abs(exact[t][s][f]["std"] - sampled[t][s][f]["std"]) for t, s, f in curves(played))
    moved = max(abs(exact[t][s][f]["mean"] - season_priors[t][s][f]["mean"]) for t, s, f in curves(played))

    assert moved > 3 * MEAN_TOL, "the games should move some means well beyond the tolerance"
    assert mean_gap < MEAN_TOL, f"largest mean gap {mean_gap:.5f}"
    assert std_gap < STD_TOL, f"largest std gap {std_gap:.5f}"
    for team_id in season_priors:
        if team_id not in played:
            assert exact[team_id] == sampled[team_id] == season_priors[team_id]


def test_exact_is_the_default_and_never_loads_pymc():
    """apply_game with no engine argument uses the exact formula, without importing pymc at all."""
    code = (f"import sys, json; sys.path.insert(0, {str(CODE_DIR)!r}); import update; "
            f"g = json.load(open({str(REPO_ROOT / 'tests' / 'fixtures' / 'games_2026-27_first5.json')!r}, encoding='utf-8'))[0]; "
            f"p = update.reset_priors(json.load(open({str(FROZEN)!r}, encoding='utf-8'))); "
            "update.apply_game(p, g); print(update.DEFAULT_ENGINE, 'pymc' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["exact", "False"]


def test_unknown_engine_is_rejected_and_changes_nothing():
    priors = toy_priors()
    before = copy.deepcopy(priors)
    with pytest.raises(ValueError):
        bayes_update.update_game(priors, {"team": "HOME", "opponent": "AWAY", **{f: 0.6 for f in FACTORS}}, engine="nope")
    assert priors == before
    with pytest.raises(SystemExit):
        update.main(["--update", "--engine", "nope"])
    with pytest.raises(SystemExit):
        dynamic_model.main(["--out", "x.json", "--before", "2026-10-20", "--engine", "nope"])


# ---------------------------------------------------------------------------
# tau > 0: widening (fit stubbed out, so only the widening is left)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_fit(monkeypatch):
    """Replace the fit with posterior = prior, so all that's left is the widening."""
    calls = []

    def fake_fit(prior, observed, random_seed=None, engine=None):
        calls.append(prior)
        return prior

    monkeypatch.setattr(bayes_update, "_fit", fake_fit)
    return calls


@pytest.mark.parametrize("m", [30, 10])
def test_tau_widens_only_the_two_teams_exactly(no_fit, m):
    priors = frozen_priors()  # trained stds differ per curve, so the formula is really exercised
    before = copy.deepcopy(priors)
    game = GAMES[0]
    tau = 0.1 / m

    update.apply_game(priors, game, m=m)

    assert len(no_fit) == 2  # one fit per side
    in_game = {game["home"]["id"], game["away"]["id"]}
    for team_id in priors:
        if team_id not in in_game:
            assert priors[team_id] == before[team_id], team_id
    for team_id, side, f in curves(in_game):
        old, new = before[team_id][side][f], priors[team_id][side][f]
        assert new["mean"] == old["mean"], (team_id, side, f)
        assert new["std"] == math.sqrt(old["std"] ** 2 + tau ** 2), (team_id, side, f)
        assert new["std"] > old["std"]


def test_one_side_widens_only_its_own_curves(no_fit):
    """One update_game call fits team offense vs opponent defense; only those 8 curves widen."""
    priors = frozen_priors()
    before = copy.deepcopy(priors)
    home, away = GAMES[0]["home"], GAMES[0]["away"]
    bayes_update.update_game(priors, {"team": home["id"], "opponent": away["id"],
                                      **{f: home[f] for f in FACTORS}}, tau=0.01)

    assert priors[home["id"]]["defense"] == before[home["id"]]["defense"]
    assert priors[away["id"]]["offense"] == before[away["id"]]["offense"]
    for f in FACTORS:
        assert priors[home["id"]]["offense"][f]["std"] == math.sqrt(before[home["id"]]["offense"][f]["std"] ** 2 + 0.01 ** 2)
        assert priors[away["id"]]["defense"][f]["std"] == math.sqrt(before[away["id"]]["defense"][f]["std"] ** 2 + 0.01 ** 2)


def test_tau0_does_not_widen(no_fit):
    priors = frozen_priors()
    before = copy.deepcopy(priors)
    home, away = GAMES[0]["home"], GAMES[0]["away"]
    bayes_update.update_game(priors, {"team": home["id"], "opponent": away["id"],
                                      **{f: home[f] for f in FACTORS}}, tau=0)
    assert priors == before


# ---------------------------------------------------------------------------
# real games, default (exact) engine
# ---------------------------------------------------------------------------

def test_winner_offense_improves(season_priors):
    """
    DET beats BOS while beating its expected offense (shoots and rebounds better, gets to
    the line more, turns it over less): DET's eFG/ORB/FTR offense means move up and its
    TOV mean (lower = better) moves down. Nobody outside the game changes.
    """
    priors = season_priors
    win, lose = "1610612765", "1610612738"
    expected = {f: (priors[win]["offense"][f]["mean"] + priors[lose]["defense"][f]["mean"]) / 2 for f in FACTORS}
    as_expected_for_loser = {f: (priors[lose]["offense"][f]["mean"] + priors[win]["defense"][f]["mean"]) / 2
                             for f in FACTORS}
    better = {"eFG": +0.06, "TOV": -0.04, "ORB": +0.06, "FTR": +0.06}
    game = {"game_id": "test", "date": "2026-11-01", "winner": win,
            "home": {"id": win, **{f: expected[f] + better[f] for f in FACTORS}},
            "away": {"id": lose, **as_expected_for_loser}}
    before = copy.deepcopy(priors)

    update.apply_game(priors, game)

    for f in ("eFG", "ORB", "FTR"):
        assert priors[win]["offense"][f]["mean"] > before[win]["offense"][f]["mean"], f
    assert priors[win]["offense"]["TOV"]["mean"] < before[win]["offense"]["TOV"]["mean"]
    for team_id in priors:
        if team_id not in (win, lose):
            assert priors[team_id] == before[team_id], team_id


MAX_GAIN = 0.1   # a game may move a mean by at most 10% of its surprise


def test_per_game_mean_shifts_damped(season_priors):
    """
    The first 5 real games of 2026-27: every mean moves by at most 0.1 x |observed - expected|,
    where expected = (offense + defense) / 2 from the curves before the game. Each fit moves
    the team's offense and the opponent's defense, both driven by the same surprise.
    """
    priors = season_priors
    too_big = []
    for game in GAMES:
        before = copy.deepcopy(priors)
        update.apply_game(priors, game)
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = game[side], game[other]
            for f in FACTORS:
                expected = (before[team["id"]]["offense"][f]["mean"] + before[opp["id"]]["defense"][f]["mean"]) / 2
                surprise = team[f] - expected
                for team_id, curve, who in ((team["id"], "offense", team["abbr"]), (opp["id"], "defense", opp["abbr"])):
                    shift = priors[team_id][curve][f]["mean"] - before[team_id][curve][f]["mean"]
                    if abs(shift) > MAX_GAIN * abs(surprise):
                        too_big.append(f"{game['date']} {who} {curve} {f}: shift {shift:+.4f}, surprise {surprise:+.4f}")
    assert not too_big, "mean shifts above 0.1 x |observed - expected|:\n" + "\n".join(too_big)


# ---------------------------------------------------------------------------
# training (dynamic_model.py)
# ---------------------------------------------------------------------------

def write_training_csv(path, dated_games):
    """Kaggle-style rows (one per team per game) for (date, game) pairs; game_type None -> dropped."""
    rows = []
    for day, g, game_type in dated_games:
        for side, other in (("home", "away"), ("away", "home")):
            rows.append({
                "gameId": g["game_id"], "gameType": game_type, "gameDateTimeEst": f"{day} 19:30:00",
                "teamId": int(g[side]["id"]), "teamName": g[side]["abbr"],
                "opponentTeamId": int(g[other]["id"]), "opponentTeamName": g[other]["abbr"],
                "effectiveFieldGoalPercentage": g[side]["eFG"], "teamTurnoverPercentage": g[side]["TOV"],
                "offensiveReboundPercentage": g[side]["ORB"], "freeThrowAttemptRate": g[side]["FTR"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


TRAINING_ROWS = [  # (date, game, gameType); window for --before 2026-10-20 --years 3 is 2023-10-20 .. 2026-10-19
    ("2023-01-15", GAMES[0], "Regular Season"),   # too old for a 3-year window
    ("2025-01-10", GAMES[1], "Regular Season"),   # in
    ("2025-04-20", GAMES[2], None),               # no gameType -> dropped (as in the original training)
    ("2026-03-01", GAMES[3], "Regular Season"),   # in
    ("2026-10-20", GAMES[4], "Regular Season"),   # opening night 2026-27 -> excluded
]


def expected_training(dated_games):
    """What training should produce: flat defaults for the window's teams, then each row through the exact update."""
    teams = sorted({g[side]["id"] for _, g, _ in dated_games for side in ("home", "away")})
    priors = default_priors(teams)
    for _, g, _ in dated_games:
        for side, other in (("home", "away"), ("away", "home")):
            bayes_update.update_game(priors, {"team": g[side]["id"], "opponent": g[other]["id"],
                                              **{f: g[side][f] for f in FACTORS}}, engine="exact")
    return priors


def assert_same_priors(actual, expected):
    """Same teams and curves (to 1e-12: the factors go through a CSV round trip)."""
    assert set(actual) == set(expected)
    for team_id, side, f in curves(expected):
        for key in ("mean", "std"):
            assert actual[team_id][side][f][key] == pytest.approx(expected[team_id][side][f][key], abs=1e-12), \
                (team_id, side, f, key)


def test_training_uses_only_the_window_before_the_cutoff(tmp_path):
    csv, out = tmp_path / "games.csv", tmp_path / "trained.json"
    write_training_csv(csv, TRAINING_ROWS)
    dynamic_model.main(["--csv", str(csv), "--out", str(out), "--before", "2026-10-20"])
    trained = json.loads(out.read_text(encoding="utf-8"))
    assert_same_priors(trained, expected_training([TRAINING_ROWS[1], TRAINING_ROWS[3]]))


def test_training_all_history_keeps_old_games_but_not_this_season(tmp_path):
    csv, out = tmp_path / "games.csv", tmp_path / "trained.json"
    write_training_csv(csv, TRAINING_ROWS)
    dynamic_model.main(["--csv", str(csv), "--out", str(out), "--before", "2026-10-20", "--all-history"])
    trained = json.loads(out.read_text(encoding="utf-8"))
    assert_same_priors(trained, expected_training([TRAINING_ROWS[0], TRAINING_ROWS[1], TRAINING_ROWS[3]]))


@pytest.mark.parametrize("name", ["priors_pre_2026-27.json", "priors.json", "applied_games.json"])
def test_training_refuses_to_overwrite_protected_files(tmp_path, name):
    csv = tmp_path / "games.csv"
    write_training_csv(csv, TRAINING_ROWS)
    target = REPO_ROOT / name
    before = sha(target)
    with pytest.raises(SystemExit):
        dynamic_model.main(["--csv", str(csv), "--out", str(target), "--before", "2026-10-20"])
    assert sha(target) == before


# ---------------------------------------------------------------------------
# PYTENSOR_FLAGS
# ---------------------------------------------------------------------------

SPY = r"""
import json, os, sys
seen = {}
class Spy:
    # records PYTENSOR_FLAGS at the moment pymc / pytensor are first imported
    def find_spec(self, name, path=None, target=None):
        if name in ("pymc", "pytensor") and name not in seen:
            seen[name] = os.environ.get("PYTENSOR_FLAGS")
        return None
sys.meta_path.insert(0, Spy())
sys.path.insert(0, sys.argv[1])
__import__(sys.argv[2])
loaded_on_import = sorted(seen)
import bayes_update
bayes_update._pymc()          # what the pymc engine does on its first fit
print(json.dumps({"loaded_on_import": loaded_on_import, "seen": seen}))
"""


@pytest.mark.parametrize("module", ["bayes_update", "dynamic_model", "update"])
def test_pytensor_flags_set_before_pymc_import(module):
    env = {k: v for k, v in os.environ.items() if k != "PYTENSOR_FLAGS"}
    result = subprocess.run([sys.executable, "-c", SPY, str(CODE_DIR), module], env=env, cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded_on_import"] == []                       # importing the code doesn't load pymc
    assert set(report["seen"]) == {"pymc", "pytensor"}, report
    assert all(flags and "cxx=" in flags for flags in report["seen"].values()), report


def test_flags_line_comes_before_every_pymc_import():
    for path in CODE_DIR.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        imports = [i for i, line in enumerate(lines) if re.match(r"\s*(import|from)\s+(pymc|pytensor)\b", line)]
        if not imports:
            continue
        sets_flags = [i for i, line in enumerate(lines) if re.search(r"os\.environ\[\s*['\"]PYTENSOR_FLAGS", line)]
        assert sets_flags and sets_flags[0] < imports[0], f"{path.name} imports pymc before setting PYTENSOR_FLAGS"
