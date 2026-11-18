"""
Tests for the shared per-game update (code/bayes_update.py), training
(code/dynamic_model.py) and the season updater (update.apply_game).

Run from the repo root:  python -m pytest tests

These run real PyMC fits (a few seconds each), so this file takes about a minute.
tests/fixtures/games_2026-27_first5.json is the first 5 games of 2026-27 exactly as
boxscores.fetch_games returned them (NBA.com four factors), so the tests run offline.
"""
import copy
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

import bayes_update  # noqa: E402  (sets PYTENSOR_FLAGS, then imports pymc)
import dynamic_model  # noqa: E402
import update  # noqa: E402
import pymc as pm  # noqa: E402
from data_prep import create_data_frame, clean_data_frame, get_four_factors  # noqa: E402

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


def curves(priors, team_ids):
    for team_id in team_ids:
        for side in SIDES:
            for f in FACTORS:
                yield team_id, side, f


# ---------------------------------------------------------------------------
# tau = 0: training behaves exactly like before the refactor
# ---------------------------------------------------------------------------

def old_update_priors(file_path):
    """dynamic_model.update_priors as it was before the refactor, copied verbatim."""
    #load priors into a variable (dictionary)
    with open("priors.json", 'r', encoding="utf-8") as file:
        current_priors = json.load(file)

    #data frame with 4 factors for every game
    four_factors = get_four_factors(clean_data_frame(create_data_frame(file_path)))


    count = 0

    #loop through games
    for index, row in four_factors.iterrows():
        team = row["teamId"]
        opponent_team = row["opponentTeamId"]

        #grab actual stat of games.
        actual_eFG = row["effectiveFieldGoalPercentage"]
        actual_TOV = row["teamTurnoverPercentage"]
        actual_ORB = row["offensiveReboundPercentage"]
        actual_FTR = row["freeThrowAttemptRate"]

        #grab offensive expected stats (team A performance)
        o_eFG_mean = current_priors[f"{team}"]["offense"]["eFG"]["mean"]
        o_eFG_std = current_priors[f"{team}"]["offense"]["eFG"]["std"]

        o_TOV_mean = current_priors[f"{team}"]["offense"]["TOV"]["mean"]
        o_TOV_std = current_priors[f"{team}"]["offense"]["TOV"]["std"]

        o_ORB_mean = current_priors[f"{team}"]["offense"]["ORB"]["mean"]
        o_ORB_std = current_priors[f"{team}"]["offense"]["ORB"]["std"]

        o_FTR_mean = current_priors[f"{team}"]["offense"]["FTR"]["mean"]
        o_FTR_std = current_priors[f"{team}"]["offense"]["FTR"]["std"]

        #grab defensive expected stats (team B performance, what they allowed)
        d_eFG_mean = current_priors[f"{opponent_team}"]["defense"]["eFG"]["mean"]
        d_eFG_std = current_priors[f"{opponent_team}"]["defense"]["eFG"]["std"]

        d_TOV_mean = current_priors[f"{opponent_team}"]["defense"]["TOV"]["mean"]
        d_TOV_std = current_priors[f"{opponent_team}"]["defense"]["TOV"]["std"]

        d_ORB_mean = current_priors[f"{opponent_team}"]["defense"]["ORB"]["mean"]
        d_ORB_std = current_priors[f"{opponent_team}"]["defense"]["ORB"]["std"]

        d_FTR_mean = current_priors[f"{opponent_team}"]["defense"]["FTR"]["mean"]
        d_FTR_std = current_priors[f"{opponent_team}"]["defense"]["FTR"]["std"]

        with pm.Model() as model:
            #current baseline offensively
            o_eFG_skill = pm.Normal("o_eFG_skill", mu = o_eFG_mean, sigma = o_eFG_std)
            o_TOV_skill = pm.Normal("o_TOV_skill", mu = o_TOV_mean, sigma = o_TOV_std)
            o_ORB_skill = pm.Normal("o_ORB_skill", mu = o_ORB_mean, sigma = o_ORB_std)
            o_FTR_skill = pm.Normal("o_FTR_skill", mu = o_FTR_mean, sigma = o_FTR_std)

            #current baseline defensively
            d_eFG_skill = pm.Normal("d_eFG_skill", mu = d_eFG_mean, sigma = d_eFG_std)
            d_TOV_skill = pm.Normal("d_TOV_skill", mu = d_TOV_mean, sigma = d_TOV_std)
            d_ORB_skill = pm.Normal("d_ORB_skill", mu = d_ORB_mean, sigma = d_ORB_std)
            d_FTR_skill = pm.Normal("d_FTR_skill", mu = d_FTR_mean, sigma = d_FTR_std)

            #account for strength of schedule
            expected_eFG = (o_eFG_skill + d_eFG_skill) / 2
            expected_TOV = (o_TOV_skill + d_TOV_skill) / 2
            expected_ORB = (o_ORB_skill + d_ORB_skill) / 2
            expected_FTR = (o_FTR_skill + d_FTR_skill) / 2

            #give model baseline
            eFG_evidence = pm.Normal("eFG_evidence", mu = expected_eFG, sigma = 0.05, observed = actual_eFG)
            TOV_evidence = pm.Normal("TOV_evidence", mu = expected_TOV, sigma = 0.05, observed = actual_TOV)
            ORB_evidence = pm.Normal("ORB_evidence", mu = expected_ORB, sigma = 0.05, observed = actual_ORB)
            FTR_evidence = pm.Normal("FTR_evidence", mu = expected_FTR, sigma = 0.05, observed = actual_FTR)

            #simulate to find new baseline (new bell curve)
            trace = pm.sample(1000, progressbar = False, cores = 1)

        #update offensive priors
        current_priors[f"{team}"]["offense"]["eFG"]["mean"] = float(trace.posterior["o_eFG_skill"].mean())
        current_priors[f"{team}"]["offense"]["eFG"]["std"] = float(trace.posterior["o_eFG_skill"].std())

        current_priors[f"{team}"]["offense"]["TOV"]["mean"] = float(trace.posterior["o_TOV_skill"].mean())
        current_priors[f"{team}"]["offense"]["TOV"]["std"] = float(trace.posterior["o_TOV_skill"].std())

        current_priors[f"{team}"]["offense"]["ORB"]["mean"] = float(trace.posterior["o_ORB_skill"].mean())
        current_priors[f"{team}"]["offense"]["ORB"]["std"] = float(trace.posterior["o_ORB_skill"].std())

        current_priors[f"{team}"]["offense"]["FTR"]["mean"] = float(trace.posterior["o_FTR_skill"].mean())
        current_priors[f"{team}"]["offense"]["FTR"]["std"] = float(trace.posterior["o_FTR_skill"].std())

        #update defensive priors
        current_priors[f"{opponent_team}"]["defense"]["eFG"]["mean"] = float(trace.posterior["d_eFG_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["eFG"]["std"] = float(trace.posterior["d_eFG_skill"].std())

        current_priors[f"{opponent_team}"]["defense"]["TOV"]["mean"] = float(trace.posterior["d_TOV_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["TOV"]["std"] = float(trace.posterior["d_TOV_skill"].std())

        current_priors[f"{opponent_team}"]["defense"]["ORB"]["mean"] = float(trace.posterior["d_ORB_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["ORB"]["std"] = float(trace.posterior["d_ORB_skill"].std())

        current_priors[f"{opponent_team}"]["defense"]["FTR"]["mean"] = float(trace.posterior["d_FTR_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["FTR"]["std"] = float(trace.posterior["d_FTR_skill"].std())

        count += 1

        #update every 50
        if count % 50 == 0:
            with open("priors.json", "w", encoding="utf-8") as file:
                json.dump(current_priors, file, indent=4)
            print(f"Checkpoint: Saved game {count}...")

    #update priors json with new stats at very end
    with open("priors.json", "w", encoding="utf-8") as file:
        json.dump(current_priors, file, indent = 4)


def write_training_csv(path, games):
    """The fixture games as Kaggle-style rows (one per team per game)."""
    rows = []
    for g in games:
        for side, other in (("home", "away"), ("away", "home")):
            rows.append({
                "gameId": g["game_id"], "gameType": "Regular Season",
                "gameDateTimeEst": f"{g['date']} 19:30:00",
                "teamId": int(g[side]["id"]), "teamName": g[side]["abbr"],
                "opponentTeamId": int(g[other]["id"]), "opponentTeamName": g[other]["abbr"],
                "effectiveFieldGoalPercentage": g[side]["eFG"], "teamTurnoverPercentage": g[side]["TOV"],
                "offensiveReboundPercentage": g[side]["ORB"], "freeThrowAttemptRate": g[side]["FTR"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_training_tau0_matches_old_code(tmp_path, monkeypatch):
    """Same CSV, same starting priors, same seed: old and new training write identical priors.json."""
    real_sample = pm.sample
    monkeypatch.setattr(pm, "sample", lambda *a, **kw: real_sample(*a, **{**kw, "random_seed": SEED}))

    csv = tmp_path / "games.csv"
    write_training_csv(csv, GAMES[:1])
    start = json.dumps(frozen_priors(), indent=4)

    results = {}
    for name, train in (("old", old_update_priors), ("new", dynamic_model.update_priors)):
        workdir = tmp_path / name
        workdir.mkdir()
        (workdir / "priors.json").write_text(start, encoding="utf-8")
        monkeypatch.chdir(workdir)  # training uses CWD-relative priors.json
        train(str(csv))
        results[name] = (workdir / "priors.json").read_text(encoding="utf-8")

    assert dynamic_model.TRAINING_TAU == 0
    assert results["new"] != start, "the game should have moved the priors"
    assert results["new"] == results["old"]


# ---------------------------------------------------------------------------
# tau > 0: widening
# ---------------------------------------------------------------------------

@pytest.fixture
def no_fit(monkeypatch):
    """Replace the PyMC fit with posterior = prior, so all that's left is the widening."""
    calls = []

    def fake_fit(prior, observed, random_seed=None):
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
    for team_id, side, f in curves(priors, in_game):
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
# real fits
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

    update.apply_game(priors, game, random_seed=SEED)

    for f in ("eFG", "ORB", "FTR"):
        assert priors[win]["offense"][f]["mean"] > before[win]["offense"][f]["mean"], f
    assert priors[win]["offense"]["TOV"]["mean"] < before[win]["offense"]["TOV"]["mean"]
    for team_id in priors:
        if team_id not in (win, lose):
            assert priors[team_id] == before[team_id], team_id


MAX_GAIN = 0.1   # a game may move a mean by at most 10% of its surprise
MCMC_TOL = 0.002  # posterior means from 2,000 draws carry ~0.0005 of sampling noise


def test_per_game_mean_shifts_damped(season_priors):
    """
    The first 5 real games of 2026-27: every mean moves by at most
    0.1 x |observed - expected| (+ MCMC noise), where expected = (offense + defense) / 2
    from the curves before the game. Each fit moves the team's offense and the
    opponent's defense, both driven by the same surprise.
    """
    priors = season_priors
    too_big = []
    for i, game in enumerate(GAMES):
        before = copy.deepcopy(priors)
        update.apply_game(priors, game, random_seed=SEED + 10 * i)
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = game[side], game[other]
            for f in FACTORS:
                expected = (before[team["id"]]["offense"][f]["mean"] + before[opp["id"]]["defense"][f]["mean"]) / 2
                surprise = team[f] - expected
                limit = MAX_GAIN * abs(surprise) + MCMC_TOL
                for team_id, curve, who in ((team["id"], "offense", team["abbr"]), (opp["id"], "defense", opp["abbr"])):
                    shift = priors[team_id][curve][f]["mean"] - before[team_id][curve][f]["mean"]
                    if abs(shift) > limit:
                        too_big.append(f"{game['date']} {who} {curve} {f}: shift {shift:+.4f}, "
                                       f"surprise {surprise:+.4f}, limit {limit:.4f}")
    assert not too_big, "mean shifts above 0.1 x |observed - expected|:\n" + "\n".join(too_big)


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
print(json.dumps(seen))
"""


@pytest.mark.parametrize("module", ["bayes_update", "dynamic_model", "update"])
def test_pytensor_flags_set_before_pymc_import(module):
    env = {k: v for k, v in os.environ.items() if k != "PYTENSOR_FLAGS"}
    result = subprocess.run([sys.executable, "-c", SPY, str(CODE_DIR), module], env=env, cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(seen) == {"pymc", "pytensor"}, seen
    assert all(flags and "cxx=" in flags for flags in seen.values()), seen


def test_flags_line_comes_before_every_pymc_import():
    for path in CODE_DIR.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        imports = [i for i, line in enumerate(lines) if re.match(r"\s*(import|from)\s+(pymc|pytensor)\b", line)]
        if not imports:
            continue
        sets_flags = [i for i, line in enumerate(lines) if re.search(r"os\.environ\[\s*['\"]PYTENSOR_FLAGS", line)]
        assert sets_flags and sets_flags[0] < imports[0], f"{path.name} imports pymc before setting PYTENSOR_FLAGS"
