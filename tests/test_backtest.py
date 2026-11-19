"""
Tests for the live-procedure backtest (code/backtest.py) on small synthetic seasons: the cutoffs never
reach 2026-27, the season's games are assembled like update.py's, the season reset happens, and every
prediction uses only earlier games (a later result never changes an earlier prediction).
No real data or network: NBA.com's neutral / rest lookups are replaced.
"""
import copy

import pandas as pd
import pytest

import backtest
import dynamic_model
import params
import predlog
import update
from fake_nba import ID
from predict import predict_game
from schedule import NBA_TEAMS
from state_manager import default_priors


def start_priors():
    """Flat priors for every NBA team, with paces (what the live recipe's output looks like)."""
    ids = list(NBA_TEAMS)
    priors = default_priors(ids)
    pp = params.pace_params()
    return dynamic_model.start_pace(priors, ids, pp["start_mean"], pp["start_std"])


def game(gid, day, home, away, home_pts, away_pts, efg_home=0.56, efg_away=0.50, rest_away="2+"):
    def side(abbr, pts, efg):
        return {"id": ID[abbr], "abbr": abbr, "pts": pts, "eFG": efg, "TOV": 0.13, "ORB": 0.26, "FTR": 0.21}
    return {"game_id": gid, "date": day, "home": side(home, home_pts, efg_home), "away": side(away, away_pts, efg_away),
            "possessions": 99.0, "neutral": False, "rest": {"home": "2+", "away": rest_away}}


GAMES = [game("0022400001", "2024-10-22", "BOS", "NYK", 120, 100),
         game("0022400002", "2024-10-23", "LAL", "GSW", 101, 110),
         game("0022400003", "2024-10-24", "BOS", "LAL", 99, 104, rest_away="b2b"),
         game("0022400004", "2024-10-25", "NYK", "BOS", 111, 108)]


def test_preseason_cutoff_never_reaches_2026_27():
    assert backtest.preseason_cutoff("2024-25") == "2024-10-03"
    assert backtest.preseason_cutoff("2025-26") == "2025-10-03"
    with pytest.raises(ValueError, match="test set"):
        backtest.preseason_cutoff("2026-27")


def test_first_prediction_is_from_the_reset_priors():
    start = start_priors()
    rows = backtest.run_season(start, GAMES)
    reset = update.reset_priors(start, update.DEFAULT_N)
    first = predict_game(ID["BOS"], ID["NYK"], reset, seed=predlog.PREDICT_SEED, sigma=params.sim_sigma(),
                         neutral=False, rest_home="2+", rest_away="2+")
    assert rows[0]["p_model"] == pytest.approx(first["home_win_pct"] / 100)
    assert rows[0]["proj_margin"] == pytest.approx(first["home_score"] - first["away_score"])
    assert (rows[0]["home_won"], rows[0]["margin"], rows[0]["total"], rows[0]["season"]) == (True, 20, 220, "2024-25")


def test_games_are_applied_after_their_prediction():
    """BOS beat NYK in game 1, so BOS's rating has moved by game 3 (it is predicted from updated priors)."""
    start = start_priors()
    rows = backtest.run_season(start, GAMES)
    unapplied = update.reset_priors(start, update.DEFAULT_N)
    stale = predict_game(ID["BOS"], ID["LAL"], unapplied, seed=predlog.PREDICT_SEED, sigma=params.sim_sigma(),
                         neutral=False, rest_home="2+", rest_away="b2b")
    assert rows[2]["p_model"] != pytest.approx(stale["home_win_pct"] / 100, abs=1e-6)


def test_a_later_result_never_changes_an_earlier_prediction():
    start = start_priors()
    base = backtest.run_season(start, GAMES)
    changed = copy.deepcopy(GAMES)
    changed[2]["home"]["eFG"], changed[2]["home"]["pts"] = 0.30, 70     # game 3 turns into a BOS blowout loss
    other = backtest.run_season(start, changed)
    assert [r["p_model"] for r in other[:3]] == [r["p_model"] for r in base[:3]]
    assert other[3]["p_model"] != base[3]["p_model"]                     # game 4 (BOS again) sees it


def test_start_priors_are_not_modified():
    start = start_priors()
    before = copy.deepcopy(start)
    backtest.run_season(start, GAMES)
    assert start == before


def test_drift_setting_changes_later_predictions_only():
    start = start_priors()
    m30, m5 = backtest.run_season(start, GAMES, m=30), backtest.run_season(start, GAMES, m=5)
    assert m30[0]["p_model"] == m5[0]["p_model"]                        # nothing applied yet
    assert m30[2]["p_model"] != m5[2]["p_model"]


# ---------------------------------------------------------------------------
# assembling a season's games from the Kaggle frame
# ---------------------------------------------------------------------------

def kaggle_row(gid, when, team, opponent, home, score, efg=0.52):
    return {"gameId": gid, "gameDateTimeEst": pd.Timestamp(when), "teamId": int(team), "opponentTeamId": int(opponent),
            "home": home, "teamScore": score, "effectiveFieldGoalPercentage": efg, "teamTurnoverPercentage": 0.13,
            "offensiveReboundPercentage": 0.25, "freeThrowAttemptRate": 0.2}


def test_season_games(monkeypatch):
    rows = []
    for gid, when, home, away, hs, as_ in ((22400002, "2024-11-02 19:30", "BOS", "NYK", 110, 104),
                                           (22400001, "2024-10-30 22:00", "LAL", "GSW", 99, 101),
                                           (22300050, "2023-11-02 19:30", "BOS", "NYK", 90, 95),    # other season
                                           (22400003, "2024-11-03 19:30", "MIA", "ORL", 100, 98)):  # no possessions
        rows += [kaggle_row(gid, when, ID[home], ID[away], 1, hs), kaggle_row(gid, when, ID[away], ID[home], 0, as_)]
    rows.append(kaggle_row(22400004, "2024-11-04 19:30", ID["DAL"], ID["DEN"], 1, 100))            # one side only
    rows += [kaggle_row(32400001, "2025-02-16 20:00", 1610616833, 1610616834, 1, 120),             # All-Star teams
             kaggle_row(32400001, "2025-02-16 20:00", 1610616834, 1610616833, 0, 118)]
    data = {"cleaned": pd.DataFrame(rows),
            "possessions": [{"game_id": str(g), "possessions": p} for g, p in
                            ((22400002, 98.5), (22400001, 101.0), (22300050, 97.0), (22400004, 99.0))]}
    looked_up = []

    def neutral_games(ids):
        looked_up.append(list(ids))
        return {i: i == "0022400002" for i in ids}

    def rest_games(ids):
        teams = {"0022400001": (ID["LAL"], ID["GSW"]), "0022400002": (ID["BOS"], ID["NYK"])}
        return {i: {teams[i][0]: {"rest": "b2b"}, teams[i][1]: {"rest": "1"}} for i in ids}

    monkeypatch.setattr(backtest.boxscores, "neutral_games", neutral_games)
    monkeypatch.setattr(backtest.rest_days, "rest_games", rest_games)
    games, skipped = backtest.season_games(data, "2024-25")

    assert [g["game_id"] for g in games] == ["0022400001", "0022400002"]          # oldest first
    assert skipped == ["0022400003", "0022400004"]                                 # All-Star game isn't NBA vs NBA
    lal = games[0]
    assert (lal["date"], lal["home"]["abbr"], lal["away"]["abbr"], lal["home"]["pts"]) == ("2024-10-30", "LAL", "GSW", 99)
    assert lal["winner"] == ID["GSW"] and lal["possessions"] == 101.0
    assert lal["rest"] == {"home": "b2b", "away": "1"} and lal["neutral"] is False
    assert games[1]["neutral"] is True
    assert set(lal["home"]) == {"id", "abbr", "pts", "eFG", "TOV", "ORB", "FTR"}
    assert sorted(looked_up[0]) == ["0022400001", "0022400002"]


def test_cli_refuses_2026_27(capsys):
    with pytest.raises(SystemExit):
        backtest.main(["--seasons", "2026-27"])
    assert "test set" in capsys.readouterr().err
