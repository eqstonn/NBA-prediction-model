"""
Shared test doubles for the update pipeline (used by test_data.py, test_log.py, test_sigma.py, test_sim.py).

NBA.com is replaced by FakeNBA (game logs, season schedule, box scores, advanced box scores) and the PyMC
fit by fake_fit, a fast deterministic stand-in whose result depends on the order games
are applied, so a game applied twice or out of order shows up as a different
priors.json. Env points every file the pipeline writes at a temp folder; the real
priors.json, applied_games.json, prediction_log.csv and stats/ are never touched.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

import bayes_update  # noqa: E402
import boxscores  # noqa: E402
import update  # noqa: E402
from schedule import NBA_TEAMS  # noqa: E402

FACTORS = bayes_update.FACTORS
ABBR = {team_id: abbr for team_id, (abbr, _) in NBA_TEAMS.items()}
ID = {abbr: team_id for team_id, abbr in ABBR.items()}

# (game_id, date, away, home, away_pts, home_pts, season types NBA.com lists it under)
FAKE_SEASON = [
    ("0022600020", "2026-11-01", "DET", "BOS", 110, 104, ["Regular Season"]),
    ("0022600002", "2026-10-20", "PHI", "NYK", 106, 100, ["Regular Season"]),
    ("0012600050", "2026-10-10", "BOS", "NYK", 99, 101, ["Pre Season"]),          # preseason
    ("0012600051", "2026-10-21", "LAL", "BOS", 90, 95, ["Regular Season"]),       # 001 id in the wrong list
    ("0022600900", "2026-10-15", "MIA", "ORL", 100, 98, ["Regular Season"]),      # before opening night
    ("0022600015", "2026-10-31", "LAL", "GSW", 105, 127, ["Regular Season", "IST"]),  # NBA Cup group game
    ("0022600001", "2026-10-20", "BOS", "DET", 113, 121, ["Regular Season"]),
    ("0062600001", "2026-12-16", "NYK", "OKC", 118, 112, ["IST"]),                # NBA Cup final
    ("0022600003", "2026-10-21", "OKC", "SAS", 116, 104, ["Regular Season"]),
]
# what should be applied, in order
EXPECTED = ["0022600001", "0022600002", "0022600003", "0022600015", "0022600020", "0062600001"]

LOG_HEADERS = ["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]


class FakeNBA:
    """
    Stands in for stats.nba.com.
      broken          games whose away factors are missing from the box score
      failures        {game_id: how many box-score requests fail before one works} (-1 = always)
      log_missing     games the game log leaves out (like DEN @ IND on 2026-11-17)
      sched_missing   games the schedule doesn't show as final
      sched_scores    {game_id: (away_pts, home_pts)} the schedule reports instead
    By default the schedule agrees with the game logs, and also lists a finished
    preseason game and a game that hasn't been played yet (both must be ignored).
    """

    def __init__(self):
        self.broken = set()
        self.failures = {}
        self.log_missing = set()
        self.sched_missing = set()
        self.sched_scores = {}
        self.no_possessions = set()   # games whose advanced box score lacks the home team's possessions
        self.log_requests = []
        self.schedule_requests = 0
        self.box_requests = []
        self.advanced_requests = []

    def schedule(self, season):
        self.schedule_requests += 1
        days = {}
        upcoming = ("0022600300", "2026-12-25", "BOS", "NYK", None, None, ["Regular Season"])
        for gid, day, away, home, a_pts, h_pts, types in FAKE_SEASON + [upcoming]:
            if gid in self.sched_missing:
                continue
            a_pts, h_pts = self.sched_scores.get(gid, (a_pts, h_pts))
            cup = "IST" in types
            days.setdefault(day, []).append({
                "gameId": gid, "gameDateEst": f"{day}T00:00:00Z",
                "gameStatus": 1 if a_pts is None else 3, "gameStatusText": "7:30 pm ET" if a_pts is None else "Final",
                "gameLabel": "Emirates NBA Cup" if cup else ("Preseason" if gid.startswith("001") else ""),
                "gameSubtype": ("in-season-knockout" if gid.startswith("006") else "in-season") if cup else "",
                "homeTeam": {"teamId": int(ID[home]), "teamTricode": home, "score": h_pts},
                "awayTeam": {"teamId": int(ID[away]), "teamTricode": away, "score": a_pts},
            })
        return {"leagueSchedule": {"seasonYear": season, "gameDates": [
            {"gameDate": f"{day} 00:00:00", "games": games} for day, games in sorted(days.items())]}}

    def game_log(self, season, season_type):
        self.log_requests.append(season_type)
        rows = []
        for gid, day, away, home, a_pts, h_pts, types in FAKE_SEASON:
            if season_type in types and gid not in self.log_missing:
                rows.append([int(ID[home]), home, gid, day, f"{home} vs. {away}", "W" if h_pts > a_pts else "L", h_pts])
                rows.append([int(ID[away]), away, gid, day, f"{away} @ {home}", "W" if a_pts > h_pts else "L", a_pts])
        return {"resultSets": [{"headers": LOG_HEADERS, "rowSet": rows}]}

    def box_score(self, game_id):
        self.box_requests.append(game_id)
        left = self.failures.get(game_id, 0)
        if left:
            self.failures[game_id] = left - 1 if left > 0 else left
            raise ConnectionError("simulated NBA.com timeout")
        _, _, away, home, *_ = next(g for g in FAKE_SEASON if g[0] == game_id)
        n = int(game_id[-3:])

        def stats(k, missing=False):
            s = {"effectiveFieldGoalPercentage": 0.48 + 0.01 * ((n + k) % 9),
                 "teamTurnoverPercentage": 0.12 + 0.005 * ((n + 2 * k) % 7),
                 "offensiveReboundPercentage": 0.24 + 0.01 * ((n + 3 * k) % 8),
                 "freeThrowAttemptRate": 0.20 + 0.01 * ((n + 5 * k) % 10)}
            if missing:
                s["effectiveFieldGoalPercentage"] = None
            return s

        return {"boxScoreFourFactors": {
            "gameId": game_id,
            "homeTeam": {"teamId": int(ID[home]), "teamTricode": home, "statistics": stats(1)},
            "awayTeam": {"teamId": int(ID[away]), "teamTricode": away,
                         "statistics": stats(2, missing=game_id in self.broken)},
        }}

    def advanced_box_score(self, game_id):
        """Each team's possessions: 95 + (game number % 11), the away team one fewer."""
        self.advanced_requests.append(game_id)
        _, _, away, home, *_ = next(g for g in FAKE_SEASON if g[0] == game_id)
        poss = 95.0 + int(game_id[-3:]) % 11
        return {"boxScoreAdvanced": {
            "gameId": game_id,
            "homeTeam": {"teamId": int(ID[home]), "teamTricode": home,
                         "statistics": {"possessions": None if game_id in self.no_possessions else poss}},
            "awayTeam": {"teamId": int(ID[away]), "teamTricode": away, "statistics": {"possessions": poss - 1}},
        }}

    @staticmethod
    def game_possessions(game_id):
        """The game's possessions the fake reports (average of the two teams)."""
        return 95.0 + int(game_id[-3:]) % 11 - 0.5


def fake_fit(prior, observed, random_seed=None, engine=None, sigma=None):
    """Deterministic stand-in for the PyMC fit: means move 10% toward the observation."""
    return {side: {f: (prior[side][f][0] + 0.1 * (observed[f] - prior[side][f][0]), prior[side][f][1] * 0.99)
                   for f in FACTORS}
            for side in ("o", "d")}


class Env:
    """Fake NBA.com + fake fit + every pipeline file in a temp folder."""

    def __init__(self, folder, monkeypatch):
        self.folder, self.mp = folder, monkeypatch
        self.nba = FakeNBA()
        self.applied_order = []
        self.sleeps = []
        real_apply = update.apply_game

        def spy_apply(priors, game, m=update.DEFAULT_M, random_seed=None, **kwargs):
            result = real_apply(priors, game, m, random_seed, **kwargs)
            self.applied_order.append(game["game_id"])
            return result

        monkeypatch.setattr(update, "apply_game", spy_apply)
        monkeypatch.setattr(bayes_update, "_fit", fake_fit)
        monkeypatch.setattr(boxscores, "_download_game_log", self.nba.game_log)
        monkeypatch.setattr(boxscores, "_download_schedule", self.nba.schedule)
        monkeypatch.setattr(boxscores, "_download_box_score", self.nba.box_score)
        monkeypatch.setattr(boxscores, "_download_advanced_box_score", self.nba.advanced_box_score)
        monkeypatch.setattr(boxscores, "_sleep", self.sleeps.append)
        monkeypatch.setattr(boxscores, "PAUSE", 0)
        self.use(folder)

    def point_at(self, folder):
        """Point priors.json / applied_games.json / prediction_log.csv / the box-score cache at `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        self.mp.setattr(update, "PRIORS_PATH", folder / "priors.json")
        self.mp.setattr(update, "APPLIED_GAMES_PATH", folder / "applied_games.json")
        self.mp.setattr(update, "LOG_PATH", folder / "prediction_log.csv")
        self.mp.setattr(boxscores, "CACHE_DIR", folder / "boxscores")
        self.mp.setattr(boxscores, "ADVANCED_CACHE_DIR", folder / "boxscores_advanced")

    def use(self, folder):
        """point_at(folder) and start the season there."""
        self.point_at(folder)
        update.start_season()

    def applied(self):
        return json.loads(update.APPLIED_GAMES_PATH.read_text(encoding="utf-8"))

    def priors_text(self):
        return update.PRIORS_PATH.read_text(encoding="utf-8")


def reference_priors(env, tmp_path, **kwargs):
    """priors.json after one clean, uninterrupted run in a separate folder (then switches back)."""
    current = update.PRIORS_PATH.parent
    env.use(tmp_path / "reference")
    update.run_update(**kwargs)
    text = env.priors_text()
    env.point_at(current)
    return text
