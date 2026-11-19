"""
Tests for the betting lines (Phase 10a): code/lines.py (The Odds API snapshots), its hook in
update.py --update (--lines), and code/historical_lines.py (the training seasons' lines).

The Odds API is never called: conftest.py's autouse fixture removes ODDS_API_KEY, makes
lines._download_odds raise, and points the lines files at a temp folder; the tests here swap in
FakeOdds. NBA.com's schedule is a small dict (or fake_nba's, through the env fixture).
"""
import csv
from datetime import datetime, timezone

import pytest
import requests

from fake_nba import ID
import historical_lines
import lines
import update

REAL_DOWNLOAD = lines._download_odds   # saved at import, before conftest's fixture replaces it
KEY = "test-key-0123456789abcdef"
NOW = datetime(2026, 11, 19, 18, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def sched_game(game_id, day, home, away, tipoff=None):
    return {"gameId": game_id, "gameDateEst": f"{day}T00:00:00Z", "gameDateTimeUTC": tipoff,
            "homeTeam": {"teamId": int(ID[home]), "teamTricode": home},
            "awayTeam": {"teamId": int(ID[away]), "teamTricode": away}}


SCHEDULE = {"leagueSchedule": {"gameDates": [
    {"gameDate": "11/19/2026 00:00:00", "games": [
        sched_game("0022600250", "2026-11-19", "LAC", "GSW", "2026-11-20T03:30:00Z"),   # 10:30 pm ET: next day in UTC
        sched_game("0022600251", "2026-11-19", "BOS", "NYK", "2026-11-20T00:30:00Z"),
        sched_game("0022600253", "2026-11-19", "MIA", "ORL", "2026-11-19T17:00:00Z"),   # already started at NOW
    ]},
    {"gameDate": "11/21/2026 00:00:00", "games": [
        sched_game("0022600252", "2026-11-21", "DEN", "IND", "2026-11-21T20:00:00Z"),   # neutral site
    ]},
    {"gameDate": "11/23/2026 00:00:00", "games": [
        sched_game("0022600254", "2026-11-23", "PHX", "SAC", "2026-11-24T02:00:00Z"),
    ]},
]}}


def book(key, home, away, spread=None, total=None, ml=None):
    """One bookmaker: spread = home team's point, total = over/under, ml = (home price, away price)."""
    markets = []
    if ml is not None:
        markets.append({"key": "h2h", "outcomes": [{"name": home, "price": ml[0]}, {"name": away, "price": ml[1]}]})
    if spread is not None:
        markets.append({"key": "spreads", "outcomes": [{"name": home, "price": -110, "point": spread},
                                                       {"name": away, "price": -110, "point": -spread}]})
    if total is not None:
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": total},
                                                      {"name": "Under", "price": -110, "point": total}]})
    return {"key": key, "title": key, "last_update": "2026-11-19T17:55:00Z", "markets": markets}


def event(event_id, home, away, commence, books=()):
    return {"id": event_id, "sport_key": "basketball_nba", "commence_time": commence,
            "home_team": home, "away_team": away, "bookmakers": list(books)}


LAC, GSW = "Los Angeles Clippers", "Golden State Warriors"
EVENTS = [
    event("ev-lac", LAC, GSW, "2026-11-20T03:30:00Z", [
        book("draftkings", LAC, GSW, spread=-2.5, total=230.0, ml=(-150, 130)),
        book("fanduel", LAC, GSW, spread=-3.0, total=231.0, ml=(-140, 120)),
        book("betmgm", LAC, GSW, spread=-3.5, total=232.5, ml=(-160, 135))]),
    event("ev-bos", "Boston Celtics", "New York Knicks", "2026-11-20T00:30:00Z", [
        book("draftkings", "Boston Celtics", "New York Knicks", spread=-5.5, total=221.5, ml=(-220, 180)),
        book("fanduel", "Boston Celtics", "New York Knicks", spread=-6.5, ml=(-240, 196))]),     # no total
    # neutral site: The Odds API calls IND the home team, NBA.com calls DEN the home team
    event("ev-ind", "Indiana Pacers", "Denver Nuggets", "2026-11-21T20:00:00Z", [
        book("draftkings", "Indiana Pacers", "Denver Nuggets", spread=4.0, total=228.0, ml=(150, -175))]),
    event("ev-mia", "Miami Heat", "Orlando Magic", "2026-11-19T17:00:00Z", [                  # live game
        book("draftkings", "Miami Heat", "Orlando Magic", spread=-1.0, total=210.0, ml=(-115, -105))]),
    event("ev-sea", "Seattle SuperSonics", "Boston Celtics", "2026-11-22T00:00:00Z", [
        book("draftkings", "Seattle SuperSonics", "Boston Celtics", spread=1.0)]),
    event("ev-chi", "Chicago Bulls", "Boston Celtics", "2026-11-23T01:00:00Z", [               # not on NBA.com's schedule
        book("draftkings", "Chicago Bulls", "Boston Celtics", spread=1.0)]),
    event("ev-phx", "Phoenix Suns", "Sacramento Kings", "2026-11-24T02:00:00Z"),                # no books yet
]


class FakeOdds:
    """Stands in for The Odds API: counts requests and returns EVENTS with quota headers."""

    def __init__(self, events=EVENTS, remaining=497):
        self.events, self.remaining, self.calls = events, remaining, []

    def __call__(self, key):
        self.calls.append(key)
        self.remaining -= 3
        return list(self.events), {"remaining": str(self.remaining), "used": str(500 - self.remaining), "last": "3"}


@pytest.fixture
def odds(monkeypatch):
    """A key in the environment, FakeOdds as The Odds API, SCHEDULE as NBA.com's schedule, the clock at NOW."""
    fake = FakeOdds()
    monkeypatch.setenv(lines.KEY_ENV, KEY)
    monkeypatch.setattr(lines, "_download_odds", fake)
    monkeypatch.setattr(lines.boxscores, "get_schedule", lambda season, refresh=False: SCHEDULE)
    monkeypatch.setattr(lines, "_now", lambda: NOW)
    return fake


def by_game(rows):
    return {r["game_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# matching to NBA.com ids
# ---------------------------------------------------------------------------

def test_eastern_date_of_late_games():
    assert lines.eastern_date("2026-11-20T03:30:00Z") == "2026-11-19"   # 10:30 pm ET
    assert lines.eastern_date("2026-11-20T05:00:00Z") == "2026-11-20"   # midnight ET
    assert lines.eastern_date("2026-03-09T02:00:00Z") == "2026-03-08"   # EDT starts that night (UTC-4 after 2 am)


def test_team_names_map_to_every_nba_team():
    names = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
             "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
             "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
             "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
             "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
             "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]
    assert len({lines.team_id(n) for n in names} - {None}) == 30
    assert lines.team_id("LA Clippers") == lines.team_id("Los Angeles Clippers") == ID["LAC"]
    assert lines.team_id("Seattle SuperSonics") is None


def test_snapshot_matches_ids_by_date_and_teams():
    rows, unmatched, started = lines.snapshot(EVENTS, lines.schedule_index(SCHEDULE), NOW)
    games = by_game(rows)
    assert set(games) == {"0022600250", "0022600251", "0022600252"}

    lac = games["0022600250"]   # 03:30 UTC on Nov 20 is Nov 19 in New York
    assert (lac["date"], lac["home"], lac["away"], lac["flipped"]) == ("2026-11-19", "LAC", "GSW", 0)
    assert lac["tipoff_utc"] == "2026-11-20T03:30:00Z"
    assert lac["fetched_at"] == "2026-11-19T18:00:00Z"
    assert lac["odds_event_id"] == "ev-lac"

    # every unmatched game is logged with its reason; the started game is neither saved nor unmatched
    reasons = {u["odds_event_id"]: u["reason"] for u in unmatched}
    assert set(reasons) == {"ev-sea", "ev-chi", "ev-phx"}
    assert "unknown team name 'Seattle SuperSonics'" in reasons["ev-sea"]
    assert reasons["ev-chi"].startswith("no NBA.com game on 2026-11-22")
    assert reasons["ev-phx"] == "no bookmaker odds yet"
    assert started == 1


def test_flipped_neutral_site_game_is_from_nba_home_side():
    rows, _, _ = lines.snapshot(EVENTS, lines.schedule_index(SCHEDULE), NOW)
    den = by_game(rows)["0022600252"]
    # The Odds API: IND home +4, IND +150 / DEN -175; NBA.com: DEN home -> DEN -4, DEN -175 / IND +150
    assert (den["home"], den["away"], den["flipped"]) == ("DEN", "IND", 1)
    assert den["spread_home"] == -4.0
    assert (den["ml_home"], den["ml_away"]) == (-175, 150)
    assert den["home_win_prob"] > 0.5


def test_consensus_is_the_median_across_books():
    rows, _, _ = lines.snapshot(EVENTS, lines.schedule_index(SCHEDULE), NOW)
    games = by_game(rows)
    lac = games["0022600250"]
    assert (lac["spread_home"], lac["total"]) == (-3.0, 231.0)
    assert (lac["ml_home"], lac["ml_away"]) == (-150, 130)       # median of decimal odds -> one book's price
    assert lac["n_books_spread"] == lac["n_books_total"] == lac["n_books_ml"] == 3
    novig = sorted((1 / lines.american_to_decimal(h)) / (1 / lines.american_to_decimal(h) + 1 / lines.american_to_decimal(a))
                   for h, a in ((-150, 130), (-140, 120), (-160, 135)))[1]
    assert lac["home_win_prob"] == round(novig, 4)

    bos = games["0022600251"]   # two books, one without a total
    assert bos["spread_home"] == -6.0 and bos["total"] == 221.5
    assert (bos["n_books_spread"], bos["n_books_total"], bos["n_books_ml"]) == (2, 1, 2)
    # median of the decimals 1.4545 and 1.4167 -> 1.4356 -> -230 (the American median would also be -230 here)
    assert bos["ml_home"] == -230


def test_american_decimal_round_trip():
    for odds in (-1400, -220, -110, 100, 130, 900):
        assert lines.decimal_to_american(lines.american_to_decimal(odds)) == pytest.approx(odds)
    assert lines.decimal_to_american(lines.american_to_decimal(-100)) == 100   # even money either way
    with pytest.raises(ValueError):
        lines.american_to_decimal(50)


def test_stale_schedule_is_refreshed_once(odds, monkeypatch):
    """A game missing from the cached schedule (e.g. rescheduled) is found after one refresh."""
    stale = {"leagueSchedule": {"gameDates": SCHEDULE["leagueSchedule"]["gameDates"][1:]}}   # no Nov 19 games
    calls = []

    def get_schedule(season, refresh=False):
        calls.append(refresh)
        return SCHEDULE if refresh else stale

    monkeypatch.setattr(lines.boxscores, "get_schedule", get_schedule)
    result = lines.fetch_lines()
    assert calls == [False, True]
    assert result["saved"] == 3
    assert len(odds.calls) == 1   # the refresh is NBA.com, not another Odds API request


# ---------------------------------------------------------------------------
# closing line
# ---------------------------------------------------------------------------

def snap(fetched_at, game_id="0022600250", tipoff="2026-11-20T03:30:00Z", spread="-3.0"):
    return {"fetched_at": fetched_at, "game_id": game_id, "tipoff_utc": tipoff, "spread_home": spread}


def test_closing_line_is_last_snapshot_before_tipoff():
    rows = [snap("2026-11-19T15:00:00Z", spread="-2.5"),
            snap("2026-11-20T03:29:59Z", spread="-3.5"),     # the closing line
            snap("2026-11-19T23:00:00Z", spread="-3.0"),     # out of order in the file
            snap("2026-11-20T03:30:00Z", spread="-4.0"),     # AT tip-off: too late
            snap("2026-11-20T04:00:00Z", spread="-9.0")]     # live odds
    assert lines.closing_line("0022600250", rows)["spread_home"] == "-3.5"


def test_closing_line_uses_the_latest_tipoff_and_skips_games_without_one():
    rows = [snap("2026-11-19T15:00:00Z", spread="-2.5"),
            snap("2026-11-20T02:00:00Z", spread="-3.0"),
            # postponed by a day: the latest snapshot's tip-off counts, so the Nov 21 snapshot closes
            snap("2026-11-21T01:00:00Z", spread="-1.5", tipoff="2026-11-21T03:30:00Z"),
            snap("2026-11-19T20:00:00Z", game_id="0022600251", tipoff="2026-11-19T19:00:00Z")]   # only after tip-off
    closing = lines.closing_lines(rows)
    assert closing["0022600250"]["spread_home"] == "-1.5"
    assert "0022600251" not in closing
    assert lines.closing_line("0022600251", rows) is None
    assert lines.closing_line("0022699999", rows) is None


def test_closing_lines_from_the_saved_csv(odds, monkeypatch):
    lines.fetch_lines()
    monkeypatch.setattr(lines, "_now", lambda: datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc))
    lines.fetch_lines()
    closing = lines.closing_lines()   # reads LINES_PATH
    assert closing["0022600251"]["fetched_at"] == "2026-11-19T23:00:00Z"
    assert set(closing) == {"0022600250", "0022600251", "0022600252"}


# ---------------------------------------------------------------------------
# append-only file
# ---------------------------------------------------------------------------

def test_snapshots_are_appended_never_overwritten(odds, monkeypatch):
    lines.fetch_lines()
    first = lines.LINES_PATH.read_bytes()
    first_unmatched = lines.UNMATCHED_PATH.read_bytes()

    monkeypatch.setattr(lines, "_now", lambda: datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc))
    lines.fetch_lines()
    second = lines.LINES_PATH.read_bytes()
    assert second.startswith(first) and len(second) > len(first)
    assert lines.UNMATCHED_PATH.read_bytes().startswith(first_unmatched)

    rows = lines.read_lines()
    assert len(rows) == 6                                     # 3 games x 2 snapshots
    assert second.decode("utf-8").count("fetched_at,") == 1   # one header
    assert [r["fetched_at"] for r in rows] == ["2026-11-19T18:00:00Z"] * 3 + ["2026-11-19T22:00:00Z"] * 3
    assert list(rows[0]) == lines.FIELDS
    assert len(odds.calls) == 2                               # one request per run
    assert len(list(lines.RAW_DIR.glob("*.json"))) == 2       # each run's raw response kept


def test_file_with_other_columns_is_refused_untouched(odds):
    lines.LINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines.LINES_PATH.write_text("date,game_id,spread\n2026-11-01,0022600100,-3\n", encoding="utf-8")
    before = lines.LINES_PATH.read_bytes()
    with pytest.raises(ValueError, match="expected"):
        lines.fetch_lines()
    assert lines.LINES_PATH.read_bytes() == before
    assert odds.calls == []   # checked before spending a request


def test_append_rows_writes_nothing_for_no_rows(tmp_path):
    path = tmp_path / "x.csv"
    assert lines.append_rows(path, []) == 0
    assert not path.exists()


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------

def test_missing_key_raises_before_any_request(monkeypatch):
    calls = []
    monkeypatch.setattr(lines, "_download_odds", lambda key: calls.append(key))
    assert lines.api_key() is None
    with pytest.raises(lines.MissingKeyError, match="ODDS_API_KEY"):
        lines.fetch_lines()
    monkeypatch.setenv(lines.KEY_ENV, "   ")            # blank counts as missing
    with pytest.raises(lines.MissingKeyError):
        lines.fetch_lines()
    assert calls == []
    assert not lines.LINES_PATH.exists()


def test_cli_without_key_exits_with_an_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        lines.main([])
    assert exit_info.value.code == 1
    assert "ODDS_API_KEY is not set" in capsys.readouterr().err


def test_update_without_key_warns_and_still_applies_games(env, capsys):
    result = update.run_update()
    out = capsys.readouterr().out
    assert "WARNING: --lines on but ODDS_API_KEY is not set" in out
    assert result["lines"] is None
    assert result["applied"]                       # the model update ran as usual
    assert not lines.LINES_PATH.exists()


def test_update_saves_one_snapshot_per_run(env, monkeypatch, capsys):
    """With a key, --update makes ONE Odds API request per run and prints the requests left."""
    fake = FakeOdds(events=[event("ev-xmas", "New York Knicks", "Boston Celtics", "2026-12-25T17:00:00Z",
                                  [book("draftkings", "New York Knicks", "Boston Celtics", spread=-1.5, total=224.5,
                                        ml=(-125, 105))])])
    monkeypatch.setenv(lines.KEY_ENV, KEY)
    monkeypatch.setattr(lines, "_download_odds", fake)
    monkeypatch.setattr(lines, "_now", lambda: NOW)

    first = update.run_update()
    assert "requests left: 494" in capsys.readouterr().out
    assert first["lines"]["saved"] == 1
    second = update.run_update()
    assert second["lines"]["saved"] == 1 and second["applied"] == []
    assert fake.calls == [KEY, KEY]
    rows = lines.read_lines()
    assert [(r["game_id"], r["home"], r["away"], r["spread_home"]) for r in rows] == \
           [("0022600300", "NYK", "BOS", "-1.5")] * 2   # fake_nba's upcoming Christmas game


def test_update_lines_off_never_asks(env, monkeypatch, capsys):
    monkeypatch.setenv(lines.KEY_ENV, KEY)   # the autouse fixture's download raises if called
    result = update.run_update(lines="off")
    assert result["lines"] is None
    assert "WARNING" not in capsys.readouterr().out


def test_update_survives_a_failed_request_without_leaking_the_key(env, monkeypatch, capsys):
    def broken(key):
        raise lines.OddsAPIError("HTTP 401: invalid key")

    monkeypatch.setenv(lines.KEY_ENV, KEY)
    monkeypatch.setattr(lines, "_download_odds", broken)
    result = update.run_update()
    out = capsys.readouterr().out
    assert "WARNING: betting lines not saved: HTTP 401" in out
    assert result["applied"] and result["lines"] is None
    assert KEY not in out


def test_lines_flag_only_with_update(env):
    with pytest.raises(SystemExit):
        update.main(["--start-season", "--lines", "off"])
    with pytest.raises(SystemExit):
        update.main(["--update", "--lines", "maybe"])


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._body


def test_request_errors_never_contain_the_key(monkeypatch):
    def unreachable(url, params, timeout):
        raise requests.ConnectionError(f"Max retries exceeded with url: /v4/sports/basketball_nba/odds?apiKey={params['apiKey']}")

    monkeypatch.setattr(lines.requests, "get", unreachable)
    with pytest.raises(lines.OddsAPIError) as err:
        REAL_DOWNLOAD(KEY)
    assert KEY not in str(err.value) and err.value.__suppress_context__

    monkeypatch.setattr(lines.requests, "get", lambda url, params, timeout: FakeResponse(
        401, {"message": f"API key {params['apiKey']} is not valid"}, {"x-requests-remaining": "0"}))
    with pytest.raises(lines.OddsAPIError) as err:
        REAL_DOWNLOAD(KEY)
    assert "HTTP 401" in str(err.value) and KEY not in str(err.value)


def test_one_request_with_the_documented_parameters(monkeypatch):
    seen = []

    def get(url, params, timeout):
        seen.append((url, dict(params)))
        return FakeResponse(200, [], {"x-requests-remaining": "497", "x-requests-used": "3", "x-requests-last": "3"})

    monkeypatch.setattr(lines.requests, "get", get)
    events, quota = REAL_DOWNLOAD(KEY)
    assert events == [] and quota == {"remaining": "497", "used": "3", "last": "3"}
    assert len(seen) == 1
    url, query = seen[0]
    assert url == "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    assert query == {"apiKey": KEY, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american",
                     "dateFormat": "iso"}


# ---------------------------------------------------------------------------
# historical lines (training seasons)
# ---------------------------------------------------------------------------

BETTING_HEADER = ["season", "date", "regular", "playoffs", "away", "home", "score_away", "score_home",
                  "whos_favored", "spread", "total", "moneyline_away", "moneyline_home"]


def write_betting_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(BETTING_HEADER)
        writer.writerows(rows)
    return path


def training_game(game_id, day, home, away, home_pts, away_pts):
    return {"game_id": game_id, "date": day, "home": ID[home], "away": ID[away], "home_pts": float(home_pts),
            "away_pts": float(away_pts), "game_type": "Regular Season"}


def test_historical_lines_match_training_ids(tmp_path):
    path = write_betting_csv(tmp_path / "nba.csv", [
        [2024, "2023-10-24", True, False, "lal", "den", 107, 119, "home", 4.5, 228.5, "", ""],
        [2024, "2023-10-25", True, False, "atl", "cha", 110, 116, "away", 4.5, 235.5, "-180", "155"],
        [2024, "2023-11-02", True, False, "orl", "atl", 120, 119, "home", 0, 230, "", ""],       # pick'em
        [2024, "2023-11-11", True, False, "sa", "gs", 101, 99, "home", 7, 233, "", ""],         # score disagrees
        [2024, "2023-12-14", True, False, "mia", "sa", 120, 110, "away", 2.5, 222, "", ""],     # neutral, swapped
        [2024, "2023-12-09", True, False, "ind", "lal", 109, 123, "home", 3, 240, "", ""],      # no training game
        [2024, "2023-12-10", True, False, "sea", "lal", 100, 101, "home", 3, 240, "", ""],      # unknown code
        [2023, "2023-04-01", True, False, "ny", "wsh", 90, 80, "away", 3, 210, "", ""],         # before the window
    ])
    games = [training_game("0022300061", "2023-10-24", "DEN", "LAL", 119, 107),
             training_game("0022300063", "2023-10-25", "CHA", "ATL", 116, 110),
             training_game("0022300100", "2023-11-02", "ATL", "ORL", 119, 120),
             training_game("0022300150", "2023-11-11", "GSW", "SAS", 0, 0),
             training_game("0022300160", "2023-12-14", "MIA", "SAS", 120, 110),
             training_game("0022300170", "2023-12-15", "NYK", "BOS", 100, 90)]   # no betting row
    rows, report = historical_lines.match_lines(historical_lines.read_betting_csv(path), games)
    got = {r["game_id"]: r for r in rows}

    assert (got["0022300061"]["home"], got["0022300061"]["spread_home"], got["0022300061"]["total"]) == ("DEN", -4.5, 228.5)
    assert got["0022300063"]["spread_home"] == 4.5                     # away favored -> home +4.5
    assert (got["0022300063"]["ml_home"], got["0022300063"]["ml_away"]) == (155, -180)
    assert str(got["0022300100"]["spread_home"]) == "0.0"              # no "-0.0"
    assert got["0022300150"]["spread_home"] == -7.0                    # kept despite the score
    sas = got["0022300160"]                                            # betting file: SAS home, MIA favored by 2.5
    assert (sas["home"], sas["away"], sas["flipped"], sas["spread_home"]) == ("MIA", "SAS", 1, -2.5)

    assert (report["games"], report["matched"], report["flipped"]) == (6, 5, 1)
    assert report["rate"] == pytest.approx(5 / 6)
    assert report["by_season"] == {"2023-24": (5, 6)}
    assert [g["game_id"] for g in report["missing"]] == ["0022300170"]
    assert [(r["date"], r["home_code"]) for r in report["unmatched_lines"]] == [("2023-12-09", "lal"), ("2023-12-10", "lal")]
    assert [g["game_id"] for g, _ in report["score_mismatch"]] == ["0022300150"]
    assert report["unknown_codes"] == {"sea": 1}


def test_historical_team_codes_cover_every_team():
    codes = ["atl", "bkn", "bos", "cha", "chi", "cle", "dal", "den", "det", "gs", "hou", "ind", "lac", "lal", "mem",
             "mia", "mil", "min", "no", "ny", "okc", "orl", "phi", "phx", "por", "sa", "sac", "tor", "utah", "wsh"]
    assert len({historical_lines.team_id(c) for c in codes} - {None}) == 30


def test_historical_cli_needs_the_csv(tmp_path, capsys):
    with pytest.raises(SystemExit):
        historical_lines.main(["--csv", str(tmp_path / "missing.csv")])
    assert "download nba_2008-2026.csv from Kaggle" in capsys.readouterr().err
