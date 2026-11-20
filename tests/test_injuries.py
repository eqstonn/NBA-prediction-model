"""
Tests for the injury warnings (Phase 12, display only): code/injuries.py's parsing (ESPN JSON, CBS HTML),
team and player matching, statuses, the key-player rule, the fetch with its backup and failure, the CSV,
and build_site.py's output: warnings on the cards, the disclaimer, a failed fetch never stops the build,
and the predictions are identical with and without warnings. No network: conftest refuses the sources.
"""
import copy
import csv
import json

import pytest

import build_site
import injuries
from fake_nba import ID, update

FETCHED = "2026-11-20T15:00:00Z"   # Friday Nov 20, 10:00 ET


def espn_entry(first, last, status, comment="", date="2026-11-20T14:00Z", body="Knee", detail="Soreness", side="Left"):
    return {"id": "1", "status": status, "date": date, "shortComment": comment,
            "athlete": {"firstName": first, "lastName": last, "displayName": f"{first} {last}"},
            "details": {"type": body, "detail": detail, "side": side}}


ESPN = {"timestamp": "2026-11-20T14:55:00Z", "injuries": [
    {"id": "2", "displayName": "Boston Celtics", "injuries": [
        espn_entry("Jayson", "Tatum", "Out", "Tatum (Achilles) is out for Friday's game."),
        espn_entry("Derrick", "White", "Day-To-Day", "White (knee) is questionable for Friday's game against the Knicks."),
        espn_entry("Payton", "Pritchard", "Day-To-Day", "Pritchard is listed as probable for Friday's game."),
        espn_entry("Sam", "Hauser", "Day-To-Day", "Hauser is out for Wednesday's game.", date="2026-11-19T14:00Z")]},
    {"id": "12", "displayName": "LA Clippers", "injuries": [
        espn_entry("Kawhi", "Leonard", "Day-To-Day", "Leonard is doubtful for Saturday's game.")]},
    {"id": "99", "displayName": "Seattle Supersonics", "injuries": [espn_entry("Shawn", "Kemp", "Out")]},
]}

CBS = """<html><body>
<div class="TeamLogoNameLockup"><span class="TeamName"><a href="/nba/teams/BOS/boston-celtics/">Boston</a></span></div>
<table class="TableBase-table"><tbody>
<tr class="TableBase-bodyTr"><td class="TableBase-bodyTd"><span class="CellPlayerName--short"><a>J. Tatum</a></span>
<span class="CellPlayerName--long"><a>Jayson Tatum</a></span></td><td class="TableBase-bodyTd"> SF </td>
<td class="TableBase-bodyTd"><span class="CellGameDate"> Fri, Nov 20 </span></td><td class="TableBase-bodyTd"> Achilles </td>
<td class="TableBase-bodyTd"> Expected to be out until at least Dec 1 </td></tr>
<tr class="TableBase-bodyTr"><td class="TableBase-bodyTd"><span class="CellPlayerName--long"><a>Derrick White</a></span></td>
<td class="TableBase-bodyTd"> G </td><td class="TableBase-bodyTd"> Fri, Nov 20 </td><td class="TableBase-bodyTd"> Knee </td>
<td class="TableBase-bodyTd"> Game Time Decision </td></tr>
</tbody></table>
<div class="TeamLogoNameLockup"><span class="TeamName"><a href="/nba/teams/LAC/la-clippers/">LA</a></span></div>
<table class="TableBase-table"><tbody>
<tr class="TableBase-bodyTr"><td class="TableBase-bodyTd"><span class="CellPlayerName--long"><a>Kawhi Leonard</a></span></td>
<td class="TableBase-bodyTd"> F </td><td class="TableBase-bodyTd"> Thu, Nov 19 </td><td class="TableBase-bodyTd"> Knee </td>
<td class="TableBase-bodyTd"> Out for the season </td></tr>
</tbody></table></body></html>"""


# ---------------------------------------------------------------------------
# parsing, teams, statuses
# ---------------------------------------------------------------------------

def test_parse_espn():
    r = injuries.parse_espn(ESPN, FETCHED)
    assert r["source"] == "ESPN" and r["updated"] == "2026-11-20T14:55:00Z" and r["fetched_at"] == FETCHED
    by = {e["player"]: e for e in r["entries"]}
    assert by["Jayson Tatum"]["status"] == "Out" and by["Jayson Tatum"]["team_id"] == ID["BOS"]
    assert by["Jayson Tatum"]["last_name"] == "Tatum" and by["Jayson Tatum"]["reason"] == "Left Knee Soreness"
    assert by["Derrick White"]["status"] == "Questionable" and by["Derrick White"]["source_status"] == "Day-To-Day"
    assert by["Payton Pritchard"]["status"] == "Probable"
    assert by["Sam Hauser"]["status"] == "Day-To-Day"            # Wednesday's game is over: the note is stale
    assert by["Kawhi Leonard"]["status"] == "Doubtful" and by["Kawhi Leonard"]["team_id"] == ID["LAC"]
    assert by["Shawn Kemp"]["team_id"] is None and r["unmatched_teams"] == ["Seattle Supersonics"]
    with pytest.raises(ValueError):
        injuries.parse_espn({"nope": 1}, FETCHED)


def test_parse_cbs():
    r = injuries.parse_cbs(CBS, FETCHED)
    assert r["source"] == "CBS" and r["unmatched_teams"] == []
    got = [(e["player"], e["team_id"], e["status"], e["reason"]) for e in r["entries"]]
    assert got == [("Jayson Tatum", ID["BOS"], "Out", "Achilles"), ("Derrick White", ID["BOS"], "Questionable", "Knee"),
                   ("Kawhi Leonard", ID["LAC"], "Out", "Knee")]
    assert r["entries"][0]["reported_at"] == "Fri, Nov 20"
    with pytest.raises(ValueError):
        injuries.parse_cbs("<html>maintenance</html>", FETCHED)


@pytest.mark.parametrize("name, team", [("Boston Celtics", "BOS"), ("LA Clippers", "LAC"), ("los-angeles-lakers", "LAL"),
                                        ("Portland Trail Blazers", "POR"), ("Philadelphia 76ers", "PHI")])
def test_team_matching(name, team):
    assert injuries.team_id_for(name) == ID[team]


def test_unknown_team_is_none():
    assert injuries.team_id_for("Seattle Supersonics") is None


@pytest.mark.parametrize("text, status", [("Out", "Out"), ("Day-To-Day", "Day-To-Day"), ("Game Time Decision", "Questionable"),
                                          ("Expected to be out until at least Dec 1", "Out"), ("Out for the season", "Out"),
                                          ("Doubtful", "Doubtful"), ("Probable", "Probable"), ("GTD", "Day-To-Day")])
def test_normalize_status(text, status):
    assert injuries.normalize_status(text) == status


def test_stale_note_uses_the_nearest_weekday():
    # updated Thursday morning about Wednesday's game: the game was yesterday
    assert injuries.stale_note("out for Wednesday's game", "2026-11-19T14:46Z", "2026-11-20T04:45Z")
    assert not injuries.stale_note("questionable for Friday's game", "2026-11-17T16:00Z", "2026-11-20T15:00Z")
    assert not injuries.stale_note("out for Thursday's game", "2026-11-20T03:00Z", "2026-11-20T04:45Z")  # ET: still Thu
    assert not injuries.stale_note("no weekday here", "2026-11-01T00:00Z", FETCHED)


def test_names():
    assert injuries.norm_name("T.J. McConnell") == injuries.norm_name("TJ McConnell") == "tj mcconnell"
    assert injuries.norm_name("Nikola Jokić") == "nikola jokic"
    assert injuries.norm_name("Jimmy Butler III") == injuries.norm_name("Jimmy Butler") == "jimmy butler"
    assert injuries.norm_name("De'Andre Hunter") == "deandre hunter"
    assert injuries.last_name("Gary Trent Jr.") == "Trent" and injuries.last_name("Karl-Anthony Towns") == "Towns"


# ---------------------------------------------------------------------------
# key players
# ---------------------------------------------------------------------------

def write_box(folder, gid, home_id, home_players, away_id, away_players):
    def team(tid, players):
        return {"teamId": int(tid), "players": [
            {"personId": pid, "firstName": first, "familyName": last, "statistics": {"minutes": mins}}
            for pid, first, last, mins in players]}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{gid}.json").write_text(json.dumps({"boxScoreAdvanced": {
        "gameId": gid, "homeTeam": team(home_id, home_players), "awayTeam": team(away_id, away_players)}}), encoding="utf-8")


BOS_ROSTER = [(1, "Jayson", "Tatum", "36:00"), (2, "Jaylen", "Brown", "34:00"), (3, "Derrick", "White", "30:00"),
              (4, "Payton", "Pritchard", "28:00"), (5, "Kristaps", "Porzingis", "24:00"), (6, "Sam", "Hauser", "22:00"),
              (7, "Luke", "Kornet", "10:00"), (8, "Bench", "Guy", "")]   # "" = did not play
NYK_ROSTER = [(11, "Jalen", "Brunson", "35:00"), (12, "Karl-Anthony", "Towns", "33:00"), (13, "Mikal", "Bridges", "32:00"),
              (14, "OG", "Anunoby", "31:00"), (15, "Josh", "Hart", "30:00"), (16, "Mitchell", "Robinson", "26:00"),
              (17, "Miles", "McBride", "20:00")]


@pytest.fixture
def minutes(tmp_path):
    cache = tmp_path / "adv"
    write_box(cache, "0022600010", ID["BOS"], BOS_ROSTER, ID["NYK"], NYK_ROSTER)
    write_box(cache, "0022600011", ID["BOS"], [(1, "Jayson", "Tatum", "PT40M00.00S")], ID["LAC"], [])
    write_box(cache, "0022500999", ID["BOS"], [(6, "Sam", "Hauser", "48:00")], ID["NYK"], [])   # last season: ignored
    applied = tmp_path / "applied.json"
    applied.write_text(json.dumps({"0022600010": {"date": "2026-11-01"}, "0022600011": {"date": "2026-11-03"}}),
                       encoding="utf-8")
    return injuries.season_minutes(cache, applied_path=applied)


def test_season_minutes(minutes):
    assert minutes[1]["games"] == 2 and minutes[1]["mpg"] == pytest.approx(38.0) and minutes[1]["team_id"] == ID["BOS"]
    assert minutes[6]["games"] == 1 and minutes[6]["mpg"] == pytest.approx(22.0)      # 2025-26's game left out
    assert 8 not in minutes                                                              # never played
    assert injuries.parse_minutes("33:15") == pytest.approx(33.25) and injuries.parse_minutes(None) == 0


def test_key_player_rule(minutes):
    keys = injuries.key_players(minutes)                                  # top 5 or 25+ min
    assert {p for p in keys if p < 10} == {1, 2, 3, 4, 5}                 # 6th man (22 min) is not key
    assert {p for p in keys if p > 10} == {11, 12, 13, 14, 15, 16}        # 6th man at 26 min is key
    assert keys[16] == 6
    assert set(injuries.key_players(minutes, top_n=2, min_mpg=99)) == {1, 2, 11, 12}
    assert 6 in injuries.key_players(minutes, top_n=5, min_mpg=21)
    with pytest.raises(ValueError):
        injuries.key_players(minutes, top_n=-1)


def test_team_warnings(minutes):
    report = injuries.parse_espn(ESPN, FETCHED)
    w = injuries.team_warnings(report, minutes)
    assert [(x["last_name"], x["status"], x["level"]) for x in w[ID["BOS"]]] == [
        ("Tatum", "Out", "strong"), ("White", "Questionable", "mild")]   # Pritchard probable, Hauser not key
    assert ID["LAC"] not in w                                             # Leonard has no minutes: can't be ranked
    assert w[ID["BOS"]][0]["mpg"] == 38.0 and w[ID["BOS"]][0]["rank"] == 1
    loose = injuries.team_warnings(report, minutes, top_n=6, min_mpg=99)
    assert [x["last_name"] for x in loose[ID["BOS"]]] == ["Tatum", "White", "Hauser"]


def test_player_matching_prefers_the_team_then_a_unique_name(minutes):
    entry = {"player": "Jayson Tatum", "team_id": ID["NYK"]}              # e.g. traded since his last game
    assert injuries.match_player(entry, minutes) == 1
    assert injuries.match_player({"player": "Nobody Here", "team_id": ID["BOS"]}, minutes) is None


# ---------------------------------------------------------------------------
# fetching and saving
# ---------------------------------------------------------------------------

def test_fetch_falls_back_to_cbs(monkeypatch, capsys):
    monkeypatch.setattr(injuries, "_download_espn", lambda: (_ for _ in ()).throw(ConnectionError("ESPN down")))
    monkeypatch.setattr(injuries, "_download_cbs", lambda: CBS)
    r = injuries.fetch_report()
    assert r["source"] == "CBS" and len(r["entries"]) == 3
    assert "trying CBS" in capsys.readouterr().out


def test_fetch_failure_raises_with_both_reasons(monkeypatch):
    monkeypatch.setattr(injuries, "_download_espn", lambda: {"injuries": []})       # empty: try the backup
    with pytest.raises(injuries.InjuryFetchError, match="ESPN: no entries.*CBS: ConnectionError"):
        injuries.fetch_report()


def test_update_report_appends_with_fetch_time(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(injuries, "_download_espn", lambda: ESPN)
    path = tmp_path / "inj.csv"
    first = injuries.update_report(path)
    injuries.update_report(path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * len(first["entries"]) and list(rows[0]) == injuries.FIELDS
    assert rows[0]["fetched_at"] and rows[0]["source"] == "ESPN"
    out = capsys.readouterr().out
    assert "not matched to an NBA team id: 'Seattle Supersonics'" in out
    path.write_text("other,columns\n", encoding="utf-8")
    with pytest.raises(ValueError):                                      # never appends to a different file
        injuries.update_report(path)


def test_injury_csv_is_gitignored():
    from fake_nba import REPO_ROOT
    assert "data/injuries_*.csv" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the site
# ---------------------------------------------------------------------------

def site_game(status="scheduled"):
    return {"game_id": "0022600300", "date": "2026-12-25", "tipoff_utc": None, "status": status, "status_text": "7:30 pm ET",
            "neutral": False,
            "home": {"id": ID["NYK"], "abbr": "NYK", "name": "New York Knicks", "score": None},
            "away": {"id": ID["BOS"], "abbr": "BOS", "name": "Boston Celtics", "score": None}}


WARN = {ID["BOS"]: [{"player": "Jayson Tatum", "last_name": "Tatum", "status": "Out", "level": "strong", "mpg": 36.0,
                     "rank": 1}, {"player": "Derrick White", "last_name": "White", "status": "Questionable",
                                  "level": "mild", "mpg": 30.0, "rank": 3}]}


def test_annotate_and_render():
    pred = {"home_win_pct": 55.0, "away_win_pct": 45.0, "home_score": 112.0, "away_score": 109.0}
    rows = [dict(site_game(), prediction=dict(pred), pick="NYK"), dict(site_game("final"), prediction=dict(pred), pick="NYK")]
    rows[1]["home"] = dict(rows[1]["home"], score=100)
    rows[1]["away"] = dict(rows[1]["away"], score=90)
    before = copy.deepcopy(rows)
    assert build_site.annotate_injuries(rows, WARN) == 2
    assert rows[0]["injuries"] == {"home": [], "away": WARN[ID["BOS"]]} and "injuries" not in rows[1]   # final: none
    assert [r["prediction"] for r in rows] == [r["prediction"] for r in before]
    card = build_site.render_card(rows[0])
    assert '<span class="inj strong"' in card and "⚠ Tatum (Out)" in card
    assert '<span class="inj mild"' in card and "⚠ White (Questionable)" in card
    assert "⚠" not in build_site.render_card(rows[1])


def test_warnings_only_for_the_next_days():
    assert build_site.last_injury_day("2026-11-20T15:00:00Z") == "2026-11-21"
    assert build_site.last_injury_day("2026-11-21T03:30:00Z") == "2026-11-21"   # 10:30 PM ET on the 20th
    assert build_site.last_injury_day("2026-11-20T15:00:00Z", days=1) == "2026-11-20"
    rows = [dict(site_game(), date=d) for d in ("2026-11-20", "2026-11-21", "2026-11-22")]
    assert build_site.annotate_injuries(rows, WARN, through="2026-11-21") == 4
    assert ["injuries" in r for r in rows] == [True, True, False]


def test_note_line():
    meta = {"on": True, "source": "ESPN", "fetched_at": "2026-11-20T15:05:00Z", "through": "2026-11-21",
            "key_top": 5, "key_mpg": 25.0}
    note = build_site.injury_note(meta)
    assert note.startswith("Predictions don't account for injuries.")
    assert "top 5 in minutes per game" in note and "25+ min" in note and "ESPN injury report fetched Nov 20, 10:05 AM ET" in note
    assert "games through Sat Nov 21" in note
    assert "couldn't be fetched" in build_site.injury_note(dict(meta, fetched_at=None))
    assert "off" in build_site.injury_note({"on": False})


@pytest.fixture
def site(env, tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_PRIORS", str(update.PRIORS_PATH))
    monkeypatch.setattr(build_site, "SITE_DIR", tmp_path / "site")
    monkeypatch.setattr(build_site, "fetch_games", lambda start, days: ([site_game()], "test"))
    monkeypatch.setattr(injuries, "season_minutes", lambda *a, **k: {
        1: {"name": "Jayson Tatum", "norm": "jayson tatum", "team_id": ID["BOS"], "games": 10, "minutes": 360.0,
            "mpg": 36.0}})
    return tmp_path / "site"


def built(site):
    return (json.loads((site / "predictions.json").read_text(encoding="utf-8")),
            (site / "index.html").read_text(encoding="utf-8"))


def test_predictions_identical_with_and_without_warnings(site):
    report = injuries.parse_espn(ESPN, "2026-12-24T15:00:00Z")          # the day before the site's game
    with_w = build_site.main(["--seed", "7"], injury_report=report)
    data_on, page_on = built(site)
    without = build_site.main(["--seed", "7", "--injuries", "off"])
    data_off, page_off = built(site)
    assert with_w["injury_warnings"] == 1 and without["injury_warnings"] is None
    assert [g["prediction"] for g in data_on["games"]] == [g["prediction"] for g in data_off["games"]]
    assert [g["pick"] for g in data_on["games"]] == [g["pick"] for g in data_off["games"]]
    assert data_on["games"][0]["injuries"]["away"][0]["last_name"] == "Tatum" and "injuries" not in data_off["games"][0]
    assert "⚠ Tatum (Out)" in page_on and "⚠" not in page_off
    assert "Predictions don't account for injuries." in page_on and "fetched Dec 24" in page_on
    later = build_site.main(["--seed", "7"], injury_report=injuries.parse_espn(ESPN, FETCHED))   # a month early
    assert later["injury_warnings"] == 0 and "⚠ Tatum" not in built(site)[1]
    assert data_on["injuries"]["source"] == "ESPN" and data_off["injuries"]["on"] is False


def test_site_builds_without_warnings_when_the_fetch_fails(site, capsys):
    result = build_site.main(["--seed", "7"])                        # conftest: both sources refuse
    data, page = built(site)
    assert result["games"] == 1 and result["injury_warnings"] is None and "ESPN" in result["injury_error"]
    assert data["games"][0]["prediction"] is not None and "⚠" not in page and "couldn't be fetched" in page
    assert "WARNING: no injury report" in capsys.readouterr().out


def test_site_cli_checks_thresholds(site):
    with pytest.raises(SystemExit):
        build_site.main(["--key-top", "-1", "--injuries", "off"])
