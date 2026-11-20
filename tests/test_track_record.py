"""
Tests for the site's track record (Phase 13, display only): code/track_record.py's numbers (the same as
report.py's), the backtest headline parsed from reports/accuracy_report.md, model vs Vegas on games with a
closing line, the "How accurate is it?" section, and that the predictions on the site don't change.
"""
import csv
import json
from datetime import date

import pytest

import build_site
import lines
import predlog
import report
import track_record
from fake_nba import ID, REPO_ROOT, update

TODAY = date(2026, 11, 20)


def log_row(gid, day, home, away, winner, pct, home_pts, away_pts):
    return {"date": day, "game_id": gid, "home": home, "away": away, "predicted_winner": winner, "win_pct": f"{pct:.2f}",
            "proj_home": "110.0", "proj_away": "105.0", "actual_home": str(home_pts), "actual_away": str(away_pts)}


def write_log(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=predlog.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


ROWS = [log_row("0022600001", "2026-11-10", "BOS", "NYK", "BOS", 70.0, 110, 100),    # right
        log_row("0022600002", "2026-11-15", "LAL", "GSW", "GSW", 60.0, 120, 100),    # wrong (home won)
        log_row("0022600003", "2026-11-19", "MIA", "ORL", "MIA", 50.0, 99, 101),     # 50/50: counts half
        log_row("0022600004", "2026-11-20", "DEN", "UTA", "DEN", 80.0, 130, 100)]    # right


@pytest.fixture
def logs(tmp_path):
    first = write_log(tmp_path / "replay.csv", ROWS[:3])
    # the live log repeats game 3 with another win % (the first log wins) and adds game 4
    second = write_log(tmp_path / "live.csv", [log_row("0022600003", "2026-11-19", "MIA", "ORL", "ORL", 90.0, 99, 101),
                                               ROWS[3]])
    return [first, second]


# ---------------------------------------------------------------------------
# the numbers
# ---------------------------------------------------------------------------

def test_same_logs_and_game_reading_as_the_report():
    assert track_record.LIVE_LOGS == report.LIVE_LOGS
    r = ROWS[1]
    ours, theirs = track_record.log_row_to_game(r), report.log_row_to_game(r)
    for key in ("game_id", "date", "home", "away", "home_won", "p_model"):
        assert ours[key] == theirs[key]
    assert ours["pick"] == "GSW" and ours["pick_pct"] == 60.0 and ours["p_model"] == pytest.approx(0.4)


def test_season_games_first_log_wins_and_sorted(logs):
    games = track_record.season_games(logs)
    assert [g["game_id"] for g in games] == ["0022600001", "0022600002", "0022600003", "0022600004"]
    assert games[2]["p_model"] == 0.5                                   # from the first log, not the 90% repeat


def test_summarize(logs):
    s = track_record.summarize(track_record.season_games(logs))
    assert s["games"] == 4 and s["right"] == 2.5 and s["accuracy"] == pytest.approx(0.625)
    # Brier: (0.7-1)^2 + (0.4-1)^2 + (0.5-0)^2 + (0.8-1)^2, over 4
    assert s["brier"] == pytest.approx((0.09 + 0.36 + 0.25 + 0.04) / 4)
    assert s["confidence"] == pytest.approx((0.7 + 0.6 + 0.5 + 0.8) / 4)
    assert track_record.summarize([]) == {"games": 0, "right": 0, "accuracy": None, "brier": None, "confidence": None}


def test_last_7_days_and_recent(logs, tmp_path):
    tr = track_record.build(TODAY, logs, tmp_path / "no_lines.csv", tmp_path / "no_report.md", recent=3)
    assert tr["since"] == "2026-11-14" and tr["last7"]["games"] == 3                       # Nov 15, 19, 20
    assert [g["game_id"] for g in tr["recent"]] == ["0022600004", "0022600003", "0022600002"]   # newest first
    assert [g["right"] for g in tr["recent"]] == [True, False, False]
    assert tr["backtest"] is None and tr["vegas"] is None
    assert track_record.build(date(2026, 12, 31), logs, tmp_path / "x.csv", tmp_path / "x.md")["last7"]["games"] == 0


def test_matches_the_report_on_the_real_logs():
    """The site's 2026-27 numbers are the report's (67.6% of 210 and Brier 0.2110 when it was generated)."""
    if not track_record.LIVE_LOGS[0].exists():
        pytest.skip("no live log")
    games = track_record.season_games()
    expected = report.evaluate([g["p_model"] for g in games], [g["home_won"] for g in games])
    s = track_record.summarize(games)
    assert s["accuracy"] == pytest.approx(expected["accuracy"]) and s["brier"] == pytest.approx(expected["brier"])
    assert s["confidence"] == pytest.approx(expected["confidence"])


# ---------------------------------------------------------------------------
# the report's headline
# ---------------------------------------------------------------------------

def test_parse_the_real_report():
    b = track_record.read_backtest()
    if b is None:
        pytest.skip("no accuracy report")
    rows = b["rows"]
    assert rows["Model"]["games"] == rows["Vegas (spread)"]["games"] == rows["Home team"]["games"] > 2000
    assert 0.5 < rows["Home team"]["accuracy"] < rows["Model"]["accuracy"] < 0.8
    assert b["sd"] == pytest.approx(13.83) and b["generated"]


REPORT = """# Accuracy report
Generated 2026-11-19 by x. SD = **13.83 points**, fitted.

## Training seasons, main: live-procedure backtest (2,638 games)

| Predictor | Games | Winner accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Model | 2638 | 66.5% | 0.2135 | 0.6161 |
| Vegas (spread) | 2638 | 68.9% | 0.1994 | 0.5823 |
| Home team | 2638 | 55.0% | 0.2475 | 0.6881 |
| Better record | 2638 | 65.2% | 0.2265 | 0.6455 |

## Next section
| Model | 9 | 1.0% | 0.9 |
"""


def test_parse_backtest():
    b = track_record.parse_backtest(REPORT)
    assert b["generated"] == "2026-11-19" and b["sd"] == 13.83
    assert b["rows"]["Model"] == {"games": 2638, "accuracy": pytest.approx(0.665), "brier": 0.2135}
    assert b["rows"]["Vegas (spread)"]["accuracy"] == pytest.approx(0.689)
    assert b["rows"]["Home team"]["accuracy"] == pytest.approx(0.55)


@pytest.mark.parametrize("text", ["no table here", REPORT.replace("| Home team |", "| Away team |"),
                                  REPORT.replace("66.5%", "n/a")])
def test_unreadable_report_gives_none(text):
    assert track_record.parse_backtest(text) is None


# ---------------------------------------------------------------------------
# Vegas on this season's games
# ---------------------------------------------------------------------------

def snapshot(gid, fetched, tipoff, spread):
    return {"fetched_at": fetched, "game_id": gid, "date": "2026-11-20", "tipoff_utc": tipoff, "home": "DEN",
            "away": "UTA", "spread_home": spread, "total": "230.5", "ml_home": "", "ml_away": "", "home_win_prob": "",
            "n_books_spread": "5", "n_books_total": "5", "n_books_ml": "0", "books_updated": "", "odds_event_id": "e",
            "flipped": "0"}


def test_vegas_only_on_games_with_a_closing_line(logs, tmp_path):
    path = tmp_path / "lines.csv"
    lines.append_rows(path, [snapshot("0022600004", "2026-11-20T15:00:00Z", "2026-11-21T02:00:00Z", "-9.5"),
                             snapshot("0022600004", "2026-11-21T01:00:00Z", "2026-11-21T02:00:00Z", "-10.5"),
                             snapshot("0022600001", "2026-11-10T03:00:00Z", "2026-11-10T00:30:00Z", "-3.5")])  # after tip
    tr = track_record.build(TODAY, logs, path, tmp_path / "x.md")
    assert tr["vegas"]["games"] == 1                                     # only DEN-UTA has a pre-tip-off line
    p = tr["vegas"]["vegas"]
    expected = float(report.spread_win_prob(-10.5, 13.83))                # the closing (last pre-tip) line
    assert p["brier"] == pytest.approx((expected - 1) ** 2) and p["accuracy"] == 1.0
    assert tr["vegas"]["model"]["games"] == 1
    assert track_record.spread_win_prob(-4.0, 12.0) == pytest.approx(float(report.spread_win_prob(-4.0, 12.0)))


# ---------------------------------------------------------------------------
# the section and the site
# ---------------------------------------------------------------------------

def test_render_section_has_counts_next_to_every_percent(logs, tmp_path):
    rp = tmp_path / "report.md"
    rp.write_text(REPORT, encoding="utf-8")
    path = tmp_path / "lines.csv"
    lines.append_rows(path, [snapshot("0022600004", "2026-11-20T15:00:00Z", "2026-11-21T02:00:00Z", "-9.5")])
    out = build_site.render_track_record(track_record.build(TODAY, logs, path, rp))
    assert "How accurate is it?" in out
    assert "62.5%</b><span>winners picked<br>2.5 of 4 games" in out
    assert "average confidence in its pick<br>vs 62.5% of picks right (4 games)" in out
    assert "<b>1.5–1.5</b><span>last 7 days (Nov 14 – Nov 20)<br>50.0% of 3 games" in out   # a 50/50 pick counts half
    assert build_site.BRIER_LINE in out
    assert "66.5%</b> of winners over 2,638 games" in out and "68.9% for the Vegas closing lines (2,638 games)" in out
    assert "55.0% for always picking the home team (2,638 games)" in out
    assert "on the 1 game with a saved closing line" in out
    assert "DEN 80%" in out and "UTA 100 – DEN 130" in out and "✓ right" in out and "✗ wrong" in out


def test_render_without_data(tmp_path):
    empty = track_record.build(TODAY, [tmp_path / "none.csv"], tmp_path / "x.csv", tmp_path / "x.md")
    out = build_site.render_track_record(empty)
    assert "No games of this season" in out and "Past seasons" not in out and "against Vegas" not in out
    assert "couldn't be read" in build_site.render_track_record(None)


@pytest.fixture
def site(env, tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_PRIORS", str(update.PRIORS_PATH))
    monkeypatch.setattr(build_site, "SITE_DIR", tmp_path / "site")
    game = {"game_id": "0022600300", "date": "2026-12-25", "tipoff_utc": None, "status": "scheduled",
            "status_text": "7:30 pm ET", "neutral": False,
            "home": {"id": ID["NYK"], "abbr": "NYK", "name": "New York Knicks", "score": None},
            "away": {"id": ID["BOS"], "abbr": "BOS", "name": "Boston Celtics", "score": None}}
    monkeypatch.setattr(build_site, "fetch_games", lambda start, days: ([game], "test"))
    return tmp_path / "site"


def test_site_shows_the_section_and_predictions_are_unchanged(site, logs, tmp_path, monkeypatch):
    rp = tmp_path / "report.md"
    rp.write_text(REPORT, encoding="utf-8")
    monkeypatch.setattr(track_record, "LIVE_LOGS", logs)
    monkeypatch.setattr(track_record, "REPORT_PATH", rp)
    result = build_site.main(["--seed", "7", "--injuries", "off"])
    with_record = json.loads((site / "predictions.json").read_text(encoding="utf-8"))
    page = (site / "index.html").read_text(encoding="utf-8")
    assert result["record_games"] == 4 and with_record["track_record"]["season"]["games"] == 4
    assert "How accurate is it?" in page and "Most recent 4 games" in page

    def broken(today):
        raise OSError("disk")
    monkeypatch.setattr(track_record, "build", broken)                  # no track record: still builds
    result = build_site.main(["--seed", "7", "--injuries", "off"])
    without = json.loads((site / "predictions.json").read_text(encoding="utf-8"))
    assert result["record_games"] is None and without["track_record"] is None
    assert "couldn't be read" in (site / "index.html").read_text(encoding="utf-8")
    assert [g["prediction"] for g in with_record["games"]] == [g["prediction"] for g in without["games"]]
    assert [g["pick"] for g in with_record["games"]] == [g["pick"] for g in without["games"]]
