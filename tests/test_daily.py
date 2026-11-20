"""
Tests for the daily automatic run (code/daily.py, Phase 11): the lock, the step order and failure handling
(nothing half-written, the site still rebuilt), retries, the Odds API budget, the pre-game time choice and
the dry run. NBA.com is fake_nba's, The Odds API is never called (lines.fetch_lines is replaced or refused),
the site and logs go to a temp folder, and the real priors.json / applied_games.json are never touched.
"""
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import boxscores
import build_site
import daily
import lines
from fake_nba import EXPECTED, ID, REPO_ROOT, reference_priors, update

NO_WAITS = (0, 0)
TODAY = date(2026, 11, 20)


def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """Never read the real user registry, never sleep."""
    monkeypatch.setattr(daily, "_registry_key", lambda: None)
    monkeypatch.setattr(daily, "_sleep", lambda s: None)


@pytest.fixture
def denv(env, tmp_path, monkeypatch):
    """env (fake NBA.com, temp pipeline files) + the site in a temp folder, built from the temp priors.json."""
    monkeypatch.setenv("NBA_PRIORS", str(update.PRIORS_PATH))
    monkeypatch.setattr(build_site, "SITE_DIR", tmp_path / "site")
    games = [{"game_id": "0022600300", "date": "2026-12-25", "tipoff_utc": None, "status": "scheduled",
              "status_text": "7:30 pm ET", "neutral": False,
              "home": {"id": ID["NYK"], "abbr": "NYK", "name": "New York Knicks", "score": None},
              "away": {"id": ID["BOS"], "abbr": "BOS", "name": "Boston Celtics", "score": None}}]
    monkeypatch.setattr(build_site, "fetch_games", lambda start, days: (games, "test"))
    env.logs = tmp_path / "logs"
    return env


def fake_lines(calls, quota="470", fail=()):
    """A lines.fetch_lines stand-in: raises the errors in `fail` first, then saves 9 games."""
    errors = list(fail)

    def fetch():
        calls.append(1)
        if errors:
            raise errors.pop(0)
        lines.LINES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(lines.LINES_PATH, "a", encoding="utf-8") as f:
            f.write("snapshot\n")
        return {"fetched_at": "x", "events": 9, "saved": 9, "started": 0, "unmatched": [],
                "quota": {"remaining": quota, "used": "30", "last": "3"}}
    return fetch


def log_text(env):
    return "".join(p.read_text(encoding="utf-8") for p in sorted(env.logs.glob("daily_*.log")))


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------

def test_lock_excludes_a_second_holder(tmp_path):
    first, second = daily.RunLock(tmp_path / "x.lock"), daily.RunLock(tmp_path / "x.lock")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_run_exits_when_another_run_holds_the_lock(denv, monkeypatch):
    monkeypatch.setenv(lines.KEY_ENV, "k")
    calls = []
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls))
    before = (sha(update.PRIORS_PATH), sha(update.APPLIED_GAMES_PATH))
    holder = daily.RunLock(denv.logs / daily.LOCK_NAME)
    assert holder.acquire()
    try:
        assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_LOCKED
    finally:
        holder.release()
    assert (sha(update.PRIORS_PATH), sha(update.APPLIED_GAMES_PATH)) == before and not calls
    assert "another run holds" in log_text(denv)
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK      # free again


# ---------------------------------------------------------------------------
# the morning run
# ---------------------------------------------------------------------------

def test_morning_run_updates_saves_lines_and_builds_the_site(denv, monkeypatch, capsys):
    monkeypatch.setenv(lines.KEY_ENV, "k")
    calls = []
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls))
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK
    assert list(denv.applied()) == EXPECTED and len(calls) == 1
    assert {e["engine"] for e in denv.applied().values()} == {"exact"}            # the live defaults
    assert (build_site.SITE_DIR / "index.html").exists() and (build_site.SITE_DIR / "predictions.json").exists()
    state = daily.load_state(denv.logs)
    assert state["quota"]["remaining"] == "470" and state["morning_lines"] == state["quota"]["at"]
    text = log_text(denv)
    for piece in ("--- update", "--- morning lines", "--- site", "games applied: 6", "lines saved: 9",
                  "Odds API credits left: 470", "site games: 1", "warnings: 0", "git add", "exit code 0"):
        assert piece in text, piece
    assert text.index("--- update") < text.index("--- morning lines") < text.index("--- site")
    assert "git add" in capsys.readouterr().out                                  # also on the console
    # a second run the same day: no new games, no second snapshot (one a day), nothing to commit
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK
    assert len(calls) == 1 and "already saved today" in log_text(denv)
    assert log_text(denv).rstrip().endswith("exit code 0") and "no tracked data changed" in log_text(denv)
    assert daily.run(["--force-lines"], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 2


def test_no_key_is_a_warning_and_the_rest_still_runs(denv):
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_WARNINGS
    assert list(denv.applied()) == EXPECTED and (build_site.SITE_DIR / "index.html").exists()
    assert "ODDS_API_KEY is not set" in log_text(denv)


def test_key_from_the_user_registry_is_used_but_never_printed(denv, monkeypatch, capsys):
    monkeypatch.setattr(daily, "_registry_key", lambda: "secret-key-123")
    calls = []
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls))
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 1
    assert "secret-key-123" not in log_text(denv) + capsys.readouterr().out


def test_failed_update_writes_nothing_and_the_site_is_still_built(denv, monkeypatch):
    monkeypatch.setenv(lines.KEY_ENV, "k")
    monkeypatch.setattr(lines, "fetch_lines", fake_lines([]))
    tries = []

    def down(season, season_type):
        tries.append(season_type)
        raise ConnectionError("NBA.com is down")
    monkeypatch.setattr(boxscores, "_download_game_log", down)
    before = (sha(update.PRIORS_PATH), sha(update.APPLIED_GAMES_PATH), sha(update.LOG_PATH))
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_FAILED
    assert (sha(update.PRIORS_PATH), sha(update.APPLIED_GAMES_PATH), sha(update.LOG_PATH)) == before
    assert len(tries) == 3 * boxscores.RETRIES                       # 3 attempts of the step, each retried per request
    assert (build_site.SITE_DIR / "index.html").exists()               # rebuilt from the unchanged priors
    text = log_text(denv)
    assert "update failed after 3 attempts" in text and "games applied: failed" in text
    assert "lines saved: 9" in text                                    # the lines step still ran


def test_update_failing_midway_keeps_exactly_the_games_applied_before(denv, tmp_path, monkeypatch):
    expected = reference_priors(denv, tmp_path, limit=2)
    real_apply = update.apply_game

    def third_fails(priors, game, *args, **kwargs):
        if game["game_id"] == EXPECTED[2]:
            raise RuntimeError("simulated crash")
        return real_apply(priors, game, *args, **kwargs)
    monkeypatch.setattr(update, "apply_game", third_fails)
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_FAILED
    assert list(denv.applied()) == EXPECTED[:2]
    assert not any("pending" in e for e in denv.applied().values())
    assert denv.priors_text() == expected
    assert update.priors_fingerprint(json.loads(denv.priors_text())) == \
        update.priors_fingerprint(json.loads(expected))


def test_site_failure_is_a_failed_step(denv, monkeypatch):
    monkeypatch.setattr(build_site, "fetch_games", lambda start, days: (_ for _ in ()).throw(OSError("ESPN down")))
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_FAILED
    assert list(denv.applied()) == EXPECTED
    assert "site not rebuilt" in log_text(denv) and "site games: not rebuilt" in log_text(denv)


def test_lines_are_retried_only_when_worth_it(denv, monkeypatch):
    monkeypatch.setenv(lines.KEY_ENV, "k")
    calls = []
    busy = [lines.OddsAPIError("HTTP 503: busy", status=503), lines.OddsAPIError("request failed (Timeout)")]
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls, fail=busy))
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls, fail=[lines.OddsAPIError("HTTP 401: bad key", 401)]))
    assert daily.run(["--force-lines"], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_WARNINGS
    assert len(calls) == 1 and "morning lines not saved: HTTP 401" in log_text(denv)


def test_odds_error_retryable():
    assert lines.OddsAPIError("x").retryable and lines.OddsAPIError("x", 429).retryable
    assert lines.OddsAPIError("x", 502).retryable
    assert not lines.OddsAPIError("x", 401).retryable and not lines.OddsAPIError("x", 422).retryable


# ---------------------------------------------------------------------------
# credits
# ---------------------------------------------------------------------------

def test_odds_budget():
    today = date(2026, 11, 20)                                         # 10 days left in November
    assert daily.days_left_in_month(today) == 10
    assert daily.odds_budget({}, "morning", today)[0]
    q = lambda remaining, at="2026-11-20": {"quota": {"remaining": str(remaining), "at": at}}   # noqa: E731
    assert daily.odds_budget(q(200), "pregame", today)[0]
    assert daily.odds_budget(q(33), "pregame", today)[0]               # 33 - 3 = 30 = 10 mornings
    assert not daily.odds_budget(q(32), "pregame", today)[0]           # would eat a morning's credits
    assert daily.odds_budget(q(32), "morning", today)[0]
    assert not daily.odds_budget(q(2), "morning", today)[0]            # out of credits, counted today
    assert daily.odds_budget(q(2, "2026-11-19"), "morning", today)[0]  # counted yesterday: one try today
    assert daily.odds_budget(q(0, "2026-10-31"), "pregame", today)[0]  # last month's count: reset since


def test_morning_lines_skipped_when_out_of_credits(denv, monkeypatch):
    monkeypatch.setenv(lines.KEY_ENV, "k")
    calls = []
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls))
    today = datetime.now(timezone.utc).astimezone(daily.EASTERN).date().isoformat()
    denv.logs.mkdir(parents=True)
    daily.save_state(denv.logs, {"quota": {"remaining": "1", "at": today}})
    assert daily.run([], log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_WARNINGS
    assert not calls and "only 1 Odds API credits left" in log_text(denv)


# ---------------------------------------------------------------------------
# the pre-game time and run
# ---------------------------------------------------------------------------

def sched(*games):
    days = {}
    for gid, day, tip_utc, status_text, *rest in games:
        g = {"gameId": gid, "gameDateEst": f"{day}T00:00:00Z", "gameDateTimeUTC": tip_utc,
             "gameStatusText": status_text, "postponedStatus": rest[0] if rest else "N",
             "homeTeam": {"teamId": int(ID["BOS"])}, "awayTeam": {"teamId": int(ID["NYK"])}}
        days.setdefault(day, []).append(g)
    return {"leagueSchedule": {"gameDates": [{"games": g} for g in days.values()]}}


DAY = sched(("0022600301", "2026-11-20", "2026-11-21T00:30:00Z", "7:30 pm ET"),     # 19:30 ET
            ("0022600302", "2026-11-20", "2026-11-20T20:00:00Z", "3:00 pm ET"),     # 15:00 ET: the first
            ("0022600303", "2026-11-20", "2026-11-21T03:00:00Z", "10:00 pm ET"),    # 22:00 ET: the last
            ("0022600304", "2026-11-20", "2026-11-20T17:00:00Z", "PPD", "Y"),       # postponed: ignored
            ("0022600305", "2026-11-21", "2026-11-21T17:00:00Z", "12:00 pm ET"))    # another day


def et(day, hh, mm=0):
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=daily.EASTERN)


def test_pregame_time_is_an_hour_before_the_first_tipoff():
    plan = daily.pregame_plan(DAY, TODAY)
    assert plan["games"] == 3 and plan["due"] == et(TODAY, 14) and plan["last"] == et(TODAY, 22)


def test_pregame_time_falls_back_to_6_pm():
    tbd = sched(("0022600401", "2026-11-20", "2026-11-20T05:00:00Z", "TBD"))
    assert daily.pregame_plan(tbd, TODAY)["due"] == et(TODAY, 18)
    assert daily.pregame_plan(None, TODAY)["due"] == et(TODAY, 18)
    assert daily.pregame_plan(DAY, date(2026, 11, 22))["due"] is None            # no games


@pytest.fixture
def pregame(tmp_path, monkeypatch):
    """--pregame with DAY's schedule and a clock; returns (calls, set_clock, log_dir)."""
    monkeypatch.setenv(lines.KEY_ENV, "k")
    monkeypatch.setattr(daily, "current_schedule", lambda: DAY)
    monkeypatch.setattr(lines, "LINES_PATH", tmp_path / "data" / "lines.csv")
    calls = []
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls))
    clock = {}
    monkeypatch.setattr(daily, "_now", lambda: clock["now"].astimezone(timezone.utc))

    def set_clock(hh, mm=0):
        clock["now"] = et(TODAY, hh, mm)
    return calls, set_clock, tmp_path / "logs"


def test_pregame_waits_until_due_then_takes_one_snapshot(pregame):
    calls, set_clock, logs = pregame
    set_clock(13, 45)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK and not calls
    assert "not due until 14:00 ET" in (logs / "daily_2026-11-20.log").read_text(encoding="utf-8") or \
        any("not due until 14:00 ET" in p.read_text(encoding="utf-8") for p in logs.glob("*.log"))
    set_clock(14, 5)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 1
    assert daily.load_state(logs)["pregame_lines"] == "2026-11-20"
    set_clock(14, 20)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 1


def test_pregame_catches_up_after_sleep_but_not_after_every_game_started(pregame):
    calls, set_clock, logs = pregame
    set_clock(20, 10)                                   # asleep at 14:00, back after the first tip-offs
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 1
    daily.save_state(logs, {})
    set_clock(22, 30)                                   # every game has started: no request, one warning
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_WARNINGS and len(calls) == 1
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK      # handled for today


def test_failed_pregame_snapshot_is_retried_by_the_next_trigger(pregame, monkeypatch):
    calls, set_clock, logs = pregame
    monkeypatch.setattr(lines, "fetch_lines", fake_lines(calls, fail=[lines.OddsAPIError("x", 500)] * 3))
    set_clock(14, 5)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_FAILED and len(calls) == 3
    assert "pregame_lines" not in daily.load_state(logs)
    set_clock(14, 20)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_OK and len(calls) == 4


def test_pregame_keeps_credits_for_the_mornings(pregame):
    calls, set_clock, logs = pregame
    logs.mkdir(parents=True)
    daily.save_state(logs, {"quota": {"remaining": "20", "at": "2026-11-20"}})
    set_clock(14, 5)
    assert daily.run(["--pregame"], log_dir=logs, waits=NO_WAITS) == daily.EXIT_WARNINGS and not calls
    assert daily.load_state(logs)["pregame_lines"] == "2026-11-20"     # not retried every 15 minutes


# ---------------------------------------------------------------------------
# dry run, summary, log
# ---------------------------------------------------------------------------

def refuse_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the dry run must not use the network")
    for name in ("_download_game_log", "_download_schedule", "_download_box_score", "_download_advanced_box_score"):
        monkeypatch.setattr(boxscores, name, refuse)
    monkeypatch.setattr(lines, "_download_odds", refuse)
    monkeypatch.setattr(lines, "fetch_lines", refuse)
    monkeypatch.setattr(build_site, "fetch_games", refuse)


@pytest.mark.parametrize("args", [["--dry-run"], ["--pregame", "--dry-run"]])
def test_dry_run_changes_no_files(denv, monkeypatch, capsys, args):
    refuse_network(monkeypatch)
    folder = update.PRIORS_PATH.parent
    before = {p: sha(p) for p in folder.rglob("*") if p.is_file()}
    assert daily.run(args, log_dir=denv.logs, waits=NO_WAITS) == daily.EXIT_OK
    assert {p: sha(p) for p in folder.rglob("*") if p.is_file()} == before
    assert not denv.logs.exists() and not build_site.SITE_DIR.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_dry_run_on_the_real_repo_changes_no_files(monkeypatch, capsys):
    refuse_network(monkeypatch)
    watched = sorted(REPO_ROOT.glob("*.json")) + sorted(REPO_ROOT.glob("*.csv")) + sorted((REPO_ROOT / "data").glob("*"))
    before = {p: sha(p) for p in watched}
    logs_before = sorted((REPO_ROOT / "logs").rglob("*")) if (REPO_ROOT / "logs").exists() else None
    for args in (["--dry-run"], ["--pregame", "--dry-run"]):
        assert daily.run(args) == daily.EXIT_OK
    assert {p: sha(p) for p in watched} == before
    assert (sorted((REPO_ROOT / "logs").rglob("*")) if (REPO_ROOT / "logs").exists() else None) == logs_before
    out = capsys.readouterr().out
    assert "1. update" in out and "pre-game snapshot" in out


def test_git_add_line_lists_only_changed_files_relative_to_the_repo():
    a, b, c = REPO_ROOT / "priors.json", REPO_ROOT / "data" / "lines_2026-27.csv", REPO_ROOT / "prediction_log.csv"
    assert daily.git_add_line({a: "1", b: "2", c: None}, {a: "1", b: "3", c: "4"}) == \
        "git add data/lines_2026-27.csv prediction_log.csv"
    assert daily.git_add_line({a: "1"}, {a: "1"}) == ""


def test_log_is_one_file_per_day_appended(denv):
    daily.run([], log_dir=denv.logs, waits=NO_WAITS)
    daily.run([], log_dir=denv.logs, waits=NO_WAITS)
    logs = list(denv.logs.glob("daily_*.log"))
    assert len(logs) == 1 and logs[0].read_text(encoding="utf-8").count("=== daily run") == 2


def test_cli():
    with pytest.raises(SystemExit) as err:
        daily.run(["--bogus"])
    assert err.value.code == 2
    assert daily.make_parser().parse_args([]).pregame is False
