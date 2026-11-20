"""
Daily automatic run (Phase 11): one entry point for Windows Task Scheduler (scripts/install_tasks.ps1).

    python code/daily.py                 # morning run: apply finished games, morning lines snapshot, rebuild the site
    python code/daily.py --pregame       # closing-line snapshot ~1 hour before the day's first tip-off (when due)
    python code/daily.py --dry-run       # what a run would do now, from local files only: no network, writes nothing
    python code/daily.py --pregame --dry-run

The live model as in params.py; this file only schedules existing steps and never changes the model,
the update logic or any read-only preseason priors file.

Morning run (default), in order
-------------------------------
  1. update   update.run_update with the live defaults (= update.py --update): applies every finished,
              not-yet-applied game to priors.json / applied_games.json and logs its prediction to
              prediction_log.csv. Its betting-lines snapshot is taken as step 2 instead (lines="off"
              here), so a lines failure can't hold up the update and vice versa.
  2. lines    one Odds API snapshot of upcoming games (lines.fetch_lines = python code/lines.py): the
              morning line. Once per US Eastern day (state in logs/daily_state.json).
  3. site     build_site.py for today + 6 days, from priors.json as it now is: if step 1 failed, the
              site is still rebuilt from the current (unchanged) priors.

Pre-game run (--pregame)
------------------------
Only step 2, once per Eastern day, for closing lines (lines.closing_line = the last snapshot before tip-off).
The run time comes from that day's NBA.com schedule (the cached stats/schedules/2026-27.json; downloaded
if missing): PREGAME_LEAD (1 hour) before the first tip-off, Eastern time; 6 PM ET when the schedule
can't be read or has no tip-off times (TBD). The scheduled task fires every 15 minutes through the
day; a run before that time exits at once ("not due"), the first run after it takes the snapshot
(so it also catches up after sleep). Days without games, and days whose games have all started, are
skipped without a request.

Odds API credits (free plan: 500 a month; one snapshot = 3)
-----------------------------------------------------------
At most one morning + one pre-game snapshot per Eastern day: 6 credits a day, <= 186 a month. The
credits left (the API's x-requests-remaining) are printed after every request and kept in the state
file. Before a request: if this month's last count is under 3, it is skipped (one attempt a day is
still allowed, in case the credits were reset); a pre-game snapshot is also skipped when it would
leave fewer than 3 credits for each remaining day's morning snapshot. --force-lines skips the
once-a-day check (not the credit check).

ODDS_API_KEY comes from the environment; if it isn't there (a scheduled task started before you set
it), it is read from your user environment in the registry (HKCU\\Environment). It is never printed
or written anywhere.

Unattended safety
-----------------
  * Lock: logs/daily.lock is held (an OS file lock, released even if the process dies) for the whole
    run; a second run while one is going exits at once with EXIT_LOCKED.
  * Retries: NBA.com requests are already retried per request (boxscores.py: 4 tries); on top of that
    each step is retried (STEP_WAITS: 3 attempts, 1 then 3 minutes apart; the Odds API only for no
    response / 429 / 5xx, never for a bad key or no credits), then skipped with a WARNING.
  * priors.json / applied_games.json are only ever written by update.run_update's crash-safe save
    (write-ahead, one game at a time), so a failure leaves them as they were after the last fully
    applied game: a failure while listing games writes nothing; a game that can't be downloaded is
    skipped and NOT recorded (the next run retries it).
  * Log: everything a run prints is also appended to logs/daily_YYYY-MM-DD.log (local date).
  * No commits: a run only changes local files. The summary prints the `git add` line for the
    tracked data it changed, for when you want to commit that day.

Exit codes: 0 everything done; 1 done with warnings (a step skipped: no key, lines failed, games
skipped, ...); 2 bad command line (argparse); 3 a step failed after its retries (the update, the
site, or the pre-game snapshot); 4 another run holds the lock.
"""
import argparse
import calendar
import hashlib
import io
import json
import os
import sys
import time
import traceback
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"
LOCK_NAME = "daily.lock"
STATE_NAME = "daily_state.json"
EASTERN = ZoneInfo("America/New_York")
PREGAME_LEAD = timedelta(hours=1)
FALLBACK_PREGAME = dtime(18, 0)            # Eastern, when the schedule has no usable tip-off time
STEP_WAITS = (60, 180)                     # seconds between a step's attempts (3 attempts)
ODDS_COST = 3                              # credits per snapshot: markets (3) x regions (1)

EXIT_OK, EXIT_WARNINGS, EXIT_FAILED, EXIT_LOCKED = 0, 1, 3, 4



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# lock, log, state
# ---------------------------------------------------------------------------

class RunLock:
    """An exclusive OS lock on a file (msvcrt on Windows, flock elsewhere); released when the process ends."""

    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def acquire(self, note: bool = True) -> bool:
        """True if this process now holds the lock. note: write the owner (pid, time) after the locked byte."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        if note:
            os.lseek(fd, 1, os.SEEK_SET)   # byte 0 is the locked one; the owner's details go after it
            os.write(fd, f" pid {os.getpid()} since {_now().isoformat(timespec='seconds')}\n".encode())
        return True

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class Tee(io.TextIOBase):
    """Writes to the log file and to the console (if there is one: pythonw has none)."""

    def __init__(self, log, console):
        self.log, self.console = log, console

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        if self.console is not None:
            try:
                self.console.write(text)
            except UnicodeEncodeError:
                enc = getattr(self.console, "encoding", None) or "ascii"
                self.console.write(text.encode(enc, "replace").decode(enc))
            self.console.flush()
        return len(text)

    def flush(self):
        if not self.log.closed:
            self.log.flush()
        if self.console is not None and not getattr(self.console, "closed", False):
            self.console.flush()


def load_state(log_dir) -> dict:
    path = Path(log_dir) / STATE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(log_dir, state: dict) -> None:
    path = Path(log_dir) / STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def data_files() -> list:
    """The tracked data a run can change (site/, stats/ and logs/ are gitignored), at the modules' own paths."""
    import lines
    import update
    return [Path(p) for p in (update.PRIORS_PATH, update.APPLIED_GAMES_PATH, update.LOG_PATH, lines.LINES_PATH,
                              lines.UNMATCHED_PATH)]


def file_hashes() -> dict:
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in data_files()}


def repo_name(path) -> str:
    """A path as `git add` wants it: relative to the repo with forward slashes (absolute if outside it)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# pre-game time
# ---------------------------------------------------------------------------

def day_games(raw: dict, day: date) -> list:
    """The schedule's games on Eastern date `day` with both teams known and not postponed."""
    out = []
    for d in raw["leagueSchedule"]["gameDates"]:
        for g in d["games"]:
            if not str(g.get("gameDateEst", "")).startswith(day.isoformat()):
                continue
            if not ((g.get("homeTeam") or {}).get("teamId") and (g.get("awayTeam") or {}).get("teamId")):
                continue
            if g.get("postponedStatus") == "Y" or "PPD" in str(g.get("gameStatusText", "")).upper():
                continue
            out.append(g)
    return out


def tipoff(g: dict):
    """A game's tip-off (UTC datetime), or None when NBA.com hasn't set it (TBD: shown as midnight Eastern)."""
    text = g.get("gameDateTimeUTC")
    if not text or str(g.get("gameStatusText", "")).strip().upper() == "TBD":
        return None
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    return None if t.astimezone(EASTERN).time() == dtime(0, 0) else t


def pregame_plan(raw, day: date) -> dict:
    """
    When the pre-game snapshot of Eastern date `day` is due: {"games", "first", "last", "due", "why"}.
    raw None (no schedule): due at FALLBACK_PREGAME. No games that day: due None. Otherwise PREGAME_LEAD
    before the first known tip-off, or FALLBACK_PREGAME when every time is TBD. Times are Eastern.
    """
    fallback = datetime.combine(day, FALLBACK_PREGAME, tzinfo=EASTERN)
    if raw is None:
        return {"games": None, "first": None, "last": None, "due": fallback, "why": "no schedule: 6 PM ET fallback"}
    games = day_games(raw, day)
    if not games:
        return {"games": 0, "first": None, "last": None, "due": None, "why": "no games"}
    tips = sorted(t.astimezone(EASTERN) for t in (tipoff(g) for g in games) if t is not None)
    if not tips:
        return {"games": len(games), "first": None, "last": None, "due": fallback,
                "why": "tip-off times TBD: 6 PM ET fallback"}
    return {"games": len(games), "first": tips[0], "last": tips[-1], "due": tips[0] - PREGAME_LEAD,
            "why": f"1 hour before the first tip-off ({tips[0]:%H:%M} ET)"}


def cached_schedule():
    """The cached season schedule (stats/schedules/), or None; never downloads."""
    import boxscores
    path = Path(boxscores.SCHEDULE_CACHE_DIR) / f"{boxscores.SEASON}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def current_schedule():
    """The season schedule: the cache, downloaded if missing; None if that fails."""
    import boxscores
    try:
        return boxscores.get_schedule(boxscores.SEASON)
    except Exception as err:   # noqa: BLE001 (a missing schedule only means the 6 PM fallback)
        print(f"WARNING: NBA.com schedule unavailable ({type(err).__name__}); using the 6 PM ET fallback")
        return None


# ---------------------------------------------------------------------------
# the Odds API key and credits
# ---------------------------------------------------------------------------

def _registry_key():
    """ODDS_API_KEY from the user environment in the registry (Windows), or None."""
    if os.name != "nt":
        return None
    import winreg
    import lines
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, lines.KEY_ENV)[0]).strip() or None
    except OSError:
        return None


def ensure_odds_key() -> bool:
    """True when ODDS_API_KEY is available (copied from the user registry into this process if needed)."""
    import lines
    if lines.api_key():
        return True
    value = _registry_key()
    if value:
        os.environ[lines.KEY_ENV] = value
        print(f"{lines.KEY_ENV} taken from your user environment (registry)")
        return True
    return False


def days_left_in_month(day: date) -> int:
    return calendar.monthrange(day.year, day.month)[1] - day.day


def odds_budget(state: dict, kind: str, today: date) -> tuple:
    """
    (allowed, reason) for one snapshot. kind "morning" or "pregame". Uses the last credit count seen
    (state["quota"]); a count from an earlier month is ignored (the credits reset monthly).
    """
    q = state.get("quota") or {}
    seen = str(q.get("at", ""))[:10]
    try:
        remaining = int(q["remaining"])
    except (KeyError, TypeError, ValueError):
        return True, "credits left unknown"
    if seen[:7] != today.isoformat()[:7]:
        return True, f"last count ({remaining}) is from an earlier month"
    if remaining < ODDS_COST:
        if seen == today.isoformat():
            return False, f"only {remaining} Odds API credits left this month"
        return True, f"{remaining} credits left at the last count; trying once today in case they were reset"
    if kind == "pregame":
        reserve = ODDS_COST * days_left_in_month(today)
        if remaining - ODDS_COST < reserve:
            return False, (f"{remaining} credits left: keeping {reserve} for the morning snapshots of the "
                           f"{days_left_in_month(today)} days left this month")
    return True, f"{remaining} credits left"


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, mode: str, log_dir, waits=STEP_WAITS, force_lines: bool = False):
        self.mode, self.log_dir, self.waits, self.force_lines = mode, Path(log_dir), tuple(waits), force_lines
        self.warnings, self.failed = [], []
        self.summary = {"mode": mode, "applied": None, "skipped_games": [], "lines_saved": None, "credits_left": None,
                        "site_games": None}
        self.state = load_state(self.log_dir)
        self.now = _now()
        self.today = self.now.astimezone(EASTERN).date()

    def warn(self, text):
        self.warnings.append(text)
        print(f"WARNING: {text}")

    def attempts(self, what, fn, retry_if=lambda err: True):
        """fn() with up to len(waits) retries; (True, result) or (False, last error)."""
        for i in range(len(self.waits) + 1):
            try:
                return True, fn()
            except KeyboardInterrupt:
                raise
            except Exception as err:   # noqa: BLE001 (a step's failure must never stop the run)
                print(f"{what}: attempt {i + 1} failed: {type(err).__name__}: {redact(err)}")
                traceback.print_exc(file=sys.stdout)
                if i == len(self.waits) or not retry_if(err):
                    return False, err
                print(f"{what}: retrying in {self.waits[i]:.0f} s")
                _sleep(self.waits[i])

    # -- steps --------------------------------------------------------------

    def step_update(self) -> bool:
        import update
        print(f"\n--- update ({_stamp()}) ---")
        ok, result = self.attempts("update", lambda: update.run_update(lines="off"))
        if not ok:
            try:
                update.load_applied()   # resolves a pending entry left by an interrupted save, as --update would
            except Exception:  # noqa: BLE001
                pass
            self.failed.append("update")
            self.warn(f"update failed after {len(self.waits) + 1} attempts ({type(result).__name__}); "
                      "priors.json / applied_games.json keep every game applied before the failure; the site is "
                      "rebuilt from them")
            return False
        self.summary["applied"] = len(result["applied"])
        self.summary["skipped_games"] = result["skipped"]
        if result["skipped"]:
            self.warn(f"{len(result['skipped'])} game(s) skipped (box score not ready); the next run retries them")
        return True

    def step_lines(self, kind: str) -> bool:
        """One Odds API snapshot; kind "morning" or "pregame". False only when the request itself failed."""
        import lines
        print(f"\n--- {kind} lines ({_stamp()}) ---")
        done_key = f"{kind}_lines"
        if self.state.get(done_key) == self.today.isoformat() and not self.force_lines:
            print(f"{kind} lines already saved today ({self.today}); skipped (--force-lines to take another)")
            return True
        if not ensure_odds_key():
            self.warn(f"{lines.KEY_ENV} is not set (environment or user registry); {kind} lines not saved")
            return True
        allowed, reason = odds_budget(self.state, kind, self.today)
        if not allowed:
            self.warn(f"{kind} lines skipped: {reason}")
            return True
        print(f"Odds API budget: {reason}")
        ok, result = self.attempts(f"{kind} lines", lines.fetch_lines,
                                   retry_if=lambda e: not isinstance(e, lines.OddsAPIError) or e.retryable)
        if not ok:
            if isinstance(result, lines.MissingKeyError):
                self.warn(f"{kind} lines not saved: {lines.KEY_ENV} is not set")
                return True
            self.warn(f"{kind} lines not saved: {redact(result)}")
            return False
        q = result.get("quota") or {}
        self.summary["lines_saved"] = result["saved"]
        self.summary["credits_left"] = q.get("remaining")
        self.state[done_key] = self.today.isoformat()
        if q.get("remaining") is not None:
            self.state["quota"] = {"remaining": q.get("remaining"), "used": q.get("used"),
                                   "at": self.today.isoformat()}
        save_state(self.log_dir, self.state)
        if result["unmatched"]:
            self.warn(f"{len(result['unmatched'])} betting line(s) not matched to an NBA.com game")
        return True

    def step_site(self) -> bool:
        import build_site
        print(f"\n--- site ({_stamp()}) ---")
        ok, result = self.attempts("site", lambda: build_site.main([]))
        if not ok:
            self.failed.append("site")
            self.warn(f"site not rebuilt ({type(result).__name__}); the previous site/ stays")
            return False
        self.summary["site_games"] = (result or {}).get("games")
        return True

    def pregame(self):
        """The pre-game snapshot if it's due (True when this run did something worth a summary)."""
        if self.state.get("pregame_lines") == self.today.isoformat() and not self.force_lines:
            print(f"pre-game lines already handled today ({self.today})")
            return False
        plan = pregame_plan(current_schedule(), self.today)
        now_et = self.now.astimezone(EASTERN)
        if plan["due"] is None:
            print(f"pre-game: no games on {self.today}; nothing to do")
            self.state["pregame_lines"] = self.today.isoformat()
            save_state(self.log_dir, self.state)
            return False
        if now_et < plan["due"]:
            print(f"pre-game: not due until {plan['due']:%H:%M} ET ({plan['why']}); nothing to do")
            return False
        if plan["last"] is not None and now_et >= plan["last"]:
            self.warn(f"pre-game snapshot missed: every game on {self.today} has started (last tip-off "
                      f"{plan['last']:%H:%M} ET); no request")
            self.state["pregame_lines"] = self.today.isoformat()
            save_state(self.log_dir, self.state)
            return True
        print(f"pre-game: due since {plan['due']:%H:%M} ET ({plan['why']}, {plan['games']} game(s))")
        if not self.step_lines("pregame"):
            self.failed.append("pregame lines")   # not marked handled: the next trigger tries again
        elif self.state.get("pregame_lines") != self.today.isoformat():
            # skipped for a reason that won't change today (no key, no credits): don't warn every 15 minutes
            self.state["pregame_lines"] = self.today.isoformat()
            save_state(self.log_dir, self.state)
        return True

    def morning(self):
        self.step_update()
        self.step_lines("morning")
        self.step_site()
        return True

    def exit_code(self) -> int:
        if self.failed:
            return EXIT_FAILED
        return EXIT_WARNINGS if self.warnings else EXIT_OK


def redact(err) -> str:
    try:
        import lines
        return lines.redact(err)
    except Exception:  # noqa: BLE001
        return str(err)


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def git_add_line(before: dict, after: dict) -> str:
    changed = [repo_name(p) for p in after if before.get(p) != after[p] and after[p] is not None]
    return f"git add {' '.join(changed)}" if changed else ""


def print_summary(run: Run, before: dict, after: dict, code: int) -> None:
    s = run.summary
    print(f"\n=== summary ({run.mode}, {_stamp()}) ===")
    if run.mode == "daily":
        applied = "failed" if s["applied"] is None else s["applied"]
        print(f"games applied: {applied}" + (f" ({len(s['skipped_games'])} skipped)" if s["skipped_games"] else ""))
    lines_saved = "none" if s["lines_saved"] is None else s["lines_saved"]
    print(f"lines saved: {lines_saved}" + (f"; Odds API credits left: {s['credits_left']}" if s["credits_left"] else ""))
    if run.mode == "daily":
        print(f"site games: {'not rebuilt' if s['site_games'] is None else s['site_games']}")
    print(f"warnings: {len(run.warnings)}")
    for w in run.warnings:
        print(f"  - {w}")
    line = git_add_line(before, after)
    print(f"to commit today's data (optional, by hand): {line}" if line else "no tracked data changed")
    print(f"exit code {code}")


# ---------------------------------------------------------------------------
# dry run: local files only
# ---------------------------------------------------------------------------

def dry_run(mode: str, log_dir, force_lines: bool = False) -> int:
    import boxscores
    import lines
    from schedule import NBA_TEAMS

    state = load_state(log_dir)
    now = _now()
    today = now.astimezone(EASTERN).date()
    raw = cached_schedule()
    print(f"DRY RUN ({mode}) at {now.astimezone(EASTERN):%Y-%m-%d %H:%M} ET: no network, nothing written")
    lock_free = not _lock_held(Path(log_dir) / LOCK_NAME)
    print(f"  lock {Path(log_dir) / LOCK_NAME}: {'free' if lock_free else 'HELD by another run (a real run would exit 4)'}")
    key = bool(lines.api_key() or _registry_key())
    print(f"  {lines.KEY_ENV}: {'set' if key else 'NOT set (lines would be skipped)'}")
    if mode == "daily":
        import update
        applied_path = Path(update.APPLIED_GAMES_PATH)
        applied = json.loads(applied_path.read_text(encoding="utf-8")) if applied_path.exists() else {}
        todo = []
        if raw is not None:
            for d in raw["leagueSchedule"]["gameDates"]:
                for g in d["games"]:
                    gid = g["gameId"]
                    if (g.get("gameStatus") == 3 and gid[:3] in ("002", "006") and gid not in applied
                            and str(g["gameDateEst"])[:10] >= boxscores.OPENING_NIGHT
                            and str((g.get("homeTeam") or {}).get("teamId")) in NBA_TEAMS):
                        todo.append(gid)
        print(f"  1. update: {len(applied)} games applied so far; {len(todo)} finished game(s) not yet applied in "
              f"the cached schedule (the real run asks NBA.com, so it may find more)")
        done = state.get("morning_lines") == today.isoformat() and not force_lines
        allowed, reason = odds_budget(state, "morning", today)
        print(f"  2. morning lines: " + ("already saved today, skip" if done else
                                         (f"1 request ({reason})" if allowed else f"skip: {reason}")))
        print("  3. site: rebuild site/ for today + 6 days from priors.json")
    else:
        plan = pregame_plan(raw, today)
        when = "no games today" if plan["due"] is None else f"due at {plan['due']:%H:%M} ET ({plan['why']})"
        done = state.get("pregame_lines") == today.isoformat() and not force_lines
        allowed, reason = odds_budget(state, "pregame", today)
        now_et = now.astimezone(EASTERN)
        action = ("already handled today" if done else "nothing (no games)" if plan["due"] is None
                  else "nothing yet (not due)" if now_et < plan["due"]
                  else "skip: every game has started" if plan["last"] and now_et >= plan["last"]
                  else f"1 request ({reason})" if allowed else f"skip: {reason}")
        print(f"  pre-game snapshot: {when}; now: {action}")
    q = state.get("quota") or {}
    print(f"  Odds API credits left at the last count: {q.get('remaining', 'unknown')}"
          + (f" ({q.get('at')})" if q.get("at") else ""))
    return EXIT_OK


def _lock_held(path: Path) -> bool:
    """True if another process holds the lock (checked without creating the file)."""
    if not path.exists():
        return False
    lock = RunLock(path)
    if lock.acquire(note=False):
        lock.release()
        return False
    return True


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily automatic run: update, lines, site (Phase 11).")
    parser.add_argument("--pregame", action="store_true",
                        help="only the pre-game lines snapshot (when due: 1 h before the first tip-off, else 6 PM ET)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what a run would do now from local files only; no network, writes nothing")
    parser.add_argument("--force-lines", action="store_true",
                        help="take a lines snapshot even if one was already taken today (credits still checked)")
    return parser


def run(argv=None, log_dir=None, waits=STEP_WAITS) -> int:
    args = make_parser().parse_args(argv)
    mode = "pregame" if args.pregame else "daily"
    log_dir = Path(log_dir or LOG_DIR)
    if args.dry_run:
        return dry_run(mode, log_dir, args.force_lines)

    log_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    log_path = log_dir / f"daily_{started.astimezone().date().isoformat()}.log"
    console_out, console_err = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as log:
        tee = Tee(log, console_out)
        sys.stdout, sys.stderr = tee, Tee(log, console_err)
        try:
            print(f"\n=== {mode} run, {started.astimezone():%Y-%m-%d %H:%M:%S} local "
                  f"({started.astimezone(EASTERN):%H:%M} ET), pid {os.getpid()} ===")
            lock = RunLock(log_dir / LOCK_NAME)
            if not lock.acquire():
                print(f"another run holds {lock.path}; exiting (code {EXIT_LOCKED})")
                return EXIT_LOCKED
            with lock:
                before = file_hashes()
                r = Run(mode, log_dir, waits, args.force_lines)
                try:
                    did = r.morning() if mode == "daily" else r.pregame()
                except Exception as err:   # noqa: BLE001 (last resort: report it, never leave silently)
                    traceback.print_exc(file=sys.stdout)
                    r.failed.append("run")
                    r.warn(f"unexpected error: {type(err).__name__}: {redact(err)}")
                    did = True
                code = r.exit_code()
                if did:
                    print_summary(r, before, file_hashes(), code)
                return code
        finally:
            sys.stdout, sys.stderr = console_out, console_err


def main():
    sys.exit(run())


if __name__ == "__main__":
    main()
