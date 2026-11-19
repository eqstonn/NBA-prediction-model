"""
Vegas betting lines for 2026-27 (Phase 10a), from The Odds API (https://the-odds-api.com).

    python code/lines.py                 # save one snapshot of every upcoming game's lines
    python code/lines.py --closing       # print the closing line of every game that has tipped off

The key comes from the ODDS_API_KEY environment variable (never from a file in the repo):

    $env:ODDS_API_KEY = "..."                                                   # this PowerShell session
    [Environment]::SetEnvironmentVariable("ODDS_API_KEY", "...", "User")        # every new session

update.py --update also saves a snapshot (--lines on, the default; skipped with a warning when
there is no key).

One run = ONE request: GET /v4/sports/basketball_nba/odds, regions=us, markets=h2h,spreads,totals.
The Odds API charges markets x regions = 3 credits for it (the free plan has 500 a month). Each
run prints the credits left (the x-requests-remaining response header). No retries, so a failed
run never spends more than one request.

What is saved
-------------
data/lines_2026-27.csv gets one row per upcoming game per run (a "snapshot"), APPENDED: old rows
are never rewritten, so the file holds every line we ever saw, with its fetch time.

    fetched_at      when we asked (UTC, 2026-11-19T15:02:03Z)
    game_id         NBA.com gameId (matched by US Eastern date + home/away team, see below)
    date            US Eastern game date
    tipoff_utc      NBA.com's tip-off time (The Odds API's commence_time if NBA.com has none)
    home, away      NBA.com's home / away team (abbreviations)
    spread_home     consensus spread for the HOME team: negative = home favored (-4.5: home by 4.5)
    total           consensus over/under
    ml_home         consensus moneyline, American odds (-150 / +130)
    ml_away
    home_win_prob   consensus vig-free home win probability from the moneylines
    n_books_spread  books behind each consensus number
    n_books_total
    n_books_ml
    books_updated   the latest bookmaker update time in the snapshot (UTC)
    odds_event_id   The Odds API's event id
    flipped         1 if The Odds API lists the teams the other way round from NBA.com (neutral
                    sites); the numbers above are always from NBA.com's home team's side

Consensus = the MEDIAN across the US books (region "us") that offer the market. Moneylines are
medians of the decimal odds, converted back to American (a median of American odds is
meaningless across +/-100); home_win_prob is the median of each book's vig-free home probability.

Started games (commence time or NBA.com tip-off already passed) are not saved: their odds are
live in-game odds. Games that can't be matched to an NBA.com gameId are printed as warnings and
appended to data/lines_unmatched_2026-27.csv. The raw API response of every run is kept in
stats/odds/<fetched_at>.json (gitignored; it contains no key).

Closing line
------------
A game's closing line is its LAST snapshot fetched before tip-off (closing_line / closing_lines).
How close to tip-off that is depends on when you run this: a morning --update gives the
morning line, so for true closing lines also run `python code/lines.py` shortly before games.

Matching to NBA.com
-------------------
The Odds API names teams in full ("Boston Celtics") and gives a UTC commence time. A game is
matched by its US Eastern date + home team + away team against NBA.com's season schedule
(boxscores.get_schedule, cached in stats/schedules/; refreshed once if something doesn't match).
If that fails, the same date with home/away swapped is tried (neutral-site games can be listed
either way) and the row is marked flipped.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import boxscores
from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
SEASON = boxscores.SEASON
DATA_DIR = REPO_ROOT / "data"
LINES_PATH = DATA_DIR / f"lines_{SEASON}.csv"
UNMATCHED_PATH = DATA_DIR / f"lines_unmatched_{SEASON}.csv"
RAW_DIR = REPO_ROOT / "stats" / "odds"

KEY_ENV = "ODDS_API_KEY"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
TIMEOUT = 30
EASTERN = ZoneInfo("America/New_York")
DEFAULT_LINES_MODE = "on"

FIELDS = ["fetched_at", "game_id", "date", "tipoff_utc", "home", "away", "spread_home", "total", "ml_home",
          "ml_away", "home_win_prob", "n_books_spread", "n_books_total", "n_books_ml", "books_updated",
          "odds_event_id", "flipped"]
UNMATCHED_FIELDS = ["fetched_at", "odds_event_id", "commence_time", "home_team", "away_team", "reason"]

# The Odds API's team names -> NBA.com team ids (NBA.com's full names, plus the ones that differ)
NAME_TO_ID = {name.lower(): team_id for team_id, (_, name) in NBA_TEAMS.items()}
NAME_TO_ID.update({"los angeles clippers": "1610612746"})


class MissingKeyError(RuntimeError):
    """ODDS_API_KEY is not set."""


class OddsAPIError(RuntimeError):
    """The Odds API request failed (the message never contains the key)."""


def api_key():
    """The key from ODDS_API_KEY, or None."""
    return os.environ.get(KEY_ENV, "").strip() or None


def redact(text, key=None) -> str:
    """`text` with the API key (given, or from the environment) replaced, for error messages."""
    text = str(text)
    for secret in {key, api_key()} - {None, ""}:
        text = text.replace(secret, "***")
    return text


# ---------------------------------------------------------------------------
# network (the only function that talks to The Odds API; tests replace it)
# ---------------------------------------------------------------------------

def _download_odds(key: str):
    """(events, quota) from ONE request; quota = {"remaining", "used", "last"} from the response headers."""
    query = {"apiKey": key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "american", "dateFormat": "iso"}
    try:
        resp = requests.get(ODDS_URL, params=query, timeout=TIMEOUT)
    except requests.RequestException as err:
        # `from None`: the original exception's message holds the URL, and the URL holds the key
        raise OddsAPIError(f"request failed ({type(err).__name__}: {redact(err, key)})") from None
    quota = {name: resp.headers.get(f"x-requests-{name}") for name in ("remaining", "used", "last")}
    if resp.status_code != 200:
        try:
            message = resp.json().get("message", "")
        except ValueError:
            message = resp.text[:200]
        raise OddsAPIError(f"HTTP {resp.status_code}: {redact(message, key)} (requests left: {quota['remaining']})")
    return resp.json(), quota


# ---------------------------------------------------------------------------
# odds arithmetic
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1 + odds / 100
    if odds <= -100:
        return 1 + 100 / -odds
    raise ValueError(f"not American odds: {odds}")


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1:
        raise ValueError(f"not decimal odds: {decimal}")
    return (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)


def _median(values, digits):
    return round(statistics.median(values), digits) if values else None


def consensus(event: dict, home_name: str, away_name: str) -> dict:
    """
    Median lines across the event's books, from the side of `home_name` (The Odds API's name of
    NBA.com's home team): spread_home, total, ml_home, ml_away, home_win_prob, n_books_*, books_updated.
    A book counts for a market only if it has a usable number for it.
    """
    spreads, totals, dec_home, dec_away, probs, updated = [], [], [], [], [], []
    for book in event.get("bookmakers") or []:
        if book.get("last_update"):
            updated.append(book["last_update"])
        markets = {m.get("key"): {o.get("name"): o for o in m.get("outcomes") or []} for m in book.get("markets") or []}
        spread = markets.get("spreads", {})
        if isinstance(spread.get(home_name, {}).get("point"), (int, float)):
            spreads.append(float(spread[home_name]["point"]))
        elif isinstance(spread.get(away_name, {}).get("point"), (int, float)):
            spreads.append(-float(spread[away_name]["point"]))
        total = markets.get("totals", {})
        point = (total.get("Over") or total.get("Under") or {}).get("point")
        if isinstance(point, (int, float)):
            totals.append(float(point))
        h2h = markets.get("h2h", {})
        prices = [h2h.get(name, {}).get("price") for name in (home_name, away_name)]
        if all(isinstance(p, (int, float)) and abs(p) >= 100 for p in prices):
            home_dec, away_dec = (american_to_decimal(p) for p in prices)
            dec_home.append(home_dec)
            dec_away.append(away_dec)
            probs.append((1 / home_dec) / (1 / home_dec + 1 / away_dec))
    return {"spread_home": _median(spreads, 2), "total": _median(totals, 2),
            "ml_home": round(decimal_to_american(statistics.median(dec_home))) if dec_home else None,
            "ml_away": round(decimal_to_american(statistics.median(dec_away))) if dec_away else None,
            "home_win_prob": _median(probs, 4),
            "n_books_spread": len(spreads), "n_books_total": len(totals), "n_books_ml": len(probs),
            "books_updated": max(updated) if updated else None}


# ---------------------------------------------------------------------------
# matching to NBA.com game ids
# ---------------------------------------------------------------------------

def parse_utc(text) -> datetime:
    """A UTC time from The Odds API / NBA.com / our CSV ("...Z" or "+00:00")."""
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def eastern_date(text) -> str:
    """The US Eastern date (YYYY-MM-DD) of a UTC time: NBA.com's game date."""
    return parse_utc(text).astimezone(EASTERN).date().isoformat()


def team_id(name):
    """NBA.com team id for one of The Odds API's team names, or None."""
    return NAME_TO_ID.get(str(name).strip().lower())


def schedule_index(raw: dict) -> dict:
    """{(Eastern date, home id, away id): {"game_id", "tipoff_utc"}} for every game with both teams in a ScheduleLeagueV2 response."""
    index = {}
    for day in raw["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            home, away = (g.get("homeTeam") or {}).get("teamId"), (g.get("awayTeam") or {}).get("teamId")
            if home and away:
                index[(g["gameDateEst"][:10], str(home), str(away))] = {
                    "game_id": g["gameId"], "tipoff_utc": g.get("gameDateTimeUTC") or None}
    return index


def match_game(index: dict, day: str, home_id: str, away_id: str):
    """(index entry, flipped) for a game on `day` between the two teams, home/away as given or swapped; (None, None) if neither."""
    if (day, home_id, away_id) in index:
        return index[(day, home_id, away_id)], False
    if (day, away_id, home_id) in index:
        return index[(day, away_id, home_id)], True
    return None, None


def snapshot(events: list, index: dict, fetched_at: datetime):
    """
    (rows, unmatched, started): one FIELDS row per matched upcoming event, one UNMATCHED_FIELDS
    row per event we couldn't match, and the number of events left out because they've started.
    """
    rows, unmatched, started = [], [], 0
    stamp = format_utc(fetched_at)
    for event in events:
        names = (event.get("home_team"), event.get("away_team"))
        commence = event.get("commence_time")
        if commence and parse_utc(commence) <= fetched_at:
            started += 1
            continue

        def miss(reason):
            unmatched.append({"fetched_at": stamp, "odds_event_id": event.get("id"), "commence_time": commence,
                              "home_team": names[0], "away_team": names[1], "reason": reason})

        ids = [team_id(n) for n in names]
        if None in ids:
            miss(f"unknown team name {names[ids.index(None)]!r}")
            continue
        if not commence:
            miss("no commence_time")
            continue
        day = eastern_date(commence)
        game, flipped = match_game(index, day, ids[0], ids[1])
        if game is None:
            miss(f"no NBA.com game on {day} between these teams")
            continue
        if game["tipoff_utc"] and parse_utc(game["tipoff_utc"]) <= fetched_at:
            started += 1
            continue
        home_name, away_name = (names[1], names[0]) if flipped else names
        home_id, away_id = (ids[1], ids[0]) if flipped else ids
        lines = consensus(event, home_name, away_name)
        if not (lines["n_books_spread"] or lines["n_books_total"] or lines["n_books_ml"]):
            miss("no bookmaker odds yet")
            continue
        rows.append({"fetched_at": stamp, "game_id": game["game_id"], "date": day,
                     "tipoff_utc": game["tipoff_utc"] or commence,
                     "home": NBA_TEAMS[home_id][0], "away": NBA_TEAMS[away_id][0], **lines,
                     "odds_event_id": event.get("id"), "flipped": int(flipped)})
    return rows, unmatched, started


# ---------------------------------------------------------------------------
# the CSV (append-only) and closing lines
# ---------------------------------------------------------------------------

def append_rows(path, rows: list, fields=FIELDS) -> int:
    """
    Append `rows` to the CSV at `path` (header first if the file is new). Never rewrites or
    truncates existing rows; refuses a file whose header isn't `fields`. Returns rows written.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8", newline="") as file:
            header = next(csv.reader(file), None)
        if header != list(fields):
            raise ValueError(f"{path.name} has columns {header}, expected {list(fields)}; not appending")
        new = False
    else:
        new = True
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows([{k: "" if row.get(k) is None else row[k] for k in fields} for row in rows])
        file.flush()
        os.fsync(file.fileno())
    return len(rows)


def read_lines(path=None) -> list:
    """Every snapshot row in the lines CSV (strings, as saved); [] if there is no file yet."""
    path = Path(path or LINES_PATH)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def closing_lines(rows=None) -> dict:
    """
    {game_id: its closing line}: for each game, the snapshot with the latest fetched_at strictly
    before tip-off. Tip-off is the tipoff_utc of the game's most recent snapshot (the latest
    schedule we saw). Games with no snapshot before tip-off are left out.
    rows: snapshot rows (read_lines), or a path, or None for data/lines_2026-27.csv.
    """
    if rows is None or isinstance(rows, (str, Path)):
        rows = read_lines(rows)
    by_game = {}
    for row in rows:
        if row.get("game_id"):
            by_game.setdefault(row["game_id"], []).append(row)
    out = {}
    for game_id, snaps in by_game.items():
        snaps = sorted(snaps, key=lambda r: parse_utc(r["fetched_at"]))
        tipoff = parse_utc(snaps[-1]["tipoff_utc"])
        before = [r for r in snaps if parse_utc(r["fetched_at"]) < tipoff]
        if before:
            out[game_id] = before[-1]
    return out


def closing_line(game_id: str, rows=None):
    """The closing line (a snapshot row) of one game, or None if we have no snapshot before its tip-off."""
    return closing_lines(rows).get(game_id)


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _save_raw(events, fetched_at: datetime) -> Path:
    path = Path(RAW_DIR) / f"{fetched_at.astimezone(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events), encoding="utf-8")
    return path


def fetch_lines(path=None, unmatched_path=None, key=None, now=None) -> dict:
    """
    One snapshot: ONE Odds API request, matched to NBA.com gameIds and appended to `path`
    (default data/lines_2026-27.csv); unmatched games to `unmatched_path`. Raises
    MissingKeyError without a key, OddsAPIError if the request fails (nothing is written then).
    Returns {"fetched_at", "events", "saved", "started", "unmatched": [rows], "quota": {...}}.
    """
    path, unmatched_path = Path(path or LINES_PATH), Path(unmatched_path or UNMATCHED_PATH)
    key = key or api_key()
    if not key:
        raise MissingKeyError(f"{KEY_ENV} is not set; get a free key at https://the-odds-api.com and set it "
                              f"(PowerShell: $env:{KEY_ENV} = \"...\")")
    for p, fields in ((path, FIELDS), (unmatched_path, UNMATCHED_FIELDS)):
        append_rows(p, [], fields)   # checks an existing file's columns before spending a request

    events, quota = _download_odds(key)
    fetched_at = (now or _now()).astimezone(timezone.utc).replace(microsecond=0)
    _save_raw(events, fetched_at)

    index = schedule_index(boxscores.get_schedule(SEASON))
    rows, unmatched, started = snapshot(events, index, fetched_at)
    if any(u["reason"].startswith("no NBA.com game") for u in unmatched):
        # the cached schedule may be stale (a rescheduled game, a Cup knockout's teams): refresh once
        index = schedule_index(boxscores.get_schedule(SEASON, refresh=True))
        rows, unmatched, started = snapshot(events, index, fetched_at)

    append_rows(path, rows)
    append_rows(unmatched_path, unmatched, UNMATCHED_FIELDS)
    for u in unmatched:
        print(f"  WARNING: betting line not matched: {u['away_team']} @ {u['home_team']} "
              f"({u['commence_time']}, event {u['odds_event_id']}): {u['reason']}")
    print(f"Betting lines {format_utc(fetched_at)}: {len(events)} events from The Odds API, saved {len(rows)} to "
          f"{path.name}, {started} already started, {len(unmatched)} unmatched. "
          f"Odds API requests left: {quota.get('remaining')} (used {quota.get('used')}, "
          f"this run {quota.get('last')})")
    return {"fetched_at": format_utc(fetched_at), "events": len(events), "saved": len(rows), "started": started,
            "unmatched": unmatched, "quota": quota}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save a snapshot of NBA betting lines from The Odds API.")
    parser.add_argument("--closing", action="store_true",
                        help=f"print the closing line of every game in {LINES_PATH.name} (no request)")
    args = parser.parse_args(argv)
    if args.closing:
        lines = sorted(closing_lines().values(), key=lambda r: (r["date"], r["game_id"]))
        now = datetime.now(timezone.utc)
        for r in lines:
            if parse_utc(r["tipoff_utc"]) <= now:
                print(f"{r['date']}  {r['away']} @ {r['home']}  ({r['game_id']})  spread {r['spread_home']}  "
                      f"total {r['total']}  ML {r['ml_home']}/{r['ml_away']}  as of {r['fetched_at']}")
        return
    try:
        fetch_lines()
    except (MissingKeyError, OddsAPIError) as err:
        print(f"ERROR: {redact(err)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
