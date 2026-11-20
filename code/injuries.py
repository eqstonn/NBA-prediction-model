"""
NBA injury report for the website's warnings (Phase 12). DISPLAY ONLY: the model, the predictions, the
priors and the prediction logs never read any of this.

    python code/injuries.py              # fetch the current report, append it to data/injuries_2026-27.csv,
                                         # print the key players who are Out / Doubtful / Questionable

Sources (free, no login, no key), tried in order:
  1. ESPN   https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries  (JSON, every team with
            an injury; ESPN's status is only "Out" or "Day-To-Day", so a Day-To-Day player's game status is
            read from ESPN's note: "is questionable for Friday's game" -> Questionable, "doubtful",
            "probable" / "expected to play" / "available" -> Probable, "ruled out" / "is out for" -> Out;
            a note about a game already played, "<weekday>'s game" before the fetch date, is ignored)
  2. CBS    https://www.cbssports.com/nba/injuries/  (HTML tables; "Game Time Decision" -> Questionable,
            "Expected to be out ..." / "Out for the season" -> Out)
NBA.com's official report is a PDF every 15 minutes whose text doesn't extract cleanly with a pure-Python
reader, so it isn't used.

Each entry: team (our NBA.com team id, matched by name; unmatched teams are printed and kept with an empty
team_id), player, status (Out, Doubtful, Questionable, Probable, Day-To-Day), the source's own status,
reason, comment and the source's report time. Every fetch is APPENDED to data/injuries_2026-27.csv with
its fetch time (never rewritten), for a later accuracy study. That file is gitignored: ESPN's and CBS's
terms don't allow republishing their content, so it stays on this PC.

Key players
-----------
Minutes per game this season from the NBA.com advanced box scores that update.py --update already caches
(stats/boxscores_advanced/<gameId>.json, every applied game): no extra request. A player is KEY if he is
in his team's top TOP_N (5) by minutes per game or averages MIN_MPG (25) or more; his team is the one of
his latest game. Both thresholds are parameters (key_players, build_site --key-top / --key-mpg).
Injury-report names are matched to box-score names within the team (accents, punctuation and Jr./III
dropped), then league-wide if the name is unique. A player with no minutes this season can't be ranked
and is never key (his absence is already in the team's results).

Warning levels (STRONG / MILD): Out and Doubtful are strong; Questionable is mild, and so is ESPN's plain
Day-To-Day when its note names no game status. Probable gets no warning.
"""
import argparse
import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

import boxscores
import lines
from schedule import NBA_TEAMS

REPO_ROOT = Path(__file__).resolve().parent.parent
SEASON = boxscores.SEASON
INJURIES_PATH = REPO_ROOT / "data" / f"injuries_{SEASON}.csv"
APPLIED_GAMES_PATH = REPO_ROOT / "applied_games.json"   # game dates, to find each player's latest team

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CBS_URL = "https://www.cbssports.com/nba/injuries/"
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-prediction-model)"}
TIMEOUT = 20

TOP_N = 5          # a team's top N players by minutes per game are key
MIN_MPG = 25.0     # ... and so is anyone averaging this many minutes
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Day-To-Day")
STRONG = ("Out", "Doubtful")
MILD = ("Questionable", "Day-To-Day")

FIELDS = ["fetched_at", "source", "source_updated", "team_id", "team", "player", "last_name", "status",
          "source_status", "reason", "comment", "reported_at"]

TEAM_ALIASES = {"la clippers": "1610612746", "los angeles clippers": "1610612746"}


class InjuryFetchError(RuntimeError):
    """No source gave a usable injury report."""


# ---------------------------------------------------------------------------
# network (the only functions that go online; tests replace them)
# ---------------------------------------------------------------------------

def _download_espn() -> dict:
    resp = requests.get(ESPN_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _download_cbs() -> str:
    resp = requests.get(CBS_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# names, teams, statuses
# ---------------------------------------------------------------------------

def _plain(text) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def team_id_for(name):
    """Our NBA.com team id for a team name ("Boston Celtics", "LA Clippers", "boston-celtics"), or None."""
    key = re.sub(r"[^a-z0-9 ]", " ", _plain(name).replace("-", " "))
    key = re.sub(r"\s+", " ", key).strip()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    for team_id, (_, full) in NBA_TEAMS.items():
        if _plain(full) == key:
            return team_id
    return None


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name) -> str:
    """A player name for matching: no accents, punctuation or Jr./III, lower case ("T.J. McConnell" -> "tj mcconnell")."""
    text = re.sub(r"[.'`’]", "", _plain(name))
    words = [w for w in re.split(r"[^a-z0-9]+", text) if w and w not in SUFFIXES]
    return " ".join(words)


def last_name(full_name: str) -> str:
    """"Jayson Tatum" -> "Tatum", "Jimmy Butler III" -> "Butler", "Gary Trent Jr." -> "Trent"."""
    words = [w for w in str(full_name).split() if w.strip(".").lower() not in SUFFIXES]
    return " ".join(words[1:]) if len(words) > 1 else (words[0] if words else "")


def normalize_status(text) -> str:
    """A source status as one of STATUSES (unknown text is kept as given)."""
    t = str(text or "").strip().lower()
    if t.startswith("out") or "out for" in t or "expected to be out" in t or t in ("o", "ofs", "suspension"):
        return "Out"
    for s in ("Doubtful", "Questionable", "Probable"):
        if s.lower() in t:
            return s
    if "day-to-day" in t or "day to day" in t or t in ("dtd", "gtd"):
        return "Day-To-Day"
    if "game time decision" in t:
        return "Questionable"
    return str(text or "").strip()


COMMENT_STATUS = (
    (re.compile(r"\b(ruled out|(is|been|listed as|remains) out\b|will not play|won't play|will miss|is out for)"), "Out"),
    (re.compile(r"\bdoubtful\b"), "Doubtful"),
    (re.compile(r"\bquestionable\b|game-time decision"), "Questionable"),
    (re.compile(r"\bprobable\b|expected to play|is available|will play\b|cleared to play"), "Probable"),
)


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
EASTERN = lines.EASTERN


def stale_note(comment: str, reported_at: str, fetched_at: str) -> bool:
    """
    True when the note is about a game that is already over: it names "<weekday>'s game" and the nearest
    such weekday to the note's date (within 3 days either way, US Eastern: ESPN's date is when the note
    was last updated, which can be the day after the game) falls before the fetch date. Unknown: not stale.
    """
    m = re.search(r"\b(" + "|".join(WEEKDAYS) + r")(?:'|’)?s?\b", _plain(comment))
    if not m or not reported_at or not fetched_at:
        return False
    try:
        noted = lines.parse_utc(reported_at).astimezone(EASTERN).date()
        fetched = lines.parse_utc(fetched_at).astimezone(EASTERN).date()
    except ValueError:
        return False
    offset = (WEEKDAYS.index(m.group(1)) - noted.weekday() + 3) % 7 - 3      # -3 .. +3 days
    return noted.toordinal() + offset < fetched.toordinal()


def refine_status(status: str, comment: str, reported_at: str = "", fetched_at: str = "") -> str:
    """
    ESPN's plain Day-To-Day made game-specific from its note ("is questionable for Friday's game"), unless
    the note is about a game that has already been played (stale_note); any other status is kept.
    """
    if status != "Day-To-Day" or stale_note(comment, reported_at, fetched_at):
        return status
    text = _plain(comment)
    for pattern, refined in COMMENT_STATUS:
        if pattern.search(text):
            return refined
    return status


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _entry(source, fetched_at, updated, team_name, player, status, source_status, reason, comment, reported):
    return {"fetched_at": fetched_at, "source": source, "source_updated": updated, "team_id": team_id_for(team_name),
            "team": team_name, "player": player, "last_name": last_name(player), "status": status,
            "source_status": source_status, "reason": reason, "comment": comment, "reported_at": reported}


def parse_espn(raw: dict, fetched_at: str) -> dict:
    """ESPN's injuries JSON -> {"source", "fetched_at", "updated", "entries", "unmatched_teams"}."""
    if not isinstance(raw, dict) or not isinstance(raw.get("injuries"), list):
        raise ValueError("ESPN injuries: no 'injuries' list in the response")
    entries = []
    for team in raw["injuries"]:
        team_name = team.get("displayName", "")
        for e in team.get("injuries") or []:
            athlete = e.get("athlete") or {}
            player = athlete.get("displayName") or " ".join(
                x for x in (athlete.get("firstName"), athlete.get("lastName")) if x)
            if not player:
                continue
            source_status = e.get("status") or ""
            comment = e.get("shortComment") or ""
            details = e.get("details") or {}
            reason = " ".join(x for x in (details.get("side") if details.get("side") not in (None, "Not Specified") else None,
                                          details.get("type"), details.get("detail")) if x) or ""
            status = refine_status(normalize_status(source_status), comment, e.get("date", ""), fetched_at)
            entries.append(_entry("ESPN", fetched_at, raw.get("timestamp", ""), team_name, player, status,
                                  source_status, reason, comment, e.get("date", "")))
    return _report("ESPN", fetched_at, raw.get("timestamp", ""), entries)


class _CBSParser(HTMLParser):
    """Collects (team name, [cells...]) rows from CBS's injury tables."""

    def __init__(self):
        super().__init__()
        self.rows, self.team = [], None
        self._in_team = self._in_td = self._in_long = False
        self._cells, self._text, self._long = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "span" and "TeamName" in cls:
            self._in_team, self._text = True, []
        elif tag == "a" and "/nba/teams/" in (a.get("href") or "") and self._in_team:
            slug = a["href"].rstrip("/").split("/")[-1]
            self.team = slug
        elif tag == "tr" and "TableBase-bodyTr" in cls:
            self._cells = []
        elif tag == "td":
            self._in_td, self._text, self._long = True, [], []
        elif tag == "span" and "CellPlayerName--long" in cls:
            self._in_long = True

    def handle_endtag(self, tag):
        if tag == "span" and self._in_team:
            self._in_team = False
        elif tag == "span" and self._in_long:
            self._in_long = False
        elif tag == "td" and self._in_td:
            self._in_td = False
            text = " ".join("".join(self._long or self._text).split())
            self._cells.append(text)
        elif tag == "tr" and self._cells:
            if len(self._cells) >= 5 and self.team:
                self.rows.append((self.team, self._cells))
            self._cells = []

    def handle_data(self, data):
        if self._in_td:
            if self._in_long:
                self._long.append(data)
            self._text.append(data)


def parse_cbs(page: str, fetched_at: str) -> dict:
    """CBS's injuries page -> the same report shape as parse_espn."""
    parser = _CBSParser()
    parser.feed(page)
    if not parser.rows:
        raise ValueError("CBS injuries: no injury tables found on the page")
    entries = []
    for slug, cells in parser.rows:
        player, _position, updated, injury, status_text = (html.unescape(c) for c in cells[:5])
        team_name = slug.replace("-", " ").title()
        entries.append(_entry("CBS", fetched_at, "", team_name, player, normalize_status(status_text), status_text,
                              injury, status_text, updated))
    return _report("CBS", fetched_at, "", entries)


def _report(source, fetched_at, updated, entries) -> dict:
    unmatched = sorted({e["team"] for e in entries if not e["team_id"]})
    return {"source": source, "fetched_at": fetched_at, "updated": updated, "entries": entries,
            "unmatched_teams": unmatched}


# ---------------------------------------------------------------------------
# one fetch
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_report(now=None) -> dict:
    """The current report from ESPN, else CBS. Raises InjuryFetchError (with both reasons) if neither works."""
    fetched_at = lines.format_utc((now or _now()).replace(microsecond=0))
    problems = []
    for name, download, parse in (("ESPN", _download_espn, parse_espn), ("CBS", _download_cbs, parse_cbs)):
        try:
            report = parse(download(), fetched_at)
        except Exception as err:   # noqa: BLE001 (any failure: try the next source)
            problems.append(f"{name}: {type(err).__name__}: {err}")
            print(f"WARNING: injury report from {name} failed ({type(err).__name__}: {err})"
                  + ("; trying CBS" if name == "ESPN" else ""))
            continue
        if not report["entries"]:
            problems.append(f"{name}: no entries")
            print(f"WARNING: injury report from {name} is empty" + ("; trying CBS" if name == "ESPN" else ""))
            continue
        return report
    raise InjuryFetchError("no injury report: " + "; ".join(problems))


def save_report(report: dict, path=None) -> int:
    """Append every entry (with its fetch time) to the injuries CSV; returns rows written."""
    return lines.append_rows(Path(path or INJURIES_PATH), report["entries"], FIELDS)


def update_report(path=None, now=None) -> dict:
    """fetch_report + save_report, printing a one-line summary and any unmatched team."""
    report = fetch_report(now)
    saved = save_report(report, path)
    for team in report["unmatched_teams"]:
        print(f"WARNING: injury report team not matched to an NBA team id: {team!r}")
    print(f"Injury report {report['fetched_at']}: {len(report['entries'])} players from {report['source']}"
          + (f" (source updated {report['updated']})" if report["updated"] else "")
          + f", saved {saved} rows to {Path(path or INJURIES_PATH).name}")
    return report


# ---------------------------------------------------------------------------
# key players (minutes per game from the cached NBA.com advanced box scores)
# ---------------------------------------------------------------------------

def parse_minutes(value) -> float:
    """"33:15" -> 33.25; "PT33M15.00S" -> 33.25; empty / None (did not play) -> 0."""
    text = str(value or "").strip()
    if not text:
        return 0.0
    m = re.fullmatch(r"PT(\d+)M([\d.]+)S", text)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    if ":" in text:
        mins, secs = text.split(":", 1)
        return int(mins or 0) + float(secs or 0) / 60
    return float(text)


def season_minutes(cache_dir=None, season: str = SEASON, applied_path=None) -> dict:
    """
    {personId: {"name", "norm", "team_id", "games", "minutes", "mpg"}} from every cached advanced box score
    of `season` (games with minutes > 0 count). team_id = the team of the player's latest game (dates from
    applied_games.json, gameId order otherwise).
    """
    cache = Path(cache_dir or boxscores.ADVANCED_CACHE_DIR)
    try:
        dates = {gid: e.get("date", "") for gid, e in
                 json.loads(Path(applied_path or APPLIED_GAMES_PATH).read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        dates = {}
    files = sorted(cache.glob("*.json"), key=lambda p: (dates.get(p.stem, ""), p.stem)) if cache.exists() else []
    players = {}
    for path in files:
        if boxscores.season_of(path.stem) != season:
            continue
        try:
            box = json.loads(path.read_text(encoding="utf-8"))["boxScoreAdvanced"]
        except (OSError, ValueError, KeyError):
            continue
        for side in ("homeTeam", "awayTeam"):
            team = box.get(side) or {}
            for p in team.get("players") or []:
                minutes = parse_minutes((p.get("statistics") or {}).get("minutes"))
                if minutes <= 0:
                    continue
                name = f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
                rec = players.setdefault(p["personId"], {"name": name, "norm": norm_name(name), "games": 0,
                                                         "minutes": 0.0})
                rec["games"] += 1
                rec["minutes"] += minutes
                rec["team_id"] = str(team.get("teamId"))
    for rec in players.values():
        rec["mpg"] = rec["minutes"] / rec["games"]
    return players


def key_players(minutes: dict, top_n: int = TOP_N, min_mpg: float = MIN_MPG) -> dict:
    """{personId: rank within his team} for every key player: team rank <= top_n by mpg, or mpg >= min_mpg."""
    if top_n < 0 or min_mpg < 0:
        raise ValueError("top_n and min_mpg must not be negative")
    by_team = {}
    for pid, rec in minutes.items():
        by_team.setdefault(rec["team_id"], []).append(pid)
    out = {}
    for pids in by_team.values():
        ranked = sorted(pids, key=lambda p: (-minutes[p]["mpg"], -minutes[p]["minutes"], str(p)))
        for rank, pid in enumerate(ranked, 1):
            if rank <= top_n or minutes[pid]["mpg"] >= min_mpg:
                out[pid] = rank
    return out


def match_player(entry: dict, minutes: dict):
    """The personId of an injury-report entry: same name on the same team, else a unique name league-wide."""
    norm = norm_name(entry["player"])
    same_team = [pid for pid, r in minutes.items() if r["norm"] == norm and r["team_id"] == entry["team_id"]]
    if len(same_team) == 1:
        return same_team[0]
    anywhere = [pid for pid, r in minutes.items() if r["norm"] == norm]
    return anywhere[0] if len(anywhere) == 1 else None


def warning_level(status: str):
    return "strong" if status in STRONG else "mild" if status in MILD else None


def team_warnings(report: dict, minutes: dict, top_n: int = TOP_N, min_mpg: float = MIN_MPG) -> dict:
    """
    {team_id: [warning, ...]} for key players with a strong or mild status; warning = {"player", "last_name",
    "status", "level", "mpg", "rank"}, strong first, then by minutes per game.
    """
    keys = key_players(minutes, top_n, min_mpg)
    out = {}
    for e in report["entries"]:
        level = warning_level(e["status"])
        if not level or not e["team_id"]:
            continue
        pid = match_player(e, minutes)
        if pid is None or pid not in keys:
            continue
        out.setdefault(e["team_id"], []).append({"player": e["player"], "last_name": e["last_name"],
                                                 "status": e["status"], "level": level,
                                                 "mpg": round(minutes[pid]["mpg"], 1), "rank": keys[pid]})
    for warnings in out.values():
        warnings.sort(key=lambda w: (w["level"] != "strong", -w["mpg"]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch the NBA injury report (display only).")
    parser.add_argument("--key-top", type=int, default=TOP_N, help=f"team's top N by minutes are key (default {TOP_N})")
    parser.add_argument("--key-mpg", type=float, default=MIN_MPG, help=f"... or this many minutes per game (default {MIN_MPG:g})")
    args = parser.parse_args(argv)
    report = update_report()
    warnings = team_warnings(report, season_minutes(), args.key_top, args.key_mpg)
    for team_id, ws in sorted(warnings.items(), key=lambda kv: NBA_TEAMS[kv[0]][0]):
        print(f"  {NBA_TEAMS[team_id][0]}: " + ", ".join(f"{w['player']} ({w['status']}, {w['mpg']} mpg)" for w in ws))


if __name__ == "__main__":
    main()
