"""
Fetch NBA games for a date range from free, no-key sources.

Primary:  NBA.com's public season schedule JSON (uses NBA team ids, same as priors.json)
Fallback: ESPN's public scoreboard API (team abbreviations mapped back to NBA ids)

Every game comes back in the same shape:
    {
        "game_id": str,
        "date": "YYYY-MM-DD",          # US Eastern game date
        "tipoff_utc": "2026-11-18T00:30:00Z" or None,
        "status": "scheduled" | "live" | "final",
        "status_text": str,
        "home": {"id": "1610612738", "abbr": "BOS", "name": "Boston Celtics", "score": int | None},
        "away": {...same...},
    }
"""
from datetime import date, datetime, timedelta

import requests

NBA_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

HEADERS = {"User-Agent": "Mozilla/5.0 (nba-prediction-model)"}

# NBA.com team id -> (abbreviation, full name)
NBA_TEAMS = {
    "1610612737": ("ATL", "Atlanta Hawks"),
    "1610612738": ("BOS", "Boston Celtics"),
    "1610612751": ("BKN", "Brooklyn Nets"),
    "1610612766": ("CHA", "Charlotte Hornets"),
    "1610612741": ("CHI", "Chicago Bulls"),
    "1610612739": ("CLE", "Cleveland Cavaliers"),
    "1610612742": ("DAL", "Dallas Mavericks"),
    "1610612743": ("DEN", "Denver Nuggets"),
    "1610612765": ("DET", "Detroit Pistons"),
    "1610612744": ("GSW", "Golden State Warriors"),
    "1610612745": ("HOU", "Houston Rockets"),
    "1610612754": ("IND", "Indiana Pacers"),
    "1610612746": ("LAC", "LA Clippers"),
    "1610612747": ("LAL", "Los Angeles Lakers"),
    "1610612763": ("MEM", "Memphis Grizzlies"),
    "1610612748": ("MIA", "Miami Heat"),
    "1610612749": ("MIL", "Milwaukee Bucks"),
    "1610612750": ("MIN", "Minnesota Timberwolves"),
    "1610612740": ("NOP", "New Orleans Pelicans"),
    "1610612752": ("NYK", "New York Knicks"),
    "1610612760": ("OKC", "Oklahoma City Thunder"),
    "1610612753": ("ORL", "Orlando Magic"),
    "1610612755": ("PHI", "Philadelphia 76ers"),
    "1610612756": ("PHX", "Phoenix Suns"),
    "1610612757": ("POR", "Portland Trail Blazers"),
    "1610612758": ("SAC", "Sacramento Kings"),
    "1610612759": ("SAS", "San Antonio Spurs"),
    "1610612761": ("TOR", "Toronto Raptors"),
    "1610612762": ("UTA", "Utah Jazz"),
    "1610612764": ("WAS", "Washington Wizards"),
}
ABBR_TO_ID = {abbr: team_id for team_id, (abbr, _) in NBA_TEAMS.items()}
# ESPN uses a few different abbreviations
ABBR_TO_ID.update({"GS": "1610612744", "NY": "1610612752", "NO": "1610612740",
                   "SA": "1610612759", "UTAH": "1610612762", "WSH": "1610612764",
                   "PHO": "1610612756", "BRK": "1610612751"})

_STATUS = {1: "scheduled", 2: "live", 3: "final"}


def _team(team_id, score=None, status="scheduled"):
    abbr, name = NBA_TEAMS.get(str(team_id), ("???", f"Team {team_id}"))
    return {"id": str(team_id), "abbr": abbr, "name": name,
            "score": int(score) if status != "scheduled" and score is not None else None}


def fetch_from_nba(start: date, end: date):
    resp = requests.get(NBA_SCHEDULE_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    game_dates = resp.json()["leagueSchedule"]["gameDates"]

    games = []
    for day in game_dates:
        # gameDate looks like "11/17/2026 00:00:00"
        game_day = datetime.strptime(day["gameDate"].split(" ")[0], "%m/%d/%Y").date()
        if not (start <= game_day <= end):
            continue
        for g in day["games"]:
            home, away = g["homeTeam"], g["awayTeam"]
            # skip placeholder games (e.g. Cup knockout slots with no teams yet)
            if not home.get("teamId") or not away.get("teamId"):
                continue
            status = _STATUS.get(g.get("gameStatus"), "scheduled")
            games.append({
                "game_id": g["gameId"],
                "date": game_day.isoformat(),
                "tipoff_utc": g.get("gameDateTimeUTC"),
                "status": status,
                "status_text": g.get("gameStatusText", ""),
                "home": _team(home["teamId"], home.get("score"), status),
                "away": _team(away["teamId"], away.get("score"), status),
            })
    return games


def fetch_from_espn(start: date, end: date):
    games = []
    day = start
    while day <= end:
        resp = requests.get(ESPN_SCOREBOARD_URL, params={"dates": day.strftime("%Y%m%d")},
                            headers=HEADERS, timeout=20)
        resp.raise_for_status()
        for event in resp.json().get("events", []):
            comp = event["competitions"][0]
            state = comp["status"]["type"]["state"]  # pre / in / post
            status = {"pre": "scheduled", "in": "live", "post": "final"}.get(state, "scheduled")
            sides = {}
            for c in comp["competitors"]:
                team_id = ABBR_TO_ID.get(c["team"]["abbreviation"].upper())
                if team_id is None:
                    break
                sides[c["homeAway"]] = _team(team_id, c.get("score"), status)
            if len(sides) != 2:
                continue
            games.append({
                "game_id": event["id"],
                "date": day.isoformat(),
                "tipoff_utc": event.get("date"),
                "status": status,
                "status_text": comp["status"]["type"].get("shortDetail", ""),
                "home": sides["home"],
                "away": sides["away"],
            })
        day += timedelta(days=1)
    return games


def fetch_games(start: date, days: int = 7):
    """Games from `start` through the next `days - 1` days, sorted by tipoff."""
    end = start + timedelta(days=days - 1)
    try:
        games = fetch_from_nba(start, end)
        source = "NBA.com"
    except Exception as err:  # network error, format change, etc.
        print(f"NBA.com schedule failed ({err}); trying ESPN...")
        games = fetch_from_espn(start, end)
        source = "ESPN"
    games.sort(key=lambda g: (g["date"], g["tipoff_utc"] or ""))
    return games, source
