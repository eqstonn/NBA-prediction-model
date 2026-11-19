"""
Rest days between games (Phase 9), from NBA.com's season schedules (ScheduleLeagueV2, cached
in stats/schedules/<season>.json by boxscores.get_schedule).

For every game and each of its two teams:

    days       full days off since the team's previous game in the same season
               (previous game yesterday -> 0); None for the team's first game of the season
    rest       "b2b" (0 days: the second night of a back-to-back), "1" (1 day), "2+" (2 or more,
               and every season opener)
    traveled   back-to-backs only: True when last night's game was in a different arena
               (arenaName) than tonight's, False when both were in the same arena; None otherwise
    opener     True for the team's first game of the season

A team's games are its regular-season (incl. NBA Cup group / knockout games), play-in,
playoff and NBA Cup final games: gameId types 002, 005, 004, 006. Preseason (001) and
All-Star (003) games don't count, so an opener is always "2+". Games from another season
never count either: the previous game is looked for in the same season (boxscores.season_of),
so the long summer gap and a season opener are both "2+". Scheduled games count too (rest of
an upcoming game depends on the games before it); postponed ones ("postponedStatus" "Y") and
games whose teams aren't known yet (Cup knockouts) don't.

Both the Kaggle training rows and NBA.com agree on every training game's date (checked when
Phase 9 was built: 0 mismatches over 7,926 rows), so one source serves training and 2026-27.
"""
from datetime import date

import boxscores

CATEGORIES = ("b2b", "1", "2+")
BASELINE = "2+"
COUNTED_TYPES = ("002", "004", "005", "006")


def category(days) -> str:
    """Rest category from full days off (None = season opener)."""
    if days is None or days >= 2:
        return "2+"
    if days < 0:
        raise ValueError(f"negative rest ({days} days): two games on the same date")
    return "b2b" if days == 0 else "1"


def schedule_games(raw: dict) -> list:
    """The counted games of one ScheduleLeagueV2 response: [{"game_id", "date", "season", "home", "away", "arena"}]."""
    out = []
    for day in raw["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            gid = g["gameId"]
            home, away = (g.get("homeTeam") or {}).get("teamId"), (g.get("awayTeam") or {}).get("teamId")
            if gid[:3] not in COUNTED_TYPES or not home or not away or g.get("postponedStatus") == "Y":
                continue
            out.append({"game_id": gid, "date": g["gameDateEst"][:10], "season": boxscores.season_of(gid),
                        "home": str(home), "away": str(away),
                        "arena": g.get("arenaName") or g.get("arenaCity") or None})
    return out


def rest_table(games: list) -> dict:
    """{gameId: {teamId: {"days", "rest", "traveled", "opener"}}} for every game in `games` (schedule_games shape)."""
    by_team = {}
    for g in games:
        for team in (g["home"], g["away"]):
            by_team.setdefault(team, []).append(g)
    out = {}
    for team, played in by_team.items():
        played.sort(key=lambda g: (g["date"], g["game_id"]))
        previous = {}   # season -> the team's previous game in it
        for g in played:
            prev = previous.get(g["season"])
            if prev is None:
                info = {"days": None, "rest": BASELINE, "traveled": None, "opener": True}
            else:
                days = (date.fromisoformat(g["date"]) - date.fromisoformat(prev["date"])).days - 1
                if days < 0:
                    raise ValueError(f"team {team} has two games on {g['date']}: {prev['game_id']}, {g['game_id']}")
                traveled = None
                if days == 0 and g["arena"] and prev["arena"]:
                    traveled = g["arena"] != prev["arena"]
                info = {"days": days, "rest": category(days), "traveled": traveled, "opener": False}
            out.setdefault(g["game_id"], {})[team] = info
            previous[g["season"]] = g
    return out


def rest_flags(raw: dict) -> dict:
    """rest_table of one season's ScheduleLeagueV2 response."""
    return rest_table(schedule_games(raw))


def rest_games(game_ids) -> dict:
    """
    {gameId: {teamId: rest info}} for the given games, from the cached schedules of their
    seasons (like boxscores.neutral_games: a season whose cache is missing, or lacks one of
    the games, is downloaded once; a game that still isn't there raises KeyError).
    """
    by_season = {}
    for gid in game_ids:
        by_season.setdefault(boxscores.season_of(gid), []).append(boxscores.game_key(gid))
    out = {}
    for season, ids in by_season.items():
        table = rest_flags(boxscores.get_schedule(season))
        if any(gid not in table for gid in ids):
            table = rest_flags(boxscores.get_schedule(season, refresh=True))
        missing = [gid for gid in ids if gid not in table]
        if missing:
            raise KeyError(f"games not in NBA.com's {season} schedule (or without both teams): {missing[:5]}")
        out.update({gid: table[gid] for gid in ids})
    return out


def rest_on_date(raw: dict, team_id: str, day: str) -> dict:
    """
    Rest info for a team's game on `day` (YYYY-MM-DD) from one season's schedule, for games
    that aren't matched by gameId (the website's ESPN games): the team's game that day if the
    schedule has it, else computed from its last game before `day` (arena unknown).
    """
    games = schedule_games(raw)
    table = rest_table(games)
    team_id = str(team_id)
    mine = sorted((g for g in games if team_id in (g["home"], g["away"])), key=lambda g: (g["date"], g["game_id"]))
    for g in mine:
        if g["date"] == day:
            return table[g["game_id"]][team_id]
    before = [g for g in mine if g["date"] < day and g["season"] == boxscores.season_of(mine[-1]["game_id"])] \
        if mine else []
    if not before:
        return {"days": None, "rest": BASELINE, "traveled": None, "opener": True}
    days = (date.fromisoformat(day) - date.fromisoformat(before[-1]["date"])).days - 1
    return {"days": days, "rest": category(days), "traveled": None, "opener": False}
