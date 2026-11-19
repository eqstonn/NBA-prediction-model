"""
Finished 2026-27 games and their four factors from NBA.com (stats.nba.com, via nba_api).

Data sources (all https://stats.nba.com/stats/<endpoint>, through the nba_api library)
--------------------------------------------------------------------------------------
    leaguegamelog          LeagueGameLog          finished games per season type
    scheduleleaguev2       ScheduleLeagueV2       the whole season schedule, with gameStatus 3 = final
    boxscorefourfactorsv3  BoxScoreFourFactorsV3  each game's official four factors
    boxscoreadvancedv3     BoxScoreAdvancedV3     each team's possessions (only with --pace on)

Which games count
-----------------
The list of finished games is the UNION of two sources, because the game log alone is
unreliable: on 2026-11-17 it listed 205, then 204, then 209 games, and it left out a
finished Nov 7 game (DEN @ IND) for hours. Every disagreement between the two lists is
printed as a warning.

  1. LeagueGameLog, season types (checked 2026-11-17):
       "Regular Season"  002xxxxxxx  includes the NBA Cup group stage, quarterfinals and semifinals
       "IST"             the NBA Cup: those same 002 games, plus the Cup final 006xxxxxxx (listed only here)
       "Pre Season" (001), "PlayIn" (005), "Playoffs" (004), "All Star"  -> never requested
  2. ScheduleLeagueV2: games with gameStatus 3 (final). NBA Cup games carry gameLabel
     "Emirates NBA Cup" / gameSubtype "in-season" or "in-season-knockout".

A game is used only if its ID starts with 002 or 006 and it was played on or after
opening night (2026-10-20).

Four factors
------------
They come from NBA.com's own four-factors box score (BoxScoreFourFactorsV3), which is
where the Kaggle training columns come from (effectiveFieldGoalPercentage,
teamTurnoverPercentage, offensiveReboundPercentage, freeThrowAttemptRate). Don't
recompute them from basic box scores: NBA.com's ORB% and TOV% use team rebounds and
its own possession count, so simple formulas are off by up to ~0.05.

Complete box scores are cached in stats/boxscores/<gameId>.json, so reruns don't
download them again. Every request is retried with backoff (2 s, 4 s, 8 s).

Neutral sites
-------------
For the home-court boost (Phase 8) every game carries "neutral": True when it is played at
a neutral site. The flag comes from NBA.com's season schedule (ScheduleLeagueV2), cached in
stats/schedules/<season>.json (refreshed by every --update run):
  * the schedule's own "isNeutral" flag (set from 2024-25 on: Mexico City, Paris, Berlin,
    London, Manchester, the NBA Cup semifinals and final), or
  * the game's arena is in a different state/country than the home team's usual arena.
    This catches 2023-24, whose schedule never sets isNeutral (Mexico City, Paris, the Las
    Vegas Cup semifinals and final), and agrees with isNeutral in every later season. The
    Spurs' games in Austin (same state) stay home games, as NBA.com has them.

Possessions
-----------
For pace (Phase 7), get_game_possessions reads each team's "possessions" from NBA.com's
advanced box score (BoxScoreAdvancedV3, the source of the Kaggle "possessions" column),
cached the same way in stats/boxscores_advanced/<gameId>.json. A game's possessions are
the average of the two teams' counts (they differ by about 1).

A game with its factors looks like:
    {
        "game_id": "0022600001",        # NBA.com gameId (not an ESPN event id)
        "date": "2026-10-20",
        "season_types": ["Regular Season"],
        "home": {"id": "1610612765", "abbr": "DET", "pts": 121,
                 "eFG": 0.541, "TOV": 0.168, "ORB": 0.354, "FTR": 0.372},   # that team's OFFENSE
        "away": {...same...},
        "winner": "1610612765",
        "neutral": False,               # played at a neutral site (see below)
    }
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

from nba_api.stats.endpoints import boxscoreadvancedv3, boxscorefourfactorsv3, leaguegamelog, scheduleleaguev2

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "stats" / "boxscores"
ADVANCED_CACHE_DIR = REPO_ROOT / "stats" / "boxscores_advanced"
SCHEDULE_CACHE_DIR = REPO_ROOT / "stats" / "schedules"

SEASON = "2026-27"
OPENING_NIGHT = "2026-10-20"
SEASON_TYPES = ("Regular Season", "IST")   # IST = NBA Cup
GAME_ID_PREFIXES = ("002", "006")          # regular season (incl. NBA Cup up to the semis), NBA Cup final
CUP_SUBTYPES = ("in-season", "in-season-knockout")   # ScheduleLeagueV2 gameSubtype of NBA Cup games

TIMEOUT = 30
PAUSE = 0.6     # seconds between stats.nba.com requests, to stay polite
RETRIES = 4     # attempts per request
BACKOFF = 2.0   # seconds before the 2nd attempt, doubling after that

V3_FIELDS = {"eFG": "effectiveFieldGoalPercentage", "TOV": "teamTurnoverPercentage",
             "ORB": "offensiveReboundPercentage", "FTR": "freeThrowAttemptRate"}

_sleep = time.sleep
_last_request = 0.0


# ---------------------------------------------------------------------------
# network (the only four functions that talk to NBA.com)
# ---------------------------------------------------------------------------

def _download_game_log(season: str, season_type: str) -> dict:
    return leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type,
                                       timeout=TIMEOUT).get_dict()


def _download_schedule(season: str) -> dict:
    return scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=TIMEOUT).get_dict()


def _download_box_score(game_id: str) -> dict:
    return boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=game_id, timeout=TIMEOUT).get_dict()


def _download_advanced_box_score(game_id: str) -> dict:
    return boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=TIMEOUT).get_dict()


def _throttle():
    global _last_request
    wait = PAUSE - (time.monotonic() - _last_request)
    if wait > 0:
        _sleep(wait)
    _last_request = time.monotonic()


def with_retries(fn, *args, what="request"):
    """fn(*args), retried RETRIES times with exponential backoff; re-raises the last error."""
    for attempt in range(1, RETRIES + 1):
        _throttle()
        try:
            return fn(*args)
        except Exception as err:
            if attempt == RETRIES:
                raise
            delay = BACKOFF * 2 ** (attempt - 1)
            print(f"  {what} failed ({type(err).__name__}: {str(err)[:80]}); "
                  f"retry {attempt}/{RETRIES - 1} in {delay:.0f} s")
            _sleep(delay)


# ---------------------------------------------------------------------------
# which games
# ---------------------------------------------------------------------------

def _games_from_log(raw: dict, season_type: str) -> list:
    result = raw["resultSets"][0]
    games = {}
    for row in result["rowSet"]:
        r = dict(zip(result["headers"], row))
        game = games.setdefault(r["GAME_ID"], {"game_id": r["GAME_ID"], "date": r["GAME_DATE"][:10],
                                               "season_types": [season_type]})
        side = "away" if "@" in r["MATCHUP"] else "home"   # "BOS @ DET" vs "DET vs. BOS"
        game[side] = {"id": str(r["TEAM_ID"]), "abbr": r["TEAM_ABBREVIATION"], "pts": r["PTS"]}
        if r["WL"] == "W":
            game["winner"] = str(r["TEAM_ID"])
    return [g for g in games.values() if "home" in g and "away" in g and "winner" in g]


def _games_from_schedule(raw: dict) -> list:
    """Finished (gameStatus 3) games from ScheduleLeagueV2, in the same shape as the game log's."""
    games = []
    for day in raw["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
            if (g.get("gameStatus") != 3 or not home.get("teamId") or not away.get("teamId")
                    or home.get("score") is None or away.get("score") is None):
                continue
            game_id = g["gameId"]
            if game_id.startswith("006"):
                season_types = ["IST"]
            elif game_id.startswith("002"):
                season_types = ["Regular Season"]
                if g.get("gameSubtype") in CUP_SUBTYPES or "NBA Cup" in (g.get("gameLabel") or ""):
                    season_types.append("IST")
            else:
                season_types = []   # preseason, play-in, playoffs, ...
            h = {"id": str(home["teamId"]), "abbr": home["teamTricode"], "pts": int(home["score"])}
            a = {"id": str(away["teamId"]), "abbr": away["teamTricode"], "pts": int(away["score"])}
            games.append({"game_id": game_id, "date": g["gameDateEst"][:10], "season_types": season_types,
                          "home": h, "away": a, "winner": h["id"] if h["pts"] > a["pts"] else a["id"]})
    return games


def _describe(game: dict) -> str:
    return f"{game['game_id']} {game['date']} {game['away']['abbr']} @ {game['home']['abbr']}"


def combine_game_lists(log_games: list, schedule_games: list):
    """
    Union of the game log and the schedule's finished games, by gameId, plus a list of
    disagreements in words. Only regular-season / NBA Cup ids (002/006) are compared.
    Where both have a game, the game log's details are used and the season types are merged.
    """
    log = {g["game_id"]: g for g in log_games}
    schedule = {g["game_id"]: g for g in schedule_games if g["game_id"][:3] in GAME_ID_PREFIXES}
    combined = {gid: {**g, "season_types": list(g["season_types"])} for gid, g in log.items()}
    disagreements = []

    for gid, s in schedule.items():
        if gid not in log:
            combined[gid] = s
            disagreements.append(f"{_describe(s)}: final in the schedule but missing from the game log (using it)")
            continue
        g = combined[gid]
        for season_type in s["season_types"]:
            if season_type not in g["season_types"]:
                g["season_types"].append(season_type)
        differences = [f"{label} {gv} vs {sv}" for label, gv, sv in (
            ("date", g["date"], s["date"]),
            ("home team", g["home"]["abbr"], s["home"]["abbr"]),
            ("away team", g["away"]["abbr"], s["away"]["abbr"]),
            ("score", f"{g['away']['pts']}-{g['home']['pts']}", f"{s['away']['pts']}-{s['home']['pts']}"),
        ) if gv != sv]
        if differences:
            disagreements.append(f"{_describe(g)}: game log vs schedule differ on {', '.join(differences)} "
                                 "(using the game log)")

    for gid, g in log.items():
        if gid[:3] in GAME_ID_PREFIXES and gid not in schedule:
            disagreements.append(f"{_describe(g)}: in the game log but not final in the schedule (using it)")

    return list(combined.values()), disagreements


def fetch_finished_games(season: str = SEASON, disagreements=None) -> list:
    """
    Every finished game in the game log (SEASON_TYPES) or the schedule, one entry per gameId
    (unfiltered; select_games filters). Disagreements between the two are printed and, if a
    list is passed, appended to `disagreements`.
    """
    log_games = {}
    for season_type in SEASON_TYPES:
        raw = with_retries(_download_game_log, season, season_type, what=f"{season_type!r} game log")
        for game in _games_from_log(raw, season_type):
            if game["game_id"] in log_games:
                log_games[game["game_id"]]["season_types"].append(season_type)
            else:
                log_games[game["game_id"]] = game
    raw_schedule = get_schedule(season, refresh=True)
    schedule_games = _games_from_schedule(raw_schedule)

    games, problems = combine_game_lists(list(log_games.values()), schedule_games)
    flags = neutral_flags(raw_schedule)
    for g in games:
        g["neutral"] = flags.get(g["game_id"], False)
    for problem in problems:
        print(f"  WARNING: {problem}")
    if disagreements is not None:
        disagreements.extend(problems)
    return games


# ---------------------------------------------------------------------------
# season schedules (cached) and neutral sites
# ---------------------------------------------------------------------------

def get_schedule(season: str, refresh: bool = False) -> dict:
    """NBA.com's ScheduleLeagueV2 for a season, from stats/schedules/<season>.json unless refresh (then re-cached)."""
    path = Path(SCHEDULE_CACHE_DIR) / f"{season}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    raw = with_retries(_download_schedule, season, what=f"{season} season schedule")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    os.replace(tmp, path)
    return raw


def neutral_flags(raw: dict) -> dict:
    """
    {gameId: True/False} for every game in a ScheduleLeagueV2 response: neutral when
    isNeutral is set, or when the arena's state/country differs from the home team's usual
    one (its most common arenaState over its regular-season, play-in and playoff home games).
    """
    games = [g for day in raw["leagueSchedule"]["gameDates"] for g in day["games"]]
    usual = {}
    for g in games:
        team = (g.get("homeTeam") or {}).get("teamId")
        if team and g["gameId"][:3] in ("002", "004", "005"):
            usual.setdefault(team, Counter())[g.get("arenaState")] += 1
    usual = {team: states.most_common(1)[0][0] for team, states in usual.items()}
    flags = {}
    for g in games:
        team = (g.get("homeTeam") or {}).get("teamId")
        away_from_home = team in usual and g.get("arenaState") != usual[team]
        flags[g["gameId"]] = bool(g.get("isNeutral")) or away_from_home
    return flags


def game_key(game_id) -> str:
    """A 10-digit NBA.com gameId string from Kaggle's integer ids (22300172 -> "0022300172")."""
    return str(int(game_id)).zfill(10)


def season_of(game_id) -> str:
    """The season a gameId belongs to: "0022300172" -> "2023-24" (digits 4-5 are the season's first year)."""
    year = 2000 + int(game_key(game_id)[3:5])
    return f"{year}-{str(year + 1)[2:]}"


def neutral_games(game_ids) -> dict:
    """
    {gameId: neutral?} for the given games, from the cached schedules of their seasons. A
    season whose cache is missing, or lacks one of the games, is downloaded once (and
    cached); a game that still isn't in its schedule raises KeyError.
    """
    by_season = {}
    for gid in game_ids:
        by_season.setdefault(season_of(gid), []).append(game_key(gid))
    out = {}
    for season, ids in by_season.items():
        flags = neutral_flags(get_schedule(season))
        if any(gid not in flags for gid in ids):
            flags = neutral_flags(get_schedule(season, refresh=True))
        missing = [gid for gid in ids if gid not in flags]
        if missing:
            raise KeyError(f"games not in NBA.com's {season} schedule: {missing[:5]}")
        out.update({gid: flags[gid] for gid in ids})
    return out


def select_games(games: list, opening_night: str = OPENING_NIGHT, through=None) -> list:
    """Regular-season / NBA Cup games from opening night through `through`, oldest first, no duplicates."""
    keep = {}
    for g in games:
        if (g["game_id"][:3] in GAME_ID_PREFIXES
                and any(st in SEASON_TYPES for st in g.get("season_types", []))
                and g["date"] >= opening_night
                and (through is None or g["date"] <= through)):
            keep.setdefault(g["game_id"], g)
    return sorted(keep.values(), key=lambda g: (g["date"], g["game_id"]))


# ---------------------------------------------------------------------------
# four factors (cached)
# ---------------------------------------------------------------------------

def parse_factors(raw: dict):
    """{"home": {"id", "eFG", "TOV", "ORB", "FTR"}, "away": {...}}, or None if any team or factor is missing."""
    try:
        box = raw["boxScoreFourFactors"]
        out = {}
        for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
            stats = box[key]["statistics"]
            factors = {name: stats.get(field) for name, field in V3_FIELDS.items()}
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 2
                       for v in factors.values()):
                return None
            out[side] = {"id": str(box[key]["teamId"]), **{k: float(v) for k, v in factors.items()}}
        return out
    except (KeyError, TypeError):
        return None


def _cache_path(game_id: str) -> Path:
    return Path(CACHE_DIR) / f"{game_id}.json"


def get_game_factors(game_id: str):
    """
    (factors, None) for a game, or (None, reason) when it can't be used.
    Reads stats/boxscores/<gameId>.json if present; otherwise downloads (with retries)
    and caches the box score, but only when both teams' four factors are complete.
    """
    path = _cache_path(game_id)
    if path.exists():
        factors = parse_factors(json.loads(path.read_text(encoding="utf-8")))
        if factors is not None:
            return factors, None

    try:
        raw = with_retries(_download_box_score, game_id, what=f"box score {game_id}")
    except Exception as err:
        return None, f"download failed {RETRIES} times ({type(err).__name__})"

    factors = parse_factors(raw)
    if factors is None:
        return None, "four factors missing for at least one team"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    os.replace(tmp, path)
    return factors, None


# ---------------------------------------------------------------------------
# possessions (cached; pace only)
# ---------------------------------------------------------------------------

def parse_possessions(raw: dict):
    """{"home": {"id", "possessions"}, "away": {...}}, or None if either team's count is missing or implausible."""
    try:
        box = raw["boxScoreAdvanced"]
        out = {}
        for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
            value = box[key]["statistics"].get("possessions")
            if not (isinstance(value, (int, float)) and not isinstance(value, bool) and 50 <= value <= 200):
                return None
            out[side] = {"id": str(box[key]["teamId"]), "possessions": float(value)}
        return out
    except (KeyError, TypeError):
        return None


def game_possessions(possessions: dict) -> float:
    """A game's possessions: the average of the two teams' counts."""
    return (possessions["home"]["possessions"] + possessions["away"]["possessions"]) / 2


def get_game_possessions(game_id: str):
    """
    (possessions, None) for a game (parse_possessions shape), or (None, reason).
    Reads stats/boxscores_advanced/<gameId>.json if present; otherwise downloads (with
    retries) and caches the advanced box score, but only when both teams' counts are there.
    """
    path = Path(ADVANCED_CACHE_DIR) / f"{game_id}.json"
    if path.exists():
        possessions = parse_possessions(json.loads(path.read_text(encoding="utf-8")))
        if possessions is not None:
            return possessions, None

    try:
        raw = with_retries(_download_advanced_box_score, game_id, what=f"advanced box score {game_id}")
    except Exception as err:
        return None, f"advanced box score download failed {RETRIES} times ({type(err).__name__})"

    possessions = parse_possessions(raw)
    if possessions is None:
        return None, "possessions missing for at least one team"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    os.replace(tmp, path)
    return possessions, None


def with_possessions(game: dict, possessions: dict) -> dict:
    """The game with "possessions" (the game's average) filled in (checks the teams match)."""
    for side in ("home", "away"):
        if possessions[side]["id"] != game[side]["id"]:
            raise ValueError(f"game {game['game_id']}: {side} team differs between game and advanced box score")
    return {**game, "possessions": game_possessions(possessions)}


def with_factors(game: dict, factors: dict) -> dict:
    """The game with each side's four factors filled in (checks the box score is for the same teams)."""
    merged = {**game, "home": dict(game["home"]), "away": dict(game["away"])}
    for side in ("home", "away"):
        if factors[side]["id"] != game[side]["id"]:
            raise ValueError(f"game {game['game_id']}: {side} team differs between game log and box score")
        merged[side].update(factors[side])
    return merged


def fetch_games(season: str = SEASON, limit=None, through=None) -> list:
    """Selected games with their four factors, oldest first; games without complete factors are left out."""
    out = []
    for game in select_games(fetch_finished_games(season), through=through):
        if limit is not None and len(out) >= limit:
            break
        factors, problem = get_game_factors(game["game_id"])
        if factors is not None:
            out.append(with_factors(game, factors))
    return out
