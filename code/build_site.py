"""
Build the predictions website for this week's NBA games.

    python code/build_site.py                 # today + next 6 days
    python code/build_site.py --start 2026-11-16 --days 7
    python code/build_site.py --priors other_priors.json
    Defaults = the live model (params.py): fitted sim sigma, --pace on, --sim-corr off, --home on,
    --rest on; needs priors with team paces (priors.json has them) and each team's rest, taken
    from NBA.com's season schedule (the stats/schedules/ cache, downloaded if missing).
    python code/build_site.py --rest off --priors priors_2026-27_pace-home.json   # the previous live model
    python code/build_site.py --sigma-sim legacy --pace off --home off --rest off  # the Phase 6 simulation
    python code/build_site.py --sim-corr on                              # Phase 7 correlated factors
    python code/build_site.py --injuries off                             # no injury report (Phase 12)
    python code/build_site.py --key-top 3 --key-mpg 30                   # stricter "key player" rule
    python code/build_site.py --seed 1                                   # reproducible win % (default: fresh draws)

Writes site/index.html (open it in a browser) and site/predictions.json.

Injury warnings (Phase 12, display only): the current injury report (injuries.py: ESPN, CBS as backup)
is fetched, appended to data/injuries_2026-27.csv, and each game not yet final within --injury-days (2:
today and tomorrow, US Eastern; statuses are about a team's next game) shows its KEY players (team top 5
by minutes per game, or 25+ minutes) who are Out / Doubtful ("strong") or Questionable / Day-To-Day ("mild"). The predictions are made before and without it: annotate_injuries only adds a
row["injuries"] key. If the report can't be fetched the site is built without warnings (with a note)
and a WARNING is printed; it never stops the build.

Track record (Phase 13, display only): a "How accurate is it?" section after the games (track_record.py:
this season from the live prediction logs, last 7 days, recent results, the backtest headline from
reports/accuracy_report.md, model vs Vegas on games with a closing line), also built after the predictions.
"""
import argparse
import html
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import boxscores
import injuries
import params
import rest as rest_days
import track_record
from predict import REPO_ROOT, load_model, predict_game
from schedule import fetch_games

SITE_DIR = REPO_ROOT / "site"
EASTERN = ZoneInfo("America/New_York")


def build_predictions(games, model, sim_sigma=None, sim_corr: str = params.DEFAULT_SIM_CORR_MODE,
                      pace: str = params.DEFAULT_PACE_MODE, home: str = params.DEFAULT_HOME_MODE,
                      rest: str = params.DEFAULT_REST_MODE, rest_lookup=None,
                      calib: str = params.DEFAULT_CALIB_MODE, react: str = params.DEFAULT_REACT_MODE, seed=None):
    """
    sim_sigma: {factor: game noise} for the simulation; None = params.sim_sigma() (fitted by default).
    seed: None = fresh draws (the default site), an int = the same win % on every build.
    sim_corr / pace / home / rest / calib: "on" or "off" (defaults in params.py, see simulation.py); a game
    with "neutral": True gets no home boost. rest="on" needs rest_lookup(team_id, date) -> rest
    info (rest.rest_lookup of the season's NBA.com schedule); each row then carries "rest".
    react: which calibration temperature calib "on" uses (Phase 10c: the one refitted for --react on);
    the ratings themselves are whatever `model` is.
    """
    temperature = params.calib_temperature(react) if params.is_on(calib, "calib") else None
    params.is_on(react, "react")
    with_rest = params.is_on(rest, "rest")
    if with_rest and rest_lookup is None:
        raise ValueError("rest is on but there is no rest_lookup (the season's schedule)")
    rows = []
    for g in games:
        days = {side: rest_lookup(g[side]["id"], g["date"])["rest"] for side in ("home", "away")} if with_rest else {}
        pred = predict_game(g["home"]["id"], g["away"]["id"], model, seed=seed, sigma=sim_sigma, sim_corr=sim_corr,
                            pace=pace,
                            home=home, neutral=g.get("neutral", False), rest=rest, rest_home=days.get("home"),
                            rest_away=days.get("away"), calib=calib, calib_temperature=temperature)
        row = dict(g, prediction=pred)
        if with_rest:
            row["rest"] = days
        if pred:
            home_fav = pred["home_win_pct"] >= pred["away_win_pct"]
            row["pick"] = g["home"]["abbr"] if home_fav else g["away"]["abbr"]
            if g["status"] == "final" and g["home"]["score"] is not None:
                winner = g["home"]["abbr"] if g["home"]["score"] > g["away"]["score"] else g["away"]["abbr"]
                row["pick_correct"] = winner == row["pick"]
        rows.append(row)
    return rows


INJURY_DAYS = 2   # warnings on games up to this many days from the report (today + tomorrow, US Eastern)


def last_injury_day(fetched_at: str, days: int = INJURY_DAYS) -> str:
    """The last game date (YYYY-MM-DD) that gets warnings from a report fetched at `fetched_at` (UTC)."""
    t = datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).astimezone(EASTERN).date()
    return (t + timedelta(days=days - 1)).isoformat()


def annotate_injuries(rows, warnings_by_team: dict, through=None) -> int:
    """
    Add row["injuries"] = {"home": [...], "away": [...]} (injuries.team_warnings entries) to every game that
    isn't final, is on or before `through` (YYYY-MM-DD; None = every date) and has a warning: statuses like
    "questionable for Friday's game" are about the next game, not next week's. Nothing else in the row changes.
    Returns the number of warnings shown.
    """
    shown = 0
    for row in rows:
        if row.get("status") == "final" or (through is not None and row["date"] > through):
            continue
        found = {side: warnings_by_team.get(row[side]["id"], []) for side in ("home", "away")}
        if any(found.values()):
            row["injuries"] = found
            shown += sum(len(w) for w in found.values())
    return shown


def render_injuries(row) -> str:
    inj = row.get("injuries")
    if not inj:
        return ""
    items = []
    for side in ("away", "home"):
        for w in inj.get(side, []):
            items.append(f'<span class="inj {w["level"]}" title="{html.escape(w["player"])}, '
                         f'{w["mpg"]:.1f} min/game, {html.escape(row[side]["abbr"])}">'
                         f'⚠ {html.escape(w["last_name"] or w["player"])} ({html.escape(w["status"])})</span>')
    return f'<div class="injuries">{"".join(items)}</div>' if items else ""


def injury_note(meta: dict) -> str:
    """The one line under the header: the disclaimer, the rule and when the report was fetched."""
    base = "Predictions don't account for injuries."
    if not meta.get("on"):
        return f"{base} Injury warnings are off for this build."
    if not meta.get("fetched_at"):
        return f"{base} The injury report couldn't be fetched for this build, so no warnings are shown."
    t = datetime.fromisoformat(meta["fetched_at"].replace("Z", "+00:00")).astimezone(EASTERN)
    when = f"{t:%b} {t.day}, {t.hour % 12 or 12}:{t:%M %p} ET"
    return (f"{base} ⚠ marks a key player (top {meta['key_top']} in minutes per game on his team, or "
            f"{meta['key_mpg']:g}+ min) who is Out or Doubtful (red) or Questionable / day-to-day (amber), "
            f"from the {html.escape(meta['source'])} injury report fetched {when}; shown for games through "
            f"{_short_date(meta['through'])}.")


def _short_date(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d:%a %b} {d.day}"


def _team_line(team, win_pct, score, favored):
    cls = "team fav" if favored else "team"
    pct = f"{win_pct:.0f}%" if win_pct is not None else "–"
    proj = f"{score:.0f}" if score is not None else ""
    return (f'<div class="{cls}"><span class="abbr">{html.escape(team["abbr"])}</span>'
            f'<span class="name">{html.escape(team["name"])}</span>'
            f'<span class="proj">{proj}</span><span class="pct">{pct}</span></div>')


def render_card(row):
    pred = row["prediction"]
    home, away = row["home"], row["away"]
    if pred:
        home_fav = pred["home_win_pct"] >= pred["away_win_pct"]
        bar = pred["away_win_pct"]
        lines = (_team_line(away, pred["away_win_pct"], pred["away_score"], not home_fav)
                 + _team_line(home, pred["home_win_pct"], pred["home_score"], home_fav))
        spread = abs(pred["home_score"] - pred["away_score"])
        footer = f'Pick: <b>{html.escape(row["pick"])}</b> by {spread:.1f}'
    else:
        bar = 50
        lines = _team_line(away, None, None, False) + _team_line(home, None, None, False)
        footer = "No prediction (team missing from model)"

    if row["status"] == "final":
        status = f'Final: {away["abbr"]} {away["score"]} – {home["abbr"]} {home["score"]}'
        if "pick_correct" in row:
            status += " ✓" if row["pick_correct"] else " ✗"
    elif row["status"] == "live":
        status = f'Live: {away["abbr"]} {away["score"]} – {home["abbr"]} {home["score"]}'
    else:
        status = html.escape(row["status_text"] or "")

    tip = html.escape(row["tipoff_utc"] or "")
    return f"""
      <div class="card {row['status']}">
        <div class="meta"><span class="tip" data-utc="{tip}">{status}</span><span class="at">away @ home</span></div>
        {lines}
        <div class="bar"><div style="width:{bar:.1f}%"></div></div>
        <div class="foot">{footer}</div>{render_injuries(row)}
      </div>"""


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _games(n: int) -> str:
    return f"{n:,} game{'s' if n != 1 else ''}"


def _day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def _wins(x: float) -> str:
    return f"{x:g}" if x != int(x) else f"{int(x)}"


BRIER_LINE = ("Brier score = the average of (predicted win chance − result)², with result 1 for a win and 0 for a "
              "loss: 0 is perfect, always saying 50% scores 0.250, lower is better.")


def render_track_record(tr) -> str:
    """The "How accurate is it?" section (track_record.build's result); a short note if there is none."""
    if tr is None:
        return ('<section class="record"><h2>How accurate is it?</h2>'
                "<p class=\"note\">The track record couldn't be read for this build.</p></section>")
    s, w = tr["season"], tr["last7"]
    parts = ['<section class="record"><h2>How accurate is it?</h2>']
    if s["games"]:
        parts.append(f'<p class="lead">This season ({_day(tr["first"])} – {_day(tr["last"])}), every game predicted '
                     f'before it was played:</p><div class="stats">'
                     f'<div class="stat"><b>{_pct(s["accuracy"])}</b><span>winners picked<br>'
                     f'{_wins(s["right"])} of {_games(s["games"])}</span></div>'
                     f'<div class="stat"><b>{s["brier"]:.3f}</b><span>Brier score<br>{_games(s["games"])}</span></div>'
                     f'<div class="stat"><b>{_pct(s["confidence"])}</b><span>average confidence in its pick<br>'
                     f'vs {_pct(s["accuracy"])} of picks right ({_games(s["games"])})</span></div>')
        if w["games"]:
            wins, losses = w["right"], w["games"] - w["right"]
            parts.append(f'<div class="stat"><b>{_wins(wins)}–{_wins(losses)}</b><span>last 7 days '
                         f'({_day(tr["since"])} – {_day(tr["today"])})<br>{_pct(w["accuracy"])} of '
                         f'{_games(w["games"])}</span></div>')
        else:
            parts.append(f'<div class="stat"><b>–</b><span>last 7 days ({_day(tr["since"])} – '
                         f'{_day(tr["today"])})<br>no games logged yet</span></div>')
        parts.append("</div>")
    else:
        parts.append('<p class="lead">No games of this season have been predicted and played yet.</p>')
    parts.append(f'<p class="note">{BRIER_LINE}</p>')

    b = tr.get("backtest")
    if b:
        m, v, h = b["rows"]["Model"], b["rows"]["Vegas (spread)"], b["rows"]["Home team"]
        parts.append(f'<p>Past seasons, for context (a backtest of 2024-25 and 2025-26 run the way the model runs '
                     f'live): it picked <b>{_pct(m["accuracy"])}</b> of winners over {_games(m["games"])}, vs '
                     f'{_pct(v["accuracy"])} for the Vegas closing lines ({_games(v["games"])}) and '
                     f'{_pct(h["accuracy"])} for always picking the home team ({_games(h["games"])}).</p>')
    vg = tr.get("vegas")
    if vg:
        mo, ve = vg["model"], vg["vegas"]
        parts.append(f'<p>This season against Vegas, on the {_games(vg["games"])} with a saved closing line: model '
                     f'{_pct(mo["accuracy"])} of winners (Brier {mo["brier"]:.3f}), Vegas {_pct(ve["accuracy"])} '
                     f'(Brier {ve["brier"]:.3f}), both on the same {_games(vg["games"])}.</p>')
    if tr["recent"]:
        rows = []
        for g in tr["recent"]:
            mark = '<span class="ok">✓ right</span>' if g["right"] else '<span class="bad">✗ wrong</span>'
            rows.append(f'<tr><td>{_day(g["date"])}</td><td>{html.escape(g["away"])} @ {html.escape(g["home"])}</td>'
                        f'<td>{html.escape(g["pick"])} {g["pick_pct"]:.0f}%</td>'
                        f'<td>{html.escape(g["away"])} {g["away_pts"]} – {html.escape(g["home"])} {g["home_pts"]}</td>'
                        f'<td>{mark}</td></tr>')
        parts.append(f'<h3>Most recent {_games(len(tr["recent"]))}</h3><div class="tablewrap"><table class="recent">'
                     '<thead><tr><th>Date</th><th>Game</th><th>Pick (win %)</th><th>Final</th><th></th></tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table></div>')
    parts.append("</section>")
    return "".join(parts)


def render_page(rows, source, priors_path, start, end, injury_line: str = "", record_html: str = ""):
    by_day = {}
    for r in rows:
        by_day.setdefault(r["date"], []).append(r)

    sections = []
    for day, day_rows in by_day.items():
        d = date.fromisoformat(day)
        label = f"{d:%A, %b} {d.day}"
        cards = "".join(render_card(r) for r in day_rows)
        plural = "s" if len(day_rows) != 1 else ""
        sections.append(f'<section><h2>{label} <small>{len(day_rows)} game{plural}</small></h2>'
                        f'<div class="grid">{cards}</div></section>')
    if not sections:
        sections.append('<p class="empty">No games scheduled this week.</p>')

    graded = [r for r in rows if "pick_correct" in r]
    record = ""
    if graded:
        right = sum(r["pick_correct"] for r in graded)
        record = f" · Model is {right}-{len(graded) - right} on finished games this week"

    now = datetime.now(timezone.utc).astimezone(EASTERN)
    updated = f"{now:%b} {now.day}, {now.year} {now.hour % 12 or 12}:{now:%M %p} ET"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Predictions</title>
<style>
  :root {{ --bg:#f6f7f9; --card:#fff; --text:#15171a; --muted:#6b7280; --line:#e5e7eb; --accent:#1d4ed8; --away:#c2410c; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1115; --card:#181b21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f37; --accent:#60a5fa; --away:#fb923c; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }}
  header, main {{ max-width:1000px; margin:0 auto; padding:16px; }}
  h1 {{ margin:8px 0 4px; font-size:26px; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  h2 {{ font-size:17px; margin:24px 0 10px; }}
  h2 small {{ color:var(--muted); font-weight:400; font-size:13px; margin-left:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .meta {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-bottom:6px; }}
  .team {{ display:grid; grid-template-columns:44px 1fr 36px 44px; align-items:center; padding:3px 0; color:var(--muted); }}
  .team.fav {{ color:var(--text); font-weight:600; }}
  .abbr {{ font-weight:700; }}
  .name {{ font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .proj {{ text-align:right; font-size:12px; color:var(--muted); font-weight:400; }}
  .pct {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .bar {{ height:6px; background:var(--accent); border-radius:3px; overflow:hidden; margin:8px 0; }}
  .bar div {{ height:100%; background:var(--away); }}
  .foot {{ font-size:13px; color:var(--muted); }}
  .foot b {{ color:var(--text); }}
  .card.final {{ opacity:.85; }}
  .empty {{ color:var(--muted); }}
  footer {{ max-width:1000px; margin:24px auto; padding:0 16px 24px; color:var(--muted); font-size:12px; }}
  .note {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .record {{ margin-top:32px; }}
  .record h3 {{ font-size:15px; margin:18px 0 8px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:10px; margin:10px 0; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
  .stat b {{ display:block; font-size:22px; font-variant-numeric:tabular-nums; }}
  .stat span {{ color:var(--muted); font-size:12px; }}
  .tablewrap {{ overflow-x:auto; }}
  table.recent {{ border-collapse:collapse; width:100%; font-size:13px; background:var(--card); }}
  table.recent th, table.recent td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
  table.recent th {{ color:var(--muted); font-weight:600; }}
  .ok {{ color:#15803d; }} .bad {{ color:#b91c1c; }}
  @media (prefers-color-scheme: dark) {{ .ok {{ color:#4ade80; }} .bad {{ color:#f87171; }} }}
  .injuries {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:4px 10px; font-size:12px; }}
  .inj.strong {{ color:#b91c1c; font-weight:600; }}
  .inj.mild {{ color:#b45309; }}
  @media (prefers-color-scheme: dark) {{ .inj.strong {{ color:#f87171; }} .inj.mild {{ color:#fbbf24; }} }}
</style>
</head>
<body>
<header>
  <h1>NBA Predictions</h1>
  <div class="sub">{start:%b} {start.day} – {end:%b} {end.day}, {end.year} · {len(rows)} games{record}</div>
  {f'<div class="note">{injury_line}</div>' if injury_line else ''}
</header>
<main>{''.join(sections)}{record_html}</main>
<footer>
  Win % from 10,000 Monte Carlo sims of each team's Four Factors (eFG%, TOV%, ORB%, FTR).
  Projected scores shown next to each team. Bar: orange = away, blue = home.
  Includes home court and rest (back-to-backs).<br>
  Schedule: {html.escape(source)} · Model: {html.escape(Path(priors_path).name)} · Updated {updated}
</footer>
<script>
  // show tipoff times in the viewer's own time zone for games that haven't started
  document.querySelectorAll('.card.scheduled .tip').forEach(el => {{
    const t = el.dataset.utc;
    if (!t) return;
    const d = new Date(t);
    if (!isNaN(d)) el.textContent = d.toLocaleTimeString([], {{hour:'numeric', minute:'2-digit', timeZoneName:'short'}});
  }});
</script>
</body>
</html>
"""


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: today, US Eastern)")
    parser.add_argument("--days", type=int, default=7, help="how many days to include (default 7)")
    parser.add_argument("--priors", help="model file to use (default: priors.json or $NBA_PRIORS)")
    parser.add_argument("--sigma-sim", choices=params.SIGMA_MODES, default=params.DEFAULT_SIM_SIGMA_MODE,
                        help=f"simulation game noise (default {params.DEFAULT_SIM_SIGMA_MODE})")
    parser.add_argument("--sim-corr", choices=params.SWITCH_MODES, default=params.DEFAULT_SIM_CORR_MODE,
                        help="draw the game factors jointly with the fitted correlation "
                             f"(default {params.DEFAULT_SIM_CORR_MODE})")
    parser.add_argument("--pace", choices=params.SWITCH_MODES, default=params.DEFAULT_PACE_MODE,
                        help="possessions from the teams' pace curves + the corrected points formula; needs "
                             f"priors with team paces (default {params.DEFAULT_PACE_MODE})")
    parser.add_argument("--home", choices=params.SWITCH_MODES, default=params.DEFAULT_HOME_MODE,
                        help="add the fitted home-court boost (none at neutral sites); use priors trained with "
                             f"--home on (default {params.DEFAULT_HOME_MODE})")
    parser.add_argument("--rest", choices=params.SWITCH_MODES, default=params.DEFAULT_REST_MODE,
                        help="add the fitted rest effect (back-to-backs, from NBA.com's season schedule); use priors "
                             f"trained with --rest on (default {params.DEFAULT_REST_MODE})")
    parser.add_argument("--calib", choices=params.SWITCH_MODES, default=params.DEFAULT_CALIB_MODE,
                        help="calibrate the win %% with the fitted temperature (Phase 10b; scores unchanged; "
                             f"default {params.DEFAULT_CALIB_MODE})")
    parser.add_argument("--react", choices=params.SWITCH_MODES, default=params.DEFAULT_REACT_MODE,
                        help="the priors come from the --react on season procedure (Phase 10c); only picks the "
                             f"calibration temperature for --calib on (default {params.DEFAULT_REACT_MODE})")
    parser.add_argument("--injuries", choices=params.SWITCH_MODES, default="on",
                        help="fetch the injury report and show key-player warnings (display only; default on)")
    parser.add_argument("--key-top", type=int, default=injuries.TOP_N,
                        help=f"key player: his team's top N by minutes per game (default {injuries.TOP_N})")
    parser.add_argument("--key-mpg", type=float, default=injuries.MIN_MPG,
                        help=f"... or at least this many minutes per game (default {injuries.MIN_MPG:g})")
    parser.add_argument("--injury-days", type=int, default=INJURY_DAYS,
                        help=f"show warnings on games up to this many days from the report, today = 1 "
                             f"(default {INJURY_DAYS}: today and tomorrow, US Eastern)")
    parser.add_argument("--seed", type=int, help="simulation seed: the same win %% on every build (default: fresh draws)")
    return parser


def track_record_for(today):
    """track_record.build(today), or None (with a WARNING) if it fails: display only, never stops the build."""
    try:
        return track_record.build(today)
    except Exception as err:   # noqa: BLE001
        print(f"WARNING: track record not built ({type(err).__name__}: {err}); the site shows a note instead")
        return None


def injury_warnings(args, injury_report=None, injury_error=None):
    """
    (warnings by team, meta for the page). injury_report: an already fetched report (daily.py fetches it within
    its retries); injury_error: the fetch already failed. Neither: fetch it now. Never raises.
    """
    meta = {"on": args.injuries == "on", "source": None, "fetched_at": None, "through": None, "error": None,
            "key_top": args.key_top, "key_mpg": args.key_mpg}
    if not meta["on"]:
        return {}, meta
    report = injury_report
    if report is None and injury_error is None:
        try:
            report = injuries.update_report()
        except Exception as err:   # noqa: BLE001 (display only: the site is built without warnings)
            injury_error = f"{type(err).__name__}: {err}"
    if report is None:
        meta["error"] = injury_error
        print(f"WARNING: no injury report ({injury_error}); building the site without injury warnings")
        return {}, meta
    meta.update(source=report["source"], fetched_at=report["fetched_at"],
                through=last_injury_day(report["fetched_at"], args.injury_days))
    try:
        return injuries.team_warnings(report, injuries.season_minutes(), args.key_top, args.key_mpg), meta
    except Exception as err:   # noqa: BLE001
        meta.update(fetched_at=None, error=f"{type(err).__name__}: {err}")
        print(f"WARNING: injury warnings failed ({meta['error']}); building the site without them")
        return {}, meta


def main(argv=None, injury_report=None, injury_error=None):
    args = make_parser().parse_args(argv)
    if args.key_top < 0 or args.key_mpg < 0 or args.injury_days < 1:
        raise SystemExit("--key-top and --key-mpg must not be negative, --injury-days must be at least 1")

    start = date.fromisoformat(args.start) if args.start else datetime.now(EASTERN).date()
    end = start + timedelta(days=args.days - 1)

    model, priors_path = load_model(args.priors)
    games, source = fetch_games(start, args.days)
    lookup = rest_days.rest_lookup(boxscores.get_schedule(boxscores.SEASON)) if args.rest == "on" else None
    rows = build_predictions(games, model, sim_sigma=params.sim_sigma(args.sigma_sim), sim_corr=args.sim_corr,
                             pace=args.pace, home=args.home, rest=args.rest, rest_lookup=lookup, calib=args.calib,
                             react=args.react, seed=args.seed)
    # injuries come after the predictions and only add row["injuries"] (display only)
    warnings_by_team, inj_meta = injury_warnings(args, injury_report, injury_error)
    record = track_record_for(datetime.now(EASTERN).date())
    shown = annotate_injuries(rows, warnings_by_team, inj_meta["through"])

    SITE_DIR.mkdir(exist_ok=True)
    data = json.dumps(
        {"start": start.isoformat(), "end": end.isoformat(), "source": source,
         "model": Path(priors_path).name, "sigma_sim": args.sigma_sim, "sim_corr": args.sim_corr, "pace": args.pace,
         "home": args.home, "rest": args.rest, "calib": args.calib, "react": args.react,
         "injuries": inj_meta, "track_record": record, "games": rows}, indent=2)
    page = render_page(rows, source, priors_path, start, end, injury_note(inj_meta), render_track_record(record))
    # each file is written whole (temp file + rename), so an interrupted run never leaves half a page
    for name, text in (("predictions.json", data), ("index.html", page)):
        tmp = SITE_DIR / f"{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, SITE_DIR / name)
    print(f"Wrote {len(rows)} games ({source}) to {SITE_DIR / 'index.html'}"
          + (f"; {shown} key-player injury warning(s) ({inj_meta['source']})" if inj_meta["fetched_at"] else
             "; no injury warnings")
          + (f"; track record: {record['season']['games']} games this season" if record else "; no track record"))
    return {"games": len(rows), "source": source, "start": start.isoformat(), "end": end.isoformat(),
            "injury_warnings": shown if inj_meta["fetched_at"] else None, "injury_source": inj_meta["source"],
            "injury_error": inj_meta["error"],
            "record_games": record["season"]["games"] if record else None}


if __name__ == "__main__":
    main()
