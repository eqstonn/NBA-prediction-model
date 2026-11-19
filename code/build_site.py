"""
Build the predictions website for this week's NBA games.

    python code/build_site.py                 # today + next 6 days
    python code/build_site.py --start 2026-11-16 --days 7
    python code/build_site.py --priors other_priors.json
    Defaults = the live model (params.py): fitted sim sigma, --pace on, --sim-corr off, --home on;
    needs priors with team paces (priors.json has them).
    python code/build_site.py --sigma-sim legacy --pace off --home off   # the previous live simulation
    python code/build_site.py --sim-corr on                              # Phase 7 correlated factors

Writes site/index.html (open it in a browser) and site/predictions.json.
"""
import argparse
import html
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import params
from predict import REPO_ROOT, load_model, predict_game
from schedule import fetch_games

SITE_DIR = REPO_ROOT / "site"
EASTERN = ZoneInfo("America/New_York")


def build_predictions(games, model, sim_sigma=None, sim_corr: str = params.DEFAULT_SIM_CORR_MODE,
                      pace: str = params.DEFAULT_PACE_MODE, home: str = params.DEFAULT_HOME_MODE):
    """
    sim_sigma: {factor: game noise} for the simulation; None = params.sim_sigma() (fitted by default).
    sim_corr / pace / home: "on" or "off" (defaults in params.py, see simulation.py); a game with
    "neutral": True gets no home boost.
    """
    rows = []
    for g in games:
        pred = predict_game(g["home"]["id"], g["away"]["id"], model, sigma=sim_sigma, sim_corr=sim_corr, pace=pace,
                            home=home, neutral=g.get("neutral", False))
        row = dict(g, prediction=pred)
        if pred:
            home_fav = pred["home_win_pct"] >= pred["away_win_pct"]
            row["pick"] = g["home"]["abbr"] if home_fav else g["away"]["abbr"]
            if g["status"] == "final" and g["home"]["score"] is not None:
                winner = g["home"]["abbr"] if g["home"]["score"] > g["away"]["score"] else g["away"]["abbr"]
                row["pick_correct"] = winner == row["pick"]
        rows.append(row)
    return rows


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
        <div class="foot">{footer}</div>
      </div>"""


def render_page(rows, source, priors_path, start, end):
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
</style>
</head>
<body>
<header>
  <h1>NBA Predictions</h1>
  <div class="sub">{start:%b} {start.day} – {end:%b} {end.day}, {end.year} · {len(rows)} games{record}</div>
</header>
<main>{''.join(sections)}</main>
<footer>
  Win % from 10,000 Monte Carlo sims of each team's Four Factors (eFG%, TOV%, ORB%, FTR).
  Projected scores shown next to each team. Bar: orange = away, blue = home.
  No home-court adjustment yet.<br>
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
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)

    start = date.fromisoformat(args.start) if args.start else datetime.now(EASTERN).date()
    end = start + timedelta(days=args.days - 1)

    model, priors_path = load_model(args.priors)
    games, source = fetch_games(start, args.days)
    rows = build_predictions(games, model, sim_sigma=params.sim_sigma(args.sigma_sim), sim_corr=args.sim_corr,
                             pace=args.pace, home=args.home)

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "predictions.json").write_text(json.dumps(
        {"start": start.isoformat(), "end": end.isoformat(), "source": source,
         "model": Path(priors_path).name, "sigma_sim": args.sigma_sim, "sim_corr": args.sim_corr, "pace": args.pace,
         "home": args.home, "games": rows}, indent=2), encoding="utf-8")
    (SITE_DIR / "index.html").write_text(render_page(rows, source, priors_path, start, end), encoding="utf-8")
    print(f"Wrote {len(rows)} games ({source}) to {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
