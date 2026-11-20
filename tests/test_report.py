"""
Tests for the accuracy report (code/report.py): the spread -> win probability conversion, the
metrics, the baselines, the 2026-27 sample's reading of logs and lines, and the report writing
(on synthetic data in a temp folder; the real training data and report files aren't touched).
"""
import csv
import math

import numpy as np
import pytest

import report

SQRT2 = math.sqrt(2)


# ---------------------------------------------------------------------------
# spread -> probability
# ---------------------------------------------------------------------------

def test_normal_cdf_known_values():
    assert report.normal_cdf(0) == 0.5
    assert report.normal_cdf(1) == pytest.approx(0.841344746, abs=1e-9)
    assert report.normal_cdf(-1.96) == pytest.approx(0.024997895, abs=1e-9)


def test_spread_win_prob():
    assert report.spread_win_prob(0, 13.8) == 0.5                                  # pick'em
    assert report.spread_win_prob(-13.8, 13.8) == pytest.approx(0.841344746)      # home favored by one SD
    assert report.spread_win_prob(7, 14) == pytest.approx(0.308537539)            # home +7: Phi(-0.5)
    for s in (-12.5, -3, 1.5, 9):
        assert report.spread_win_prob(s, 12) + report.spread_win_prob(-s, 12) == pytest.approx(1)
    assert report.spread_win_prob(-5, 10) > report.spread_win_prob(-5, 14) > 0.5  # a wider SD is less sure
    probs = report.spread_win_prob(np.array([-7.0, 0.0, 3.5]), 14)
    assert probs == pytest.approx([report.normal_cdf(0.5), 0.5, report.normal_cdf(-0.25)])
    with pytest.raises(ValueError):
        report.spread_win_prob(-3, 0)


def test_fit_spread_sd_by_hand():
    """margins 10, -3, 5 with home spreads -7, +2, -5: residuals (margin + spread) 3, -1, 0."""
    fit = report.fit_spread_sd([10, -3, 5], [-7, 2, -5])
    assert fit["sd"] == pytest.approx(math.sqrt(10 / 3))
    assert fit["bias"] == pytest.approx(2 / 3)
    assert fit["n"] == 3
    with pytest.raises(ValueError):
        report.fit_spread_sd([], [])


def test_fit_spread_sd_recovers_simulated_noise():
    rng = np.random.default_rng(7)
    spreads = rng.uniform(-12, 12, 20000)
    margins = -spreads + rng.normal(0, 13.5, spreads.size)
    assert report.fit_spread_sd(margins, spreads)["sd"] == pytest.approx(13.5, rel=0.02)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

P, WON = [0.8, 0.3, 0.5, 0.6], [True, True, False, False]


def test_per_game_by_hand():
    each = report.per_game(P, WON)
    assert each["hit"].tolist() == [1, 0, 0.5, 0]                  # P = 0.5 counts half
    assert each["brier"] == pytest.approx([0.04, 0.49, 0.25, 0.36])
    assert each["log_loss"] == pytest.approx([-math.log(0.8), -math.log(0.3), -math.log(0.5), -math.log(0.4)])
    assert each["confidence"] == pytest.approx([0.8, 0.7, 0.5, 0.6])


def test_log_loss_is_clipped_not_infinite():
    each = report.per_game([1.0, 0.0], [False, True])
    assert each["log_loss"] == pytest.approx([-math.log(report.EPS)] * 2)
    assert each["brier"].tolist() == [1.0, 1.0]


def test_per_game_rejects_bad_input():
    with pytest.raises(ValueError):
        report.per_game([0.5, 1.2], [True, False])
    with pytest.raises(ValueError):
        report.per_game([0.5], [True, False])


def test_evaluate_by_hand():
    m = report.evaluate(P, WON, proj_margin=[5, -2, 0, 3], margin=[8, 1, -4, -2],
                        proj_total=[220, 230, 210, 225], total=[230, 228, 200, 225])
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.375)
    assert m["brier"] == pytest.approx(0.285)
    assert m["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.3) + math.log(0.5) + math.log(0.4)) / 4)
    assert m["confidence"] == pytest.approx(0.65)
    assert m["margin_mae"] == pytest.approx((3 + 3 + 4 + 5) / 4)
    assert m["total_mae"] == pytest.approx((10 + 2 + 10 + 0) / 4)
    assert report.evaluate(P, WON)["margin_mae"] is None


def test_brier_of_a_coin_flip_and_a_perfect_forecast():
    won = [True, False] * 50
    assert report.evaluate([0.5] * 100, won)["brier"] == 0.25
    assert report.evaluate([0.5] * 100, won)["accuracy"] == 0.5
    perfect = report.evaluate([1.0 if w else 0.0 for w in won], won)
    assert perfect["brier"] == 0 and perfect["accuracy"] == 1


def test_paired_difference_and_se():
    d = report.paired([1, 2, 3], [0, 0, 0])
    assert d["diff"] == 2
    assert d["se"] == pytest.approx(1 / math.sqrt(3))       # SD 1 (ddof=1) / sqrt(3)
    assert d["z"] == pytest.approx(2 * math.sqrt(3))
    assert math.isnan(report.paired([1], [0])["se"])


def test_calibration_bins():
    p = [0.52, 0.48, 0.57, 0.25, 1.0, 0.95]   # confidences 0.52, 0.52, 0.57, 0.75, 1.0, 0.95
    won = [True, True, False, False, True, False]
    bins = {(lo, hi): (n, conf, acc) for lo, hi, n, conf, acc in report.calibration_bins(p, won)}
    assert bins[(0.5, 0.55)] == (2, pytest.approx(0.52), 0.5)          # one home pick won, one away pick lost
    assert bins[(0.55, 0.6)] == (1, pytest.approx(0.57), 0.0)
    assert bins[(0.75, 0.8)] == (1, pytest.approx(0.75), 1.0)          # P(home) 0.25 -> away pick, home lost
    assert bins[(0.9, 1.0)] == (2, pytest.approx(0.975), 0.5)          # 1.0 lands in the last bin
    assert sum(b[0] for b in bins.values()) == len(p)


def test_disagreements():
    d = report.disagreements(p_model=[0.6, 0.4, 0.7, 0.55, 0.3], p_vegas=[0.4, 0.6, 0.8, 0.5, 0.2],
                             home_won=[True, True, True, False, False])
    # game 1: model home, Vegas away, home won -> model right; game 2: model away, Vegas home, home won -> Vegas
    # game 4: Vegas pick'em -> not a disagreement
    assert d == {"games": 2, "model_right": 1, "vegas_right": 1, "vegas_pickem": 1}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def game(gid, day, home, away, home_won):
    return {"game_id": gid, "date": day, "home": home, "away": away, "home_won": home_won}


SEASON_GAMES = [
    game("0022400001", "2024-10-22", "A", "B", True),
    game("0022400002", "2024-10-22", "C", "D", False),
    game("0022400003", "2024-10-23", "A", "C", False),   # same day as nothing; A 1-0, C 0-1
    game("0022400004", "2024-10-23", "B", "D", True),
    game("0022400005", "2024-10-24", "C", "A", True),    # A 1-1, C 1-1
    game("0022500001", "2025-10-21", "A", "C", True),    # new season: both 0-0
]


def test_records_before_counts_only_earlier_days_of_the_same_season():
    r = report.records_before(SEASON_GAMES)
    assert r["0022400001"] == ((0, 0), (0, 0))
    assert r["0022400002"] == ((0, 0), (0, 0))           # the same day's other game doesn't count
    assert r["0022400003"] == ((1, 0), (0, 1))
    assert r["0022400004"] == ((0, 1), (1, 0))
    assert r["0022400005"] == ((1, 1), (1, 1))
    assert r["0022500001"] == ((0, 0), (0, 0))


def test_better_record_side():
    assert report.better_record_side(((3, 1), (1, 3))) == 1
    assert report.better_record_side(((1, 3), (3, 1))) == -1
    assert report.better_record_side(((2, 2), (1, 1))) == 0
    assert report.better_record_side(((0, 0), (1, 1))) == 0      # no games = .500
    assert report.better_record_side(((0, 0), (0, 1))) == 1


def test_fit_baselines_and_probabilities():
    fitted = report.fit_baselines(SEASON_GAMES)
    assert fitted["home"] == pytest.approx(4 / 6)
    # different records only in 0022400003 (better: A at home, lost) and 0022400004 (better: D away, lost)
    assert fitted["record_n"] == 2 and fitted["record"] == 0.0
    probs = report.baseline_probs(SEASON_GAMES, {"home": 0.55, "record": 0.65}, report.records_before(SEASON_GAMES))
    assert probs["home"].tolist() == [0.55] * 6
    assert probs["record"] == pytest.approx([0.55, 0.55, 0.65, 0.35, 0.55, 0.55])   # equal records -> home


# ---------------------------------------------------------------------------
# wording rules for the summary and the resume bullets
# ---------------------------------------------------------------------------

def test_verdict_needs_two_standard_errors():
    assert report.verdict({"diff": -0.01, "se": 0.004, "z": -2.5}, lower_is_better=True) == "better"
    assert report.verdict({"diff": 0.01, "se": 0.004, "z": 2.5}, lower_is_better=True) == "worse"
    assert report.verdict({"diff": 0.01, "se": 0.006, "z": 1.7}, lower_is_better=True) == "same"
    assert report.verdict({"diff": 0.03, "se": 0.01, "z": 3.0}, lower_is_better=False) == "better"
    assert report.verdict({"diff": 1.0, "se": float("nan"), "z": float("nan")}, lower_is_better=True) == "same"


def test_baseline_phrase_never_claims_an_unearned_win():
    model, base = {"accuracy": 0.625, "brier": 0.2255}, {"accuracy": 0.652, "brier": 0.2266}
    d = {"accuracy": {"diff": -0.027, "se": 0.01, "z": -2.7}, "brier": {"diff": -0.001, "se": 0.0027, "z": -0.4}}
    text = report.baseline_phrase("the better record", d, model, base)
    assert text.startswith("was level with the better record on Brier")
    assert "trailed it on winners picked (62.5% vs 65.2%)" in text
    assert "beat" not in text


def test_significance_sentence():
    worse = {"diff": 0.02, "se": 0.002, "z": 10.0}
    level = {"diff": 0.001, "se": 0.002, "z": 0.5}
    assert report.significance_sentence({"brier": worse, "accuracy": level}) == "Vegas is better by 2+ SE on Brier."
    assert report.significance_sentence({"brier": level}) == "No metric differs from Vegas by 2 SE or more."


# ---------------------------------------------------------------------------
# 2026-27 sample: logs + closing lines
# ---------------------------------------------------------------------------

LOG_ROWS = [
    # the model favored the AWAY team in the second game: P(home) = 1 - 0.60
    ["2026-11-20", "0022600280", "BOS", "NYK", "BOS", "64.00", "112.0", "108.0", "110", "100"],
    ["2026-11-20", "0022600281", "LAL", "GSW", "GSW", "60.00", "109.0", "111.0", "115", "105"],
    ["2026-11-21", "0022600290", "BOS", "LAL", "BOS", "55.00", "110.0", "109.0", "100", "104"],
]
LINE_FIELDS = __import__("lines").FIELDS


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def line_row(fetched, gid, tipoff, spread, total, prob=""):
    row = {k: "" for k in LINE_FIELDS}
    row.update({"fetched_at": fetched, "game_id": gid, "tipoff_utc": tipoff, "spread_home": spread, "total": total,
                "home_win_prob": prob})
    return [row[k] for k in LINE_FIELDS]


@pytest.fixture
def season_files(tmp_path):
    log = write_csv(tmp_path / "log.csv", __import__("predlog").COLUMNS, LOG_ROWS)
    later = write_csv(tmp_path / "later.csv", __import__("predlog").COLUMNS,
                      [LOG_ROWS[0][:5] + ["99.00"] + LOG_ROWS[0][6:],             # duplicate: the first log wins
                       ["2026-11-22", "0022600300", "NYK", "BOS", "NYK", "51.00", "105.0", "104.0", "99", "101"]])
    lines_csv = write_csv(tmp_path / "lines.csv", LINE_FIELDS, [
        line_row("2026-11-20T15:00:00Z", "0022600280", "2026-11-21T00:30:00Z", "-5.5", "221.5", "0.6600"),
        line_row("2026-11-20T23:30:00Z", "0022600280", "2026-11-21T00:30:00Z", "-4.5", "220.0", "0.6400"),  # closing
        line_row("2026-11-21T01:00:00Z", "0022600280", "2026-11-21T00:30:00Z", "-12.0", "230.0", "0.9000"),  # live
        line_row("2026-11-20T18:00:00Z", "0022600281", "2026-11-21T03:00:00Z", "3.0", "232.0", ""),        # no ML
        line_row("2026-11-21T02:00:00Z", "0022600290", "2026-11-21T01:00:00Z", "-2.0", "225.0", ""),       # after tip
    ])
    return [log, later], lines_csv


def test_log_row_to_game_uses_home_probability():
    away_pick = report.log_row_to_game(dict(zip(__import__("predlog").COLUMNS, LOG_ROWS[1])))
    assert away_pick["p_model"] == pytest.approx(0.40)
    assert (away_pick["home_won"], away_pick["margin"], away_pick["total"]) == (True, 10, 220)
    assert (away_pick["proj_margin"], away_pick["proj_total"]) == (-2.0, 220.0)


def test_season_sample_takes_closing_lines_only(season_files):
    logs, lines_csv = season_files
    s = report.season_sample(logs, lines_csv)
    assert [g["game_id"] for g in s["games"]] == ["0022600280", "0022600281", "0022600290", "0022600300"]
    assert s["games"][0]["p_model"] == pytest.approx(0.64)        # not the duplicate's 99%
    rows = {r["game_id"]: r for r in s["rows"]}
    assert set(rows) == {"0022600280", "0022600281"}               # 0022600290's only snapshot is after tip-off
    assert (rows["0022600280"]["spread_home"], rows["0022600280"]["vegas_total"]) == (-4.5, 220.0)
    assert rows["0022600280"]["p_ml"] == pytest.approx(0.64)
    assert rows["0022600281"]["p_ml"] is None
    assert s["hours_before"] == pytest.approx((1.0 + 9.0) / 2)     # 23:30 -> 00:30 and 18:00 -> 03:00


def test_build_writes_the_report_and_chart(tmp_path, season_files, monkeypatch):
    """The whole report on synthetic training rows: markdown + PNG in a temp folder, every section present."""
    rng = np.random.default_rng(3)
    rows, games = [], []
    for i in range(400):
        season = "2024-25" if i < 200 else "2025-26"
        gid = f"00{'224' if i < 200 else '225'}{i:05d}"
        day = f"{2024 if i < 200 else 2025}-11-{1 + i % 28:02d}"
        spread = float(np.round(rng.uniform(-12, 12) * 2) / 2)
        margin = int(round(-spread + rng.normal(0, 13)))
        margin = margin or 1
        p_model = float(np.clip(report.normal_cdf(-spread / 14 + rng.normal(0, 0.3)), 0.02, 0.98))
        rows.append({"game_id": gid, "date": day, "season": season, "game_type": "Regular Season",
                     "home_won": margin > 0, "margin": margin, "total": 225 + int(rng.normal(0, 15)),
                     "p_model": p_model, "proj_margin": -spread + rng.normal(0, 3), "proj_total": 226.0,
                     "spread_home": spread, "vegas_total": 225.5, "p_ml": None})
        games.append({"game_id": gid, "date": day, "home": f"T{i % 30}", "away": f"T{(i + 7) % 30}",
                      "home_won": margin > 0})
    pass_rows = [{**r, "p_model": float(np.clip(r["p_model"] + rng.normal(0, 0.05), 0.01, 0.99))} for r in rows[2:]]
    train = {"rows": rows, "pass_rows": pass_rows, "scored": len(pass_rows),
             "live": {"n": 20, "m": 30, "seasons": {
                 s: {"cutoff": f"{s[:4]}-10-03", "trained_on": "x to y", "train_games": 2600, "games": 200, "skipped": []}
                 for s in ("2024-25", "2025-26")}},
             "sd": report.fit_spread_sd([r["margin"] for r in rows], [r["spread_home"] for r in rows]),
             "baselines": report.fit_baselines(games), "records": report.records_before(games),
             "match": {"matched": 400, "games": 400, "rate": 1.0}, "first": "2023-10-24", "last": "2026-06-13"}
    monkeypatch.setattr(report, "training_sample", lambda training_csv, betting_csv, calib="off", react="off": train)
    betting = tmp_path / "betting.csv"
    betting.write_text("season,date\n", encoding="utf-8")
    logs, lines_csv = season_files
    out = tmp_path / "reports" / "accuracy_report.md"

    result = report.build("unused.csv", betting, logs, lines_csv, out)
    text = out.read_text(encoding="utf-8")
    png = out.parent / report.CHART_NAME
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for heading in ("## Summary", "## Resume bullets", "## Method", "## Training seasons, main: live-procedure backtest (400",
                    "## Training seasons, secondary: training-pass backtest (398", "**Live procedure − training pass** on the 398",
                    "**Paired differences vs Vegas",
                    "**Model vs the baselines**", "**Disagreements:**", "**Calibration**", "## 2026-27 season",
                    "### With Vegas closing lines (2 games)", "**Games with a moneyline:**", "## Caveats"):
        assert heading in text, heading
    bullets = text.split("## Resume bullets")[1].split("## Method")[0]
    assert 3 <= bullets.count("\n- ") <= 4
    assert result["season"]["vegas"]["n"] == 2 and result["season"]["all"]["n"] == 4
    assert result["train"]["result"]["n"] == 400


def test_calibration_sentence_uses_two_binomial_ses():
    # n = 2,500 at 66% accuracy: SE ~ 0.95 pts, so a 4-point gap is underconfident and a 1-point gap is fine
    assert "underconfident" in report.calibration_sentence("x", {"n": 2500, "accuracy": 0.66, "confidence": 0.62})
    assert "overconfident" in report.calibration_sentence("x", {"n": 2500, "accuracy": 0.62, "confidence": 0.66})
    assert "calibrated within noise" in report.calibration_sentence("x", {"n": 2500, "accuracy": 0.66, "confidence": 0.65})
    assert "calibrated within noise" in report.calibration_sentence("x", {"n": 100, "accuracy": 0.66, "confidence": 0.62})


def test_compare_procedures_pairs_common_games_only():
    def row(gid, p, won, proj, margin):
        return {"game_id": gid, "p_model": p, "home_won": won, "proj_margin": proj, "margin": margin}
    live = [row("a", 0.8, True, 6, 10), row("b", 0.6, False, 2, -3), row("c", 0.7, True, 4, 5)]
    old = [row("a", 0.6, True, 2, 10), row("b", 0.4, False, -1, -3), row("z", 0.5, True, 0, 1)]
    proc = report.compare_procedures(live, old)
    assert proc["n"] == 2                                              # "c" and "z" aren't in both
    # Brier live (0.04, 0.36) vs old (0.16, 0.16): diffs -0.12 and +0.20
    assert proc["brier"]["diff"] == pytest.approx(0.04)
    assert proc["accuracy"]["diff"] == pytest.approx(-0.5)            # live misses "b", old gets it
    assert proc["margin_mae"]["diff"] == pytest.approx(((4 - 8) + (5 - 2)) / 2)
    assert report.compare_procedures(live, [])["n"] == 0


def test_cli_refuses_calib_with_the_live_logs(tmp_path):
    """--calib on with the default (uncalibrated live) logs would mix two models in one report."""
    betting = tmp_path / "b.csv"
    betting.write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main(["--betting-csv", str(betting), "--calib", "on", "--out", str(tmp_path / "r.md")])


def test_cli_refuses_react_with_the_live_logs(tmp_path, capsys):
    """--react on with the default (live N, M) logs would mix two models in one report."""
    betting = tmp_path / "b.csv"
    betting.write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main(["--betting-csv", str(betting), "--react", "on", "--out", str(tmp_path / "r.md")])
    assert "--react on needs --log" in capsys.readouterr().err


def test_cli_checks_its_inputs(tmp_path):
    with pytest.raises(SystemExit):
        report.main(["--betting-csv", str(tmp_path / "missing.csv")])
    betting = tmp_path / "b.csv"
    betting.write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main(["--betting-csv", str(betting), "--out", str(tmp_path / "priors.json")])
