"""
Side-by-side accuracy of two or more prediction logs (predlog.py format).

    python code/compare.py log_base.csv log_m20.csv [more.csv ...]

Metrics are computed over the games that are in EVERY log, so each column is scored on
the same games; how many games each log has in total is shown too. The best value in
each row is marked with *. "avg confidence" (mean win % of the predicted winner) has no
best: a model is well calibrated when it matches its winner accuracy. Likewise "projected
total" should match "actual total", and "margin error SD" (how far real margins land from
the projected margin) is what the simulation's own margin spread should be.
"""
import argparse
import sys
from pathlib import Path

import predlog

ROWS = [  # (label, metric key, format, higher is better; None = no "best")
    ("winner accuracy", "accuracy", "{:.1%}", True),
    ("avg confidence", "confidence", "{:.1%}", None),   # compare with the accuracy: calibration
    ("spread MAE (pts)", "spread_mae", "{:.2f}", False),
    ("Brier score", "brier", "{:.4f}", False),
    ("projected total", "proj_total", "{:.1f}", None),   # compare with the actual total: bias
    ("actual total", "actual_total", "{:.1f}", None),
    ("margin error SD", "margin_sd", "{:.2f}", None),    # SD of actual - projected margin
]


def compare(paths) -> str:
    logs = {Path(p).stem: predlog.read_log(p) for p in paths}
    if len(logs) < len(paths):
        raise ValueError("log file names must be different")
    common = set.intersection(*({r["game_id"] for r in rows} for rows in logs.values()))
    results = {name: predlog.metrics([r for r in rows if r["game_id"] in common]) for name, rows in logs.items()}

    width = max(12, *(len(name) + 2 for name in logs))
    lines = [f"{'':<18}" + "".join(f"{name:>{width}}" for name in logs),
             f"{'games in log':<18}" + "".join(f"{len(rows):>{width}}" for rows in logs.values()),
             f"{'games compared':<18}" + "".join(f"{len(common):>{width}}" for _ in logs)]
    if common:
        for label, key, fmt, higher in ROWS:
            values = [results[name][key] for name in logs]
            best = None if higher is None else (max(values) if higher else min(values))
            cells = [fmt.format(v) + ("*" if v == best and len(set(values)) > 1 else " ") for v in values]
            lines.append(f"{label:<18}" + "".join(f"{c:>{width}}" for c in cells))
    else:
        lines.append("(no game is in every log, nothing to compare)")
    if any(len(rows) != len(common) for rows in logs.values()):
        lines.append(f"\nOnly the {len(common)} games present in every log are scored.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare prediction logs side by side.")
    parser.add_argument("logs", nargs="+", metavar="LOG_CSV")
    args = parser.parse_args(argv)
    if len(args.logs) < 2:
        parser.error("give at least two log files")
    missing = [p for p in args.logs if not Path(p).exists()]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    try:
        print(compare(args.logs))
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    sys.exit(main())
