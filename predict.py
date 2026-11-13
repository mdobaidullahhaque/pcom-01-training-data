"""Predict new GHSA counts for the next 90 days for npm packages in the PCOM-01 record schema.

Usage:
  .venv\\Scripts\\python.exe predict.py input.csv [-o output.csv] [--strategy S0|S1|S2]

Writes exactly the 5 target columns, one row per input row in the same order, and prints a
table for typing the results into a form. Exit codes: 0 = done; 1 = output written but some or
all rows fell back to zeros; 2 = the input could not be read, so no output was written.
"""
import time

STARTED = time.perf_counter()

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "pcom_entry_1"
TARGET_COLUMNS = [
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
]
MISSING_TOKENS = {"", "null", "na", "nan", "none"}
STRATEGIES = ("S0", "S1", "S2")


def warn(message):
    print(f"WARNING: {message}", flush=True)


def read_rows(path, known_names=()):
    """Return (rows as dicts, column names).

    Column names are trimmed and matched case-insensitively, then mapped back to their exact
    schema spelling (the monthly columns end in a capital M, as in ghsa_low_count_M0).
    """
    canonical = {name.lower(): name for name in known_names}
    text = Path(path).read_bytes().decode("utf-8-sig", errors="replace")
    first_line = text.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = next(reader, None)
    if header is None:
        return [], []
    columns = [canonical.get(name.strip().lower(), name.strip().lower()) for name in header]
    rows = []
    for values in reader:
        if not any(value.strip() for value in values):
            continue  # skip blank lines
        row = {}
        for name, value in zip(columns, values):
            value = value.strip()
            if name and name not in row:
                row[name] = None if value.lower() in MISSING_TOKENS else value
        rows.append(row)
    return rows, columns


def write_output(path, counts):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TARGET_COLUMNS)
        writer.writerows([[int(value) for value in row] for row in counts])


def load_lightgbm():
    """Import LightGBM without its optional scikit-learn and pandas support.

    Only the core Booster is needed here, and importing scikit-learn alone took about 2.4 s.
    Marking the optional packages as unavailable makes LightGBM skip them.
    """
    for name in ("sklearn", "pandas", "dask", "pyarrow"):
        sys.modules.setdefault(name, None)
    import lightgbm

    return lightgbm


def score_rows(rows, strategy):
    """Return (counts per row, scores per row, number of rows that fell back to zeros)."""
    import numpy as np

    lgb = load_lightgbm()
    from src.model import counts as ct
    from src.model import features as ft

    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    X, names = ft.build_features(rows, None, use_reference_features=meta["use_reference_features"])
    if not (booster.feature_name() == meta["feature_names"] == names):
        raise RuntimeError("model.txt, model_meta.json and features.py disagree on the feature list; "
                           "rebuild the model with: python -m src.model.build")
    settings = meta["counts"]["strategies"]
    strategy = strategy or meta["counts"]["default_strategy"]

    failed = np.zeros(len(rows), dtype=bool)
    try:
        probability = booster.predict(X)
    except Exception:  # fall back to one row at a time, so one bad row cannot spoil the rest
        probability = np.zeros(len(rows))
        for index in range(len(rows)):
            try:
                probability[index] = booster.predict(X[index:index + 1])[0]
            except Exception as error:
                failed[index] = True
                warn(f"row {index + 1}: scoring failed ({error}); predicting zeros")
    ghsa_total = np.nan_to_num(X[:, names.index("ghsa_total")])
    scores = ghsa_total + 0.5 * probability  # scorer D: past GHSA total, probability breaks ties
    scores[failed] = np.nan
    counts = ct.predict_counts(scores, settings, strategy)
    counts[failed] = 0
    empty = np.isnan(X).all(axis=1) & ~failed
    if empty.any():
        warn(f"{int(empty.sum())} row(s) had no usable feature values: {[int(i) + 1 for i in np.flatnonzero(empty)][:10]}")
    missing = sorted(ft.INPUT_COLUMNS - set(rows[0]) if rows else ft.INPUT_COLUMNS)
    if missing:
        print(f"{len(missing)} of {len(ft.INPUT_COLUMNS)} expected input columns are missing, used as NaN"
              f" (first: {', '.join(missing[:5])})")
    else:
        print(f"All {len(ft.INPUT_COLUMNS)} expected input columns found")
    print(f"Strategy {strategy}: {meta['counts']['strategies'].get(strategy, {}) and strategy_note(strategy, settings)}")
    return counts.tolist(), scores.tolist(), int(failed.sum())


def strategy_note(strategy, settings):
    if strategy == "S2":
        rule = settings["S2"]
        return f"score >= {rule['threshold']:.3f} predicts 1 {rule['severity']}"
    if strategy == "S1":
        return f"score buckets from {settings['S1']['edges']}"
    return "all zeros"


def print_table(rows, counts, scores):
    width = min(60, max([len(row.get("purl") or "") for row in rows] + [4]))
    print(f"{'row':>4}  {'purl':<{width}}  {'low':>4} {'mod':>4} {'high':>4} {'crit':>4} {'unkn':>4}  {'score':>7}")
    for index, (row, count, score) in enumerate(zip(rows, counts, scores), 1):
        purl = (row.get("purl") or "-")[:width]
        shown = "  n/a" if score != score else f"{score:7.3f}"  # NaN check without numpy
        print(f"{index:>4}  {purl:<{width}}  " + " ".join(f"{value:>4}" for value in count) + f"  {shown:>7}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict the 5 target_ghsa_*_count_90d values for each package in a PCOM-01 CSV.",
        epilog="Example: .venv\\Scripts\\python.exe predict.py input.csv",
    )
    parser.add_argument("input", help="input CSV in the PCOM-01 record schema")
    parser.add_argument("-o", "--output", help="output CSV (default: <input name>_predictions.csv next to the input)")
    parser.add_argument("--strategy", choices=STRATEGIES, help="score-to-counts rule (default: from model_meta.json, S2)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_predictions.csv")
    try:
        sys.path.insert(0, str(ROOT))
        from src.model.features import INPUT_COLUMNS

        known = {*INPUT_COLUMNS, "purl", "prediction_date", *TARGET_COLUMNS}
    except Exception as error:  # the column map is a convenience; reading still works without it
        warn(f"feature module not available ({error})")
        known = set()
    try:
        rows, _ = read_rows(input_path, known)
    except Exception as error:
        print(f"ERROR: cannot read {input_path}: {error}")
        print(f"Time: {time.perf_counter() - STARTED:.2f} s")
        return 2
    print(f"Read {len(rows)} row(s) from {input_path}")

    exit_code = 0
    if rows:
        try:
            counts, scores, failed = score_rows(rows, args.strategy)
            if failed:
                exit_code = 1
        except Exception as error:
            warn(f"the model could not be used ({type(error).__name__}: {error})")
            warn("ALL predictions below are zeros")
            counts, scores, exit_code = [[0] * 5 for _ in rows], [float("nan")] * len(rows), 1
        print()
        print_table(rows, counts, scores)
    else:
        counts = []
        warn("the input has no data rows; writing a header-only file")

    try:
        write_output(output_path, counts)
        print(f"\nOutput: {output_path.resolve()}")
    except Exception as error:
        print(f"ERROR: cannot write {output_path}: {error}")
        exit_code = 2
    print(f"Time: {time.perf_counter() - STARTED:.2f} s")
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
