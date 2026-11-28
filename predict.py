"""Predict new GHSA counts for the next 90 days for npm packages in the PCOM-01 record schema.

Usage:
  .venv\\Scripts\\python.exe predict.py input.csv [-o output.csv] [--strategy S0|S1|S2]
                             [--float] [--float-strategy EV|HR|HRMAX]

Writes exactly the 5 target columns, one row per input row in the same order, and prints a
table for typing the results into a form. Exit codes: 0 = done; 1 = output written but some or
all rows fell back to zeros; 2 = the input could not be read, so no output was written.
"""
import time

STARTED = time.perf_counter()

import argparse
import csv
import hashlib
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
FLOAT_STRATEGIES = ("EV", "HR", "HRMAX")  # the same names as src/model/counts.py


def warn(message):
    print(f"WARNING: {message}", flush=True)


def read_rows(path, known_names=()):
    """Return (rows as dicts, column names).

    Column names are trimmed and matched case-insensitively, then mapped back to their exact
    schema spelling (the monthly columns end in a capital M, as in ghsa_low_count_M0).
    """
    canonical = {name.lower(): name for name in known_names}
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        warn("the file is not valid UTF-8; reading it as Windows-1252 (Latin-1)")
        text = data.decode("cp1252", errors="replace")
    first_line = text.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = next(reader, None)
    if header is None:
        return [], []
    columns = [canonical.get(name.strip().lower(), name.strip().lower()) for name in header]
    rows, skipped = [], 0
    for values in reader:
        if not any(value.strip() for value in values):
            skipped += 1  # a blank line or a row of empty cells is not a package
            continue
        row = {}
        for name, value in zip(columns, values):
            value = value.strip()
            if name and name not in row:
                row[name] = None if value.lower() in MISSING_TOKENS else value
        rows.append(row)
    if skipped:
        warn(f"skipped {skipped} empty row(s); the output has one row per non-empty input row")
    return rows, columns


def write_output(path, values, as_float=False):
    """Integers by default (the schema); with as_float, expected values with up to 6 decimals."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TARGET_COLUMNS)
        if as_float:
            writer.writerows([[f"{float(value):.6f}" for value in row] for row in values])
        else:
            writer.writerows([[int(value) for value in row] for row in values])


def load_lightgbm():
    """Import LightGBM without its optional scikit-learn and pandas support.

    Only the core Booster is needed here, and importing scikit-learn alone took about 2.4 s.
    Marking the optional packages as unavailable makes LightGBM skip them.
    """
    for name in ("sklearn", "pandas", "dask", "pyarrow"):
        sys.modules.setdefault(name, None)
    import lightgbm

    return lightgbm


def resolve_float_strategy(meta, requested=None):
    """The float strategy to use: the requested one, else default_float_strategy, else EV.

    An unknown requested name raises; an unknown name in model_meta.json warns and uses EV.
    """
    if requested:
        if requested not in FLOAT_STRATEGIES:
            raise ValueError(f"unknown float strategy {requested!r}; choose one of {', '.join(FLOAT_STRATEGIES)}")
        return requested
    default = meta.get("counts", {}).get("default_float_strategy") or "EV"
    if default not in FLOAT_STRATEGIES:
        warn(f"model_meta.json names an unknown float strategy {default!r}; using EV")
        return "EV"
    return default


def score_rows(rows, strategy, float_strategy=None, details=None):
    """Return (counts, scores, rows that fell back to zeros, float values or None) per row.

    The float values come from float_strategy (EV, HR or HRMAX; default: default_float_strategy in
    model_meta.json, else EV). They are None when model_meta.json has no EV settings. If details is
    a dict, the float strategy used is stored in details["float_strategy"].
    """
    import numpy as np

    lgb = load_lightgbm()
    from src.model import counts as ct
    from src.model import features as ft

    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    # A damaged model file can crash LightGBM's C++ code outright, which no Python error handling
    # can catch. So normalise line endings (git may have converted them) and check the hash first.
    model_text = (MODEL_DIR / "model.txt").read_bytes().decode("utf-8").replace("\r\n", "\n")
    expected_hash = meta.get("model_sha256")
    if expected_hash and hashlib.sha256(model_text.encode("utf-8")).hexdigest() != expected_hash:
        raise RuntimeError("model.txt does not match model_meta.json (changed or damaged); "
                           "restore models/pcom_entry_1 from git")
    booster = lgb.Booster(model_str=model_text)
    X, names = ft.build_features(rows, None, use_reference_features=meta["use_reference_features"])
    if not (booster.feature_name() == meta["feature_names"] == names):
        raise RuntimeError("model.txt, model_meta.json and features.py disagree on the feature list; "
                           "rebuild the model with: python -m src.model.build")
    settings = meta["counts"]["strategies"]
    strategy = strategy or meta["counts"]["default_strategy"]
    float_strategy = resolve_float_strategy(meta, float_strategy)
    if details is not None:
        details["float_strategy"] = float_strategy

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
    expected = None
    if "EV" in settings:
        expected = ct.predict_float(scores, rows, settings, float_strategy)
        expected[failed] = 0.0
        expected = expected.tolist()
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
    print(f"Float strategy {float_strategy}: {float_strategy_note(float_strategy, settings)}")
    return counts.tolist(), scores.tolist(), int(failed.sum()), expected


def float_strategy_note(strategy, settings):
    divisor = settings.get("HR", {}).get("divisor", 8)
    if strategy == "HR":
        return f"per severity, GHSAs in the last 12 months / {divisor:g}"
    if strategy == "HRMAX":
        return f"per severity, max(EV bucket mean, GHSAs in the last 12 months / {divisor:g})"
    return "EV bucket means"


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


def print_float_table(rows, expected, float_strategy=None):
    """Float values (EV, HR or HRMAX), rounded to 3 decimals, for a form that asks for floats."""
    width = min(60, max([len(row.get("purl") or "") for row in rows] + [4]))
    print("\nValues for the form (float):")
    if float_strategy:
        print(f"(float strategy {float_strategy})")
    print(f"{'row':>4}  {'purl':<{width}}  {'low':>7} {'mod':>7} {'high':>7} {'crit':>7} {'unkn':>7}")
    for index, (row, values) in enumerate(zip(rows, expected), 1):
        purl = (row.get("purl") or "-")[:width]
        print(f"{index:>4}  {purl:<{width}}  " + " ".join(f"{value:7.3f}" for value in values))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict the 5 target_ghsa_*_count_90d values for each package in a PCOM-01 CSV.",
        epilog="Example: .venv\\Scripts\\python.exe predict.py input.csv",
    )
    parser.add_argument("input", help="input CSV in the PCOM-01 record schema")
    parser.add_argument("-o", "--output", help="output CSV (default: <input name>_predictions.csv next to the input)")
    parser.add_argument("--strategy", choices=STRATEGIES, help="score-to-counts rule (default: from model_meta.json, S2)")
    parser.add_argument("--float", dest="as_float", action="store_true",
                        help="write the float values (see --float-strategy) to the output CSV instead of integers")
    parser.add_argument("--float-strategy", choices=FLOAT_STRATEGIES,
                        help="float values: EV, HR (half-rate) or HRMAX (default: from model_meta.json)")
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
    expected = None
    details = {}
    if rows:
        try:
            counts, scores, failed, expected = score_rows(rows, args.strategy, args.float_strategy, details)
            if failed:
                exit_code = 1
        except Exception as error:
            warn(f"the model could not be used ({type(error).__name__}: {error})")
            warn("ALL predictions below are zeros")
            counts, scores, exit_code = [[0] * 5 for _ in rows], [float("nan")] * len(rows), 1
            expected = [[0.0] * 5 for _ in rows]
        print()
        print_table(rows, counts, scores)
        if expected is None:
            warn("model_meta.json has no EV settings; float values are zeros (rebuild: python -m src.model.build)")
            expected = [[0.0] * 5 for _ in rows]
            exit_code = max(exit_code, 1) if args.as_float else exit_code
        print_float_table(rows, expected, details.get("float_strategy"))
    else:
        counts = []
        warn("the input has no data rows; writing a header-only file")

    try:
        write_output(output_path, expected if args.as_float and rows else counts, args.as_float)
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
