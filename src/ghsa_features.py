"""Compute GHSA features from the t0 snapshot and targets from the target_end snapshot.

Features use only what was known at t0, including severity. Targets use severity and status at
the end of the outcome window. Names are matched exactly.

Usage: python -u -m src.ghsa_features [--packages data/packages_test.csv]
"""
import argparse
import csv
import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import config
from src.advisories import tracked_files
from src.select_packages import read_csv

log = logging.getLogger("ghsa_features")

FEATURE_SNAPSHOT, TARGET_SNAPSHOT = "t0", "target_end"
SEVERITIES = config.SEVERITIES
KNOWN_SEVERITIES = set(SEVERITIES) - {"unknown"}
MONTHS = range(config.FEATURE_MONTHS)
GENERAL_COLUMNS = [f"ghsa_{severity}_count" for severity in SEVERITIES]
MONTHLY_COLUMNS = [f"ghsa_{severity}_count_M{month}" for severity in SEVERITIES for month in MONTHS]
TARGET_COLUMNS = [f"target_ghsa_{severity}_count_90d" for severity in SEVERITIES]
COLUMNS = ["package_name", "prediction_date", *GENERAL_COLUMNS, *MONTHLY_COLUMNS, *TARGET_COLUMNS]


def normalize_severity(value, ghsa_id=""):
    """Return a schema severity. Missing or unrecognised values become unknown."""
    severity = value.strip().lower() if isinstance(value, str) else None
    if severity in KNOWN_SEVERITIES:
        return severity
    if value is not None:
        log.warning("Unrecognised severity %r on %s counted as unknown", value, ghsa_id)
    return "unknown"


def load_active(db_path, snapshot):
    """Return (ghsa_id, package, published, severity) for advisories not withdrawn in a snapshot."""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT ghsa_id, package_name, published, severity FROM advisories"
        " WHERE snapshot = ? AND withdrawn IS NULL",
        (snapshot,),
    ).fetchall()
    db.close()
    if not rows:
        raise ValueError(f"No {snapshot} advisories in {db_path}; run src.advisories on that snapshot")
    return [
        (ghsa_id, package, config.parse_timestamp(published), normalize_severity(severity, ghsa_id))
        for ghsa_id, package, published, severity in rows
        if published
    ]


def count_features(advisories, t0):
    """Return {package: {column: set of GHSA ids}} from t0-snapshot advisories published before t0."""
    windows = config.feature_months(t0)
    result = defaultdict(lambda: defaultdict(set))
    for ghsa_id, package, published, severity in advisories:
        if published >= t0:
            continue
        result[package][f"ghsa_{severity}_count"].add(ghsa_id)
        for month, (start, end) in windows.items():
            if start <= published < end:
                result[package][f"ghsa_{severity}_count_M{month}"].add(ghsa_id)
                break
    return result


def count_targets(advisories, t0):
    """Return {package: {column: set of GHSA ids}} from advisories published in the target window."""
    start, end = config.target_window(t0)
    result = defaultdict(lambda: defaultdict(set))
    for ghsa_id, package, published, severity in advisories:
        if start <= published < end:
            result[package][f"target_ghsa_{severity}_count_90d"].add(ghsa_id)
    return result


def build_rows(names, features, targets, t0):
    """One row per package. A package with no advisories gets 0 in every column."""
    rows = []
    for name in names:
        counted = {**features.get(name, {}), **targets.get(name, {})}
        row = {"package_name": name, "prediction_date": config.format_timestamp(t0)}
        row.update({column: len(counted.get(column, ())) for column in COLUMNS[2:]})
        rows.append(row)
    return rows


def validate(rows):
    """Fail loudly if a package's monthly counts exceed its whole-history count."""
    for row in rows:
        for severity in SEVERITIES:
            monthly = sum(row[f"ghsa_{severity}_count_M{month}"] for month in MONTHS)
            if monthly > row[f"ghsa_{severity}_count"]:
                raise ValueError(f"{row['package_name']} {severity}: monthly {monthly} above general")


def is_positive(row):
    return any(row[column] > 0 for column in TARGET_COLUMNS)


def recount_targets_sql(db_path, names, t0):
    """Recount targets by direct SQL, sharing no counting code with the main path."""
    start, end = (config.format_timestamp(value) for value in config.target_window(t0))
    db = sqlite3.connect(db_path)
    db.execute("CREATE TEMP TABLE wanted (name TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO wanted VALUES (?)", [(name,) for name in names])
    rows = db.execute(
        """
        SELECT package_name,
               CASE WHEN LOWER(severity) IN ('low', 'moderate', 'high', 'critical')
                    THEN LOWER(severity) ELSE 'unknown' END AS level,
               COUNT(DISTINCT ghsa_id)
        FROM advisories JOIN wanted ON package_name = name
        WHERE snapshot = 'target_end' AND withdrawn IS NULL AND published >= ? AND published < ?
        GROUP BY package_name, level
        """,
        (start, end),
    ).fetchall()
    db.close()
    return {(name, level): count for name, level, count in rows}


def compare_targets(rows, recount):
    """Return the differences between main-path rows and the SQL recount."""
    main = {
        (row["package_name"], severity): row[f"target_ghsa_{severity}_count_90d"]
        for row in rows for severity in SEVERITIES
        if row[f"target_ghsa_{severity}_count_90d"]
    }
    return sorted(
        (key, main.get(key), recount.get(key))
        for key in set(main) | set(recount)
        if main.get(key) != recount.get(key)
    )


def snapshot_rows(db_path, snapshot):
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT ghsa_id, package_name, published, withdrawn, severity FROM advisories WHERE snapshot = ?",
        (snapshot,),
    ).fetchall()
    db.close()
    return rows


def reviewed_after_t0(db_path, t0, t0_snapshot_ids):
    """Target_end advisories published before t0 that the t0 snapshot lacks: neither feature nor target."""
    t0_text = config.format_timestamp(t0)
    t0_rows = snapshot_rows(db_path, FEATURE_SNAPSHOT)
    t0_ids = {row[0] for row in t0_rows}
    t0_pairs = {(row[0], row[1]) for row in t0_rows}
    late = [row for row in snapshot_rows(db_path, TARGET_SNAPSHOT) if row[2] and row[2] < t0_text]
    absent = {row[0] for row in late if row[0] not in t0_ids}
    return {
        "npm_ids_not_in_t0": sorted(absent),
        "not_active_at_target_end": sorted({row[0] for row in late if row[0] in absent and row[3]}),
        "file_absent_from_t0_snapshot": sorted(absent - t0_snapshot_ids),
        "file_in_t0_without_npm_entry": sorted(absent & t0_snapshot_ids),
        "packages_added_to_ids_in_t0": sorted(
            {(row[0], row[1]) for row in late if row[0] in t0_ids and (row[0], row[1]) not in t0_pairs}
        ),
        "packages": sorted({row[1] for row in late if row[0] in absent}),
    }


def severity_changes(db_path):
    """Return {ghsa_id: (t0 severity, target_end severity)} where the two differ."""
    before = {row[0]: normalize_severity(row[4], row[0]) for row in snapshot_rows(db_path, FEATURE_SNAPSHOT)}
    after = {row[0]: normalize_severity(row[4], row[0]) for row in snapshot_rows(db_path, TARGET_SNAPSHOT)}
    return {
        ghsa_id: (before[ghsa_id], after[ghsa_id])
        for ghsa_id in sorted(before.keys() & after.keys())
        if before[ghsa_id] != after[ghsa_id]
    }


def withdrawn_after_t0(db_path):
    active_at_t0 = {row[0] for row in snapshot_rows(db_path, FEATURE_SNAPSHOT) if not row[3]}
    withdrawn_later = {row[0] for row in snapshot_rows(db_path, TARGET_SNAPSHOT) if row[3]}
    return sorted(active_at_t0 & withdrawn_later)


def case_only_matches(db_path, names):
    """Advisory package names equal to a wanted package only when case is ignored."""
    lowered = defaultdict(set)
    for name in names:
        lowered[name.lower()].add(name)
    found = set()
    for snapshot in (FEATURE_SNAPSHOT, TARGET_SNAPSHOT):
        for _, package, *_ in snapshot_rows(db_path, snapshot):
            if package.lower() in lowered and package not in lowered[package.lower()]:
                found.add((snapshot, package, tuple(sorted(lowered[package.lower()]))))
    return sorted(found)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact(row):
    """A row's non-zero counts, for reading."""
    return {key: value for key, value in row.items() if key in ("package_name", "prediction_date") or value}


def report(rows, late, changes, withdrawn, case_only, whole):
    print(f"\n=== Report: {len(rows)} rows, {len(COLUMNS)} columns ===")
    print("Package list positives per severity:",
          {severity: sum(row[f"target_ghsa_{severity}_count_90d"] > 0 for row in rows) for severity in SEVERITIES})
    print("Package list positives, any severity:", [row["package_name"] for row in rows if is_positive(row)])
    print(f"Reviewed after t0: {len(late['npm_ids_not_in_t0'])} GHSA ids, {len(late['packages'])} packages")
    print(f"  file absent from t0 snapshot: {len(late['file_absent_from_t0_snapshot'])}")
    print(f"  file in t0 but no npm entry then: {len(late['file_in_t0_without_npm_entry'])}")
    print(f"  of these, withdrawn at target_end: {len(late['not_active_at_target_end'])}")
    print(f"  (GHSA, package) pairs added to advisories already in t0: {len(late['packages_added_to_ids_in_t0'])}")
    print(f"  examples: {late['npm_ids_not_in_t0'][:5]}")
    print(f"Severity changed between t0 and target_end: {len(changes)}")
    print(f"  direction: {dict(Counter(changes.values()))}")
    print(f"  examples: {dict(list(changes.items())[:5])}")
    print(f"Active at t0, withdrawn by target_end: {len(withdrawn)} {withdrawn[:5]}")
    print(f"Case-only name matches for this package list: {case_only}")
    print("Whole package list recount (separate SQL):")
    print(f"  packages {whole['packages']}, positives {whole['positives']}, per stratum {whole['per_stratum']}")
    print(f"  differences from main code: {len(whole['differences'])} {whole['differences'][:10]}")
    print(f"  monthly <= general passed for {whole['packages']} packages")
    print("Example rows, non-zero values only:")
    positive = [row for row in rows if is_positive(row)][:2]
    other = [row for row in rows if not is_positive(row) and row["ghsa_low_count"] + row["ghsa_moderate_count"] + row["ghsa_high_count"] + row["ghsa_critical_count"]][:1]
    zero = [row for row in rows if compact(row).keys() == {"package_name", "prediction_date"}][:1]
    for row in positive + other + zero:
        print(f"  {compact(row)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=config.PACKAGES_TEST_CSV)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "ghsa_features.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # The advisory snapshots are tied to one t0, so another date needs its own snapshots.
    if len(config.PREDICTION_DATES) != 1:
        raise ValueError("ghsa_features supports one prediction date until more snapshots exist")
    t0 = config.parse_timestamp(config.PREDICTION_DATES[0])
    db_path = config.ADVISORIES_SQLITE
    features = count_features(load_active(db_path, FEATURE_SNAPSHOT), t0)
    targets = count_targets(load_active(db_path, TARGET_SNAPSHOT), t0)

    names = [row["package_name"] for row in read_csv(args.packages)]
    rows = build_rows(names, features, targets, t0)
    validate(rows)
    write_rows(config.GHSA_FEATURES_CSV, rows)

    package_rows = read_csv(config.PACKAGES_CSV)
    all_names = [row["package_name"] for row in package_rows]
    all_rows = build_rows(all_names, features, targets, t0)
    validate(all_rows)
    positive_names = {row["package_name"] for row in all_rows if is_positive(row)}
    whole = {
        "packages": len(all_rows),
        "positives": len(positive_names),
        "per_stratum": dict(Counter(row["stratum"] for row in package_rows if row["package_name"] in positive_names)),
        "differences": compare_targets(all_rows, recount_targets_sql(db_path, all_names, t0)),
    }
    t0_snapshot_ids = {
        Path(path).stem for path in tracked_files(config.ADVISORY_SNAPSHOT_DIR / FEATURE_SNAPSHOT)
    }
    report(
        rows,
        reviewed_after_t0(db_path, t0, t0_snapshot_ids),
        severity_changes(db_path),
        withdrawn_after_t0(db_path),
        case_only_matches(db_path, names),
        whole,
    )


if __name__ == "__main__":
    main()
