"""Assemble the feature files into data/pcom01_training.sqlite and .csv, and validate them.

GitHub columns come only from cached results. In this initial version every other package, and
every security placeholder (their shared repo says nothing about the package), has them NULL.

Usage: python -u -m src.build_dataset
"""
import csv
import logging
import os
import re
import sqlite3
from collections import Counter

import config
from src import ghsa_features as gf
from src import github_activity as ga
from src.select_packages import read_csv

log = logging.getLogger("build_dataset")

TABLE = "pcom01_training"
EXPECTED_COLUMNS = 225
TEXT_COLUMNS = {"purl", "package_created_at", "published_at", "prediction_date"}
TIMESTAMP_COLUMNS = ["package_created_at", "published_at", "prediction_date"]
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def schema_columns(db):
    """Return [(name, required)] in schema order."""
    return [(row[1], bool(row[3])) for row in db.execute(f"PRAGMA table_info({TABLE})")]


def to_value(column, text):
    if text is None or text == "":
        return None
    return text if column in TEXT_COLUMNS else int(text)


def assemble(meta_rows, download_rows, ghsa_rows, github_rows, columns):
    """Join by (package, prediction date). Fails if a required source row is missing."""
    keyed = lambda rows: {(row["package_name"], row["prediction_date"]): row for row in rows}
    downloads, ghsa, github = keyed(download_rows), keyed(ghsa_rows), keyed(github_rows)
    assembled = []
    for meta in meta_rows:
        key = (meta["package_name"], meta["prediction_date"])
        merged = {**meta, **downloads[key], **ghsa[key]}
        source = github.get(key)
        if str(meta["is_security_placeholder"]) == "1":
            reason = "security placeholder, shared repo"
            source = None
        else:
            reason = "not collected in initial version"
        if source:
            merged.update({column: source[column] for column in ga.SERIES_COLUMNS})
        else:
            merged.update({column: None for column in ga.SERIES_COLUMNS})
            log.info("GitHub columns NULL for %s: %s", meta["package_name"], reason)
        assembled.append({column: to_value(column, merged[column]) for column in columns})
    return assembled


def validate(rows, columns):
    """Raise, listing the first problems, unless every essential check passes."""
    names = [name for name, _ in columns]
    problems = []
    if len(names) != EXPECTED_COLUMNS:
        problems.append(f"{len(names)} columns, expected {EXPECTED_COLUMNS}")
    for row in rows:
        purl = row["purl"]
        t0 = config.parse_timestamp(row["prediction_date"])
        for name, required in columns:
            value = row[name]
            if required and value is None:
                problems.append(f"{purl}: {name} is NULL")
            if isinstance(value, int) and value < 0:
                problems.append(f"{purl}: {name} is negative")
        for name in TIMESTAMP_COLUMNS:
            if not TIMESTAMP.fullmatch(row[name] or ""):
                problems.append(f"{purl}: {name} has format {row[name]!r}")
        if config.parse_timestamp(row["published_at"]) >= t0:
            problems.append(f"{purl}: published on or after t0")
        for month, (start, end) in config.feature_months(t0).items():
            observed = row[f"package_download_days_observed_M{month}"]
            if observed is not None and observed > (end - start).days:
                problems.append(f"{purl}: {observed} observed days in M{month}")
    if len({row["purl"] for row in rows}) != len(rows):
        problems.append("duplicate purl")
    if problems:
        raise ValueError(f"{len(problems)} validation problems: {problems[:20]}")


def write_outputs(rows, names, db_path, csv_path, schema_text):
    """Write to temporary files first, so a failure never leaves a half-built dataset."""
    temporary = db_path.with_name(db_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    db = sqlite3.connect(temporary)
    db.executescript(schema_text)
    marks = ", ".join("?" for _ in names)
    db.executemany(f"INSERT INTO {TABLE} VALUES ({marks})", [[row[name] for name in names] for row in rows])
    db.commit()
    stored = db.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    db.close()
    if stored != len(rows):
        raise ValueError(f"Stored {stored} rows, expected {len(rows)}")
    os.replace(temporary, db_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        writer.writerows([[row[name] for name in names] for row in rows])


def main():
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(config.LOG_DIR / "build_dataset.log", mode="w", encoding="utf-8")],
    )
    schema_text = config.SCHEMA_PATH.read_text(encoding="utf-8")
    memory = sqlite3.connect(":memory:")
    memory.executescript(schema_text)
    columns = schema_columns(memory)
    memory.close()
    names = [name for name, _ in columns]

    meta_rows = read_csv(config.NPM_META_CSV)
    rows = assemble(
        meta_rows, read_csv(config.NPM_DOWNLOADS_CSV), read_csv(config.GHSA_FEATURES_CSV),
        read_csv(config.GITHUB_ACTIVITY_CSV), names,
    )
    rows.sort(key=lambda row: row["purl"])
    validate(rows, columns)
    write_outputs(rows, names, config.OUTPUT_SQLITE, config.OUTPUT_CSV, schema_text)

    strata = {row["purl"]: row["stratum"] for row in read_csv(config.PACKAGES_INCLUDED_CSV)}
    print(f"Rows: {len(rows)}; columns: {len(names)}")
    print("Rows per stratum:", dict(Counter(strata[row["purl"]] for row in rows)))
    print("Positive rows per severity:", {
        severity: sum(row[f"target_ghsa_{severity}_count_90d"] > 0 for row in rows) for severity in gf.SEVERITIES
    })
    positive = [row for row in rows if any(row[column] > 0 for column in gf.TARGET_COLUMNS)]
    print(f"Positive rows, any severity: {len(positive)} ({len(positive) / len(rows):.1%})")
    print("Positive rows per stratum:", dict(Counter(strata[row["purl"]] for row in positive)))
    cells = len(rows) * len(ga.SERIES_COLUMNS)
    nulls = sum(row[column] is None for row in rows for column in ga.SERIES_COLUMNS)
    print(f"GitHub columns NULL share: {nulls / cells:.2%}")
    with_github = sum(row[ga.PR_COLUMNS[0]] is not None for row in rows)
    print(f"Rows with GitHub data: {with_github}")
    print(f"Sizes: sqlite {config.OUTPUT_SQLITE.stat().st_size / 1024**2:.1f} MB, "
          f"csv {config.OUTPUT_CSV.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
