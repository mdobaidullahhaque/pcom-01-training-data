"""Compute monthly download features from npm's daily download data.

Usage: python -u -m src.npm_downloads [--packages data/packages_test.csv]
"""
import argparse
import csv
import logging
from datetime import date
from pathlib import Path
from urllib.parse import quote

import config
from src.http_client import get_json
from src.select_packages import feature_year, read_csv

log = logging.getLogger("npm_downloads")

MONTHS = range(config.FEATURE_MONTHS)
DOWNLOAD_COLUMNS = [f"package_downloads_M{month}" for month in MONTHS]
DAYS_COLUMNS = [f"package_download_days_observed_M{month}" for month in MONTHS]
COLUMNS = ["package_name", "prediction_date", *DOWNLOAD_COLUMNS, *DAYS_COLUMNS]


def fetch_one(name, window, get=get_json):
    """Return (daily entries, None), or (None, reason) when npm has no data."""
    status, body = get(f"{config.NPM_DOWNLOADS_URL}/{window}/{quote(name, safe='@')}")
    if status != 200:
        return None, f"HTTP {status}: {(body or {}).get('error')}"
    return (body or {}).get("downloads"), None


def fetch_series(names, window, get=get_json):
    """Return {name: (daily entries, reason)}. Scoped names cannot use npm's bulk endpoint."""
    unscoped = [name for name in names if not name.startswith("@")]
    result = {name: fetch_one(name, window, get) for name in names if name.startswith("@")}
    for start in range(0, len(unscoped), config.NPM_BULK_MAX_NAMES):
        chunk = unscoped[start:start + config.NPM_BULK_MAX_NAMES]
        if len(chunk) == 1:
            result[chunk[0]] = fetch_one(chunk[0], window, get)
            continue
        joined = ",".join(quote(name, safe="") for name in chunk)
        status, body = get(f"{config.NPM_DOWNLOADS_URL}/{window}/{joined}")
        if status != 200:
            log.warning("Bulk lookup returned %s; asking for %d names singly", status, len(chunk))
            result.update({name: fetch_one(name, window, get) for name in chunk})
            continue
        for name in chunk:
            entry = (body or {}).get(name)
            result[name] = (entry.get("downloads"), None) if entry else (None, "null in bulk response")
    return result


def daily_values(entries):
    """Return {date: downloads}, keeping a reported 0 and dropping unusable entries."""
    values = {}
    for entry in entries or []:
        try:
            day = date.fromisoformat(entry["day"])
            downloads = entry["downloads"]
        except (KeyError, TypeError, ValueError):
            log.warning("Unusable daily entry ignored: %r", entry)
            continue
        if isinstance(downloads, bool) or not isinstance(downloads, int) or downloads < 0:
            log.warning("Unusable daily value ignored: %r", entry)
            continue
        values.setdefault(day, downloads)
    return values


def null_row(name, t0):
    return {"package_name": name, "prediction_date": config.format_timestamp(t0),
            **{column: None for column in DOWNLOAD_COLUMNS + DAYS_COLUMNS}}


def build_row(name, t0, created_at, entries):
    """Return (row, stats), or (row of 24 NULLs, None) when no day falls in the feature window.

    npm reports 0 for days before a package existed, which is not a real observation: the
    package could not have been downloaded. So days before the UTC day of package_created_at
    are left out of both the observed-day count and the sum. The creation day itself counts.
    """
    windows = config.feature_months(t0)
    first_day = min(start.date() for start, _ in windows.values())
    end_day = max(end.date() for _, end in windows.values())
    values = {day: value for day, value in daily_values(entries).items() if first_day <= day < end_day}
    row = null_row(name, t0)
    if not values:
        return row, None

    created_day = created_at.date()
    stats = {
        "days_before_creation": sum(day < created_day for day in values),
        "downloads_before_creation": sum(value for day, value in values.items() if day < created_day),
    }
    for month, (start, end) in windows.items():
        observed = [
            value for day, value in values.items()
            if start.date() <= day < end.date() and day >= created_day
        ]
        row[f"package_download_days_observed_M{month}"] = len(observed)
        row[f"package_downloads_M{month}"] = sum(observed) if observed else None
    return row, stats


def collect(meta_rows, get=get_json):
    """Build one row per npm_meta row, grouped by prediction date."""
    rows, nulls, removed = [], [], {}
    for prediction_date in sorted({row["prediction_date"] for row in meta_rows}):
        t0 = config.parse_timestamp(prediction_date)
        members = [row for row in meta_rows if row["prediction_date"] == prediction_date]
        window = feature_year(t0)
        print(f"{prediction_date}: {len(members)} packages, window {window}")
        series = fetch_series([row["package_name"] for row in members], window, get)
        for member in members:
            name = member["package_name"]
            entries, reason = series[name]
            created_at = config.parse_timestamp(member["package_created_at"])
            row, stats = build_row(name, t0, created_at, entries)
            if stats is None:
                reason = reason or "no daily data in the feature window"
                nulls.append((name, prediction_date, reason))
                log.warning("All download columns NULL for %s: %s", name, reason)
            elif stats["days_before_creation"]:
                removed[name] = stats
                log.info("%s: left out %d days before creation (%d downloads)", name,
                         stats["days_before_creation"], stats["downloads_before_creation"])
            rows.append(row)
    return rows, nulls, removed


def yearly_sum(row):
    values = [row[column] for column in DOWNLOAD_COLUMNS]
    return None if all(value is None for value in values) else sum(value or 0 for value in values)


def cross_check(rows, package_rows):
    """Compare each 12-month sum with the yearly total used in selection (strata B and C)."""
    selection = {
        row["package_name"]: int(row["selection_downloads_feature_year"])
        for row in package_rows if row["selection_downloads_feature_year"]
    }
    checked = [row for row in rows if row["package_name"] in selection]
    mismatches = [
        (row["package_name"], yearly_sum(row), selection[row["package_name"]])
        for row in checked if yearly_sum(row) != selection[row["package_name"]]
    ]
    return len(checked), mismatches


def validate(rows):
    """Fail loudly if a row reports more observed days than its month has."""
    for row in rows:
        t0 = config.parse_timestamp(row["prediction_date"])
        for month, (start, end) in config.feature_months(t0).items():
            observed = row[f"package_download_days_observed_M{month}"]
            if observed is not None and not 0 <= observed <= (end - start).days:
                raise ValueError(f"{row['package_name']} M{month}: {observed} observed days")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(rows, nulls, removed, checked, mismatches):
    print(f"\n=== Report: {len(rows)} rows ===")
    print("All-NULL packages:", nulls)
    print("NULL cells:", {
        "downloads": sum(row[column] is None for row in rows for column in DOWNLOAD_COLUMNS),
        "days_observed": sum(row[column] is None for row in rows for column in DAYS_COLUMNS),
    })
    print("Packages with days left out before creation:", removed)
    print(f"Cross-check: {checked} checked, {len(mismatches)} mismatches")
    for mismatch in mismatches:
        print(f"  {mismatch}")
    for row in rows:
        print(f"  {row['package_name']}: total {yearly_sum(row)}, "
              f"days {[row[column] for column in DAYS_COLUMNS]}")
    print("Example rows:")
    for row in rows[:1] + [row for row in rows if row["package_name"] in removed][:1] + rows[-1:]:
        print(f"  {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=config.PACKAGES_TEST_CSV)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "npm_downloads.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    names = {row["package_name"] for row in read_csv(args.packages)}
    meta_rows = [row for row in read_csv(config.NPM_META_CSV) if row["package_name"] in names]
    missing = names - {row["package_name"] for row in meta_rows}
    for name in sorted(missing):
        log.warning("No npm_meta row, so no download row: %s", name)
    rows, nulls, removed = collect(meta_rows)
    validate(rows)
    write_rows(config.NPM_DOWNLOADS_CSV, rows)
    checked, mismatches = cross_check(rows, read_csv(config.PACKAGES_CSV))
    report(rows, nulls, removed, checked, mismatches)


if __name__ == "__main__":
    main()
