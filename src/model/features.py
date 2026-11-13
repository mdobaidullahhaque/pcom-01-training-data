"""Model features from PCOM-01 rows, shared by training and prediction.

Uses only numpy and the standard library, so predict.py stays fast. Rows are dicts of column
name to value: strings from a CSV, or numbers and None from SQLite. A missing column or an
unreadable value becomes NaN; nothing here raises on bad input.

Only the columns listed in INPUT_COLUMNS are ever read, so GitHub, unknown-severity and
target columns cannot influence a feature.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np

import config

NAN = float("nan")
MONTHS = range(config.FEATURE_MONTHS)
SEVERITIES = ["low", "moderate", "high", "critical"]  # unknown is always 0 in the training data
DEPENDENCY_COLUMNS = [
    "dependency_count", "optional_dependency_count", "peer_dependency_count",
    "dev_dependency_count", "bundled_dependency_count",
]
# Durations measured from the prediction date; Task 3 decides whether to keep them.
REFERENCE_FEATURES = ["package_age_days", "days_since_version_published"]
FEATURE_NAMES = [
    "log_unpacked_size", "log_file_count", *DEPENDENCY_COLUMNS,
    "days_created_to_published", *REFERENCE_FEATURES,
    "log_downloads_12m", "log_downloads_3m", "downloads_trend", "log_downloads_max_month", "download_days_observed",
    "releases_12m", "releases_3m", "release_months_active",
    *[f"ghsa_{severity}_total" for severity in SEVERITIES],
    "ghsa_total", "ghsa_high_critical", "has_any_ghsa", "ghsa_12m", "ghsa_3m", "months_since_last_ghsa",
]
INPUT_COLUMNS = frozenset(
    ["unpacked_size", "file_count", *DEPENDENCY_COLUMNS, "package_created_at", "published_at"]
    + [f"package_downloads_M{m}" for m in MONTHS]
    + [f"package_download_days_observed_M{m}" for m in MONTHS]
    + [f"package_stable_release_count_M{m}" for m in MONTHS]
    + [f"ghsa_{severity}_count" for severity in SEVERITIES]
    + [f"ghsa_{severity}_count_M{m}" for severity in SEVERITIES for m in MONTHS]
)


def recent_first(column_prefix):
    """Monthly column names ordered from the most recent month, whatever the M0 direction."""
    months = list(MONTHS) if config.M0_IS_MOST_RECENT else list(reversed(MONTHS))
    return [f"{column_prefix}_M{month}" for month in months]


def number(value):
    """Return a finite, non-negative float, or NaN for anything missing or unusable."""
    if value is None or isinstance(value, bool):
        return NAN
    try:
        result = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, AttributeError):
        return NAN
    return result if math.isfinite(result) and result >= 0 else NAN


def timestamp(value):
    """Return an aware UTC datetime, or None. Accepts RFC 3339, with or without Z, or a date."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00").replace("z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def days_between(start, end):
    """Days from start to end, never negative; NaN if either is unknown."""
    if start is None or end is None:
        return NAN
    return max(0.0, (end - start).total_seconds() / 86400)


def log1p(value):
    return math.log1p(value) if not math.isnan(value) else NAN


def total(values):
    """Sum of known values; NaN when none are known."""
    known = [value for value in values if not math.isnan(value)]
    return float(sum(known)) if known else NAN


def first_month_after(moment):
    year, month = (moment.year + 1, 1) if moment.month == 12 else (moment.year, moment.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc)


def resolve_reference_dates(rows, override=None):
    """Return (one reference datetime or None per row, a description of where it came from).

    Order: the override (--prediction-date), then each row's prediction_date, then the start of
    the month after the latest published_at in the file. The last is a guess: t0 is assumed to
    fall on the first of a month, as it did in training.
    """
    if override is not None:
        moment = timestamp(override) if isinstance(override, str) else override
        if moment is not None:
            return [moment] * len(rows), f"--prediction-date {config.format_timestamp(moment)}"
    column = [timestamp(row.get("prediction_date")) for row in rows]
    published = [moment for moment in (timestamp(row.get("published_at")) for row in rows) if moment]
    inferred = first_month_after(max(published)) if published else None
    if any(moment is not None for moment in column):
        filled = [moment or inferred for moment in column]
        missing = sum(moment is None for moment in column)
        note = f" ({missing} rows without it use the inferred date)" if missing else ""
        return filled, f"prediction_date column{note}"
    if inferred is not None:
        return [inferred] * len(rows), f"inferred from published_at: {config.format_timestamp(inferred)}"
    return [None] * len(rows), "unknown: no prediction date and no readable published_at"


def row_features(row, reference):
    """Return the feature values for one row, in FEATURE_NAMES order."""
    get = lambda column: number(row.get(column)) if column in INPUT_COLUMNS else NAN
    created, published = timestamp(row.get("package_created_at")), timestamp(row.get("published_at"))

    downloads = [get(column) for column in recent_first("package_downloads")]
    last3, before = downloads[:3], downloads[3:]
    known_last3, known_before = total(last3), total(before)
    if math.isnan(known_last3) or math.isnan(known_before):
        trend = NAN
    else:
        # Ratio of monthly averages, smoothed by 1 so that zero months cannot divide by zero.
        trend = (known_last3 / 3 + 1) / (known_before / len(before) + 1)
    known_downloads = [value for value in downloads if not math.isnan(value)]

    releases = [get(column) for column in recent_first("package_stable_release_count")]
    known_releases = [value for value in releases if not math.isnan(value)]

    totals = {severity: get(f"ghsa_{severity}_count") for severity in SEVERITIES}
    monthly = [
        total([get(f"ghsa_{severity}_count_M{column.rsplit('_M', 1)[1]}") for severity in SEVERITIES])
        for column in recent_first("ghsa_low_count")
    ]
    all_totals = total(totals.values()) if all(not math.isnan(v) for v in totals.values()) else NAN
    recent_months = [index for index, value in enumerate(monthly) if not math.isnan(value) and value > 0]
    if recent_months:
        months_since = float(recent_months[0])
    else:
        months_since = NAN

    values = {
        "log_unpacked_size": log1p(get("unpacked_size")),
        "log_file_count": log1p(get("file_count")),
        **{column: get(column) for column in DEPENDENCY_COLUMNS},
        "days_created_to_published": days_between(created, published),
        "package_age_days": days_between(created, reference),
        "days_since_version_published": days_between(published, reference),
        "log_downloads_12m": log1p(total(downloads)),
        "log_downloads_3m": log1p(known_last3),
        "downloads_trend": trend,
        "log_downloads_max_month": log1p(max(known_downloads)) if known_downloads else NAN,
        "download_days_observed": total([get(column) for column in recent_first("package_download_days_observed")]),
        "releases_12m": total(releases),
        "releases_3m": total(releases[:3]),
        "release_months_active": float(sum(value > 0 for value in known_releases)) if known_releases else NAN,
        **{f"ghsa_{severity}_total": totals[severity] for severity in SEVERITIES},
        "ghsa_total": all_totals,
        "ghsa_high_critical": total([totals["high"], totals["critical"]]) if not (math.isnan(totals["high"]) or math.isnan(totals["critical"])) else NAN,
        "has_any_ghsa": float(all_totals > 0) if not math.isnan(all_totals) else NAN,
        "ghsa_12m": total(monthly),
        "ghsa_3m": total(monthly[:3]),
        "months_since_last_ghsa": months_since,
    }
    return [values[name] for name in FEATURE_NAMES]


def build_features(rows, reference_dates=None, use_reference_features=True):
    """Return (float matrix with one row per input row, ordered feature names)."""
    rows = list(rows)
    reference_dates = reference_dates if reference_dates is not None else [None] * len(rows)
    names = [name for name in FEATURE_NAMES if use_reference_features or name not in REFERENCE_FEATURES]
    keep = [FEATURE_NAMES.index(name) for name in names]
    matrix = np.full((len(rows), len(names)), np.nan, dtype=float)
    for index, (row, reference) in enumerate(zip(rows, reference_dates)):
        try:
            values = row_features(row if isinstance(row, dict) else {}, reference)
        except Exception:  # one malformed row must never stop a prediction run
            continue
        matrix[index] = [values[position] for position in keep]
    return matrix, names
