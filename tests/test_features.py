import csv
import math
import random
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

import config
from src.model import features as ft

T0 = datetime(2025, 8, 1, tzinfo=timezone.utc)


def read_csv_rows():
    with open(config.OUTPUT_CSV, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_sqlite_rows():
    db = sqlite3.connect(config.OUTPUT_SQLITE)
    db.row_factory = sqlite3.Row
    rows = [dict(row) for row in db.execute("SELECT * FROM pcom01_training ORDER BY purl")]
    db.close()
    return rows


def same(first, second):
    return np.array_equal(first, second, equal_nan=True)


def test_csv_and_sqlite_give_the_same_features():
    from_csv = sorted(read_csv_rows(), key=lambda row: row["purl"])
    from_sqlite = read_sqlite_rows()
    assert [row["purl"] for row in from_csv] == [row["purl"] for row in from_sqlite]
    references, _ = ft.resolve_reference_dates(from_csv)
    assert same(ft.build_features(from_csv, references)[0], ft.build_features(from_sqlite, references)[0])


def test_names_match_columns_and_reference_features_can_be_switched_off():
    matrix, names = ft.build_features([{}], [T0])
    assert names == ft.FEATURE_NAMES and matrix.shape == (1, len(names))
    matrix, names = ft.build_features([{}], [T0], use_reference_features=False)
    assert not set(ft.REFERENCE_FEATURES) & set(names)
    assert matrix.shape == (1, len(ft.FEATURE_NAMES) - 2)


@pytest.mark.parametrize("row", [{}, {"purl": "pkg:npm/a@1.0.0"}, {"unpacked_size": "big", "published_at": "soon"}, None, "not a dict"])
def test_missing_or_unreadable_input_gives_nan_not_an_error(row):
    matrix, _ = ft.build_features([row], [T0])
    assert np.isnan(matrix).all()


@pytest.mark.parametrize("value", ["", None, "abc", "-1", "inf", "nan", True])
def test_unusable_numbers_become_nan(value):
    assert math.isnan(ft.number(value))


def test_targets_and_other_excluded_columns_cannot_influence_features():
    rows = read_csv_rows()[:300]
    references, _ = ft.resolve_reference_dates(rows)
    before, names = ft.build_features(rows, references)
    rng = random.Random(0)
    excluded = [c for c in rows[0] if c.startswith(("target_", "issues_", "pull_requests_")) or "unknown" in c or c == "prediction_date"]
    assert len(excluded) == 5 + 108 + 13 + 1
    changed = [{**row, **{column: str(rng.randint(0, 50)) for column in excluded}} for row in rows]
    assert same(before, ft.build_features(changed, references)[0])
    assert not any("target" in name or "unknown" in name for name in names)
    assert not (ft.INPUT_COLUMNS & set(excluded))


def test_durations_are_never_negative():
    row = {"package_created_at": "2025-07-01T00:00:00.000Z", "published_at": "2024-01-01T00:00:00.000Z"}
    matrix, names = ft.build_features([row], [datetime(2020, 1, 1, tzinfo=timezone.utc)])
    for name in ("days_created_to_published", *ft.REFERENCE_FEATURES):
        assert matrix[0, names.index(name)] == 0
    rows = read_csv_rows()
    matrix, names = ft.build_features(rows, ft.resolve_reference_dates(rows)[0])
    for name in ("days_created_to_published", *ft.REFERENCE_FEATURES):
        column = matrix[:, names.index(name)]
        assert (column[~np.isnan(column)] >= 0).all()


def full_row(**changes):
    row = {
        "unpacked_size": "1023", "file_count": "9", "dependency_count": "2",
        "package_created_at": "2025-01-01T00:00:00.000Z", "published_at": "2025-01-11T00:00:00.000Z",
        "ghsa_low_count": "1", "ghsa_moderate_count": "2", "ghsa_high_count": "0", "ghsa_critical_count": "1",
    }
    for month in ft.MONTHS:
        row[f"package_downloads_M{month}"] = "30"
        row[f"package_download_days_observed_M{month}"] = "30"
        row[f"package_stable_release_count_M{month}"] = "0"
        for severity in ft.SEVERITIES:
            row[f"ghsa_{severity}_count_M{month}"] = "0"
    row.update(changes)
    return row


def features_of(row, reference=T0):
    matrix, names = ft.build_features([row], [reference])
    return dict(zip(names, matrix[0]))


def test_aggregates_on_a_known_row():
    values = features_of(full_row(package_downloads_M0="120", package_stable_release_count_M1="2",
                                  package_stable_release_count_M7="1", ghsa_high_count_M4="1"))
    assert values["log_unpacked_size"] == pytest.approx(math.log(1024))
    assert values["dependency_count"] == 2 and math.isnan(values["peer_dependency_count"])
    assert values["days_created_to_published"] == 10
    assert values["package_age_days"] == 212
    assert values["log_downloads_12m"] == pytest.approx(math.log1p(450))
    assert values["log_downloads_3m"] == pytest.approx(math.log1p(180))
    assert values["downloads_trend"] == pytest.approx((180 / 3 + 1) / (270 / 9 + 1))
    assert values["log_downloads_max_month"] == pytest.approx(math.log1p(120))
    assert values["download_days_observed"] == 360
    assert (values["releases_12m"], values["releases_3m"], values["release_months_active"]) == (3, 2, 2)
    assert (values["ghsa_total"], values["ghsa_high_critical"], values["has_any_ghsa"]) == (4, 1, 1)
    assert (values["ghsa_12m"], values["ghsa_3m"], values["months_since_last_ghsa"]) == (1, 0, 4)


def test_no_ghsa_in_window_gives_nan_months_since():
    values = features_of(full_row())
    assert values["ghsa_12m"] == 0 and math.isnan(values["months_since_last_ghsa"])


def test_recency_follows_the_month_direction(monkeypatch):
    row = full_row(package_downloads_M0="300", ghsa_low_count_M0="1")
    assert features_of(row)["months_since_last_ghsa"] == 0
    monkeypatch.setattr(config, "M0_IS_MOST_RECENT", False)
    values = features_of(row)
    assert values["months_since_last_ghsa"] == 11
    assert values["log_downloads_3m"] == pytest.approx(math.log1p(90))
    assert values["log_downloads_12m"] == pytest.approx(math.log1p(630))


def test_reference_date_resolution_order():
    rows = [{"published_at": "2025-07-31T23:00:00Z", "prediction_date": "2025-08-01T00:00:00.000Z"},
            {"published_at": "2024-02-10T00:00:00Z"}]
    dates, source = ft.resolve_reference_dates(rows, "2026-01-01")
    assert dates == [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2 and "--prediction-date" in source
    dates, source = ft.resolve_reference_dates(rows)
    assert dates == [T0, T0] and "column" in source and "1 rows" in source
    dates, source = ft.resolve_reference_dates([{"published_at": "2025-12-05T10:00:00Z"}])
    assert dates == [datetime(2026, 1, 1, tzinfo=timezone.utc)] and "inferred" in source
    dates, source = ft.resolve_reference_dates([{}])
    assert dates == [None] and "unknown" in source
    dates, _ = ft.resolve_reference_dates(rows, "not a date")
    assert dates == [T0, T0]


def test_training_data_reference_date_is_t0():
    dates, source = ft.resolve_reference_dates(read_csv_rows())
    assert set(dates) == {T0} and source == "prediction_date column"
