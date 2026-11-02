from datetime import date, timedelta

import pytest

import config
from src import npm_downloads as nd

T0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")
OLD = config.parse_timestamp("2020-01-01T00:00:00Z")
WINDOW_DAYS = 365


def full_year(value=1):
    return [
        {"day": (date(2024, 8, 1) + timedelta(days=offset)).isoformat(), "downloads": value}
        for offset in range(WINDOW_DAYS)
    ]


def days_in_month(month):
    start, end = config.feature_months(T0)[month]
    return (end - start).days


def test_month_bucketing_at_utc_edges():
    entries = [
        {"day": "2025-07-31", "downloads": 1},  # last day of M0
        {"day": "2025-07-01", "downloads": 2},  # first day of M0
        {"day": "2025-06-30", "downloads": 4},  # last day of M1
        {"day": "2024-08-01", "downloads": 8},  # first day of M11
        {"day": "2024-07-31", "downloads": 100},  # before the window
        {"day": "2025-08-01", "downloads": 100},  # t0, excluded
    ]
    row, _ = nd.build_row("a", T0, OLD, entries)
    assert row["package_downloads_M0"] == 3
    assert row["package_download_days_observed_M0"] == 2
    assert row["package_downloads_M1"] == 4
    assert row["package_downloads_M11"] == 8
    assert nd.yearly_sum(row) == 15


def test_full_year_observes_every_day_of_every_month():
    row, stats = nd.build_row("a", T0, OLD, full_year())
    for month in nd.MONTHS:
        assert row[f"package_download_days_observed_M{month}"] == days_in_month(month)
        assert row[f"package_downloads_M{month}"] == days_in_month(month)
    assert nd.yearly_sum(row) == WINDOW_DAYS
    assert stats == {"days_before_creation": 0, "downloads_before_creation": 0}
    nd.validate([row])


def test_reported_zero_counts_as_observed():
    row, _ = nd.build_row("a", T0, OLD, full_year(0))
    assert row["package_downloads_M0"] == 0
    assert row["package_download_days_observed_M0"] == 31


def test_package_created_mid_month():
    created = config.parse_timestamp("2025-06-16T13:45:00.000Z")
    row, stats = nd.build_row("a", T0, created, full_year())
    assert row["package_download_days_observed_M1"] == 15  # 16 to 30 June, creation day included
    assert row["package_downloads_M1"] == 15
    assert row["package_download_days_observed_M0"] == 31
    assert row["package_download_days_observed_M2"] == 0
    assert row["package_downloads_M2"] is None
    assert row["package_download_days_observed_M11"] == 0
    assert row["package_downloads_M11"] is None
    assert stats["days_before_creation"] == WINDOW_DAYS - 46
    nd.validate([row])


def test_downloads_before_creation_are_left_out_and_reported():
    created = config.parse_timestamp("2025-07-31T00:00:00Z")
    row, stats = nd.build_row("a", T0, created, full_year(2))
    assert row["package_downloads_M0"] == 2
    assert stats["downloads_before_creation"] == 2 * (WINDOW_DAYS - 1)


def test_month_with_no_observed_days_is_null_not_zero():
    row, _ = nd.build_row("a", T0, OLD, [{"day": "2025-07-10", "downloads": 5}])
    assert row["package_downloads_M0"] == 5
    assert row["package_downloads_M1"] is None
    assert row["package_download_days_observed_M1"] == 0


@pytest.mark.parametrize("entries", [None, [], [{"day": "2020-01-01", "downloads": 5}]])
def test_no_data_gives_all_24_columns_null(entries):
    row, stats = nd.build_row("a", T0, OLD, entries)
    assert stats is None
    assert all(row[column] is None for column in nd.DOWNLOAD_COLUMNS + nd.DAYS_COLUMNS)
    assert len(nd.DOWNLOAD_COLUMNS + nd.DAYS_COLUMNS) == 24


def test_duplicate_days_count_once_and_bad_entries_are_ignored():
    row, _ = nd.build_row(
        "a", T0, OLD,
        [
            {"day": "2025-07-10", "downloads": 5},
            {"day": "2025-07-10", "downloads": 5},
            {"day": "bad", "downloads": 1},
            {"day": "2025-07-11", "downloads": None},
            {"day": "2025-07-12", "downloads": -1},
        ],
    )
    assert row["package_download_days_observed_M0"] == 1
    assert row["package_downloads_M0"] == 5


def test_month_direction_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "M0_IS_MOST_RECENT", False)
    row, _ = nd.build_row("a", T0, OLD, [{"day": "2025-07-10", "downloads": 5}])
    assert row["package_downloads_M11"] == 5
    assert row["package_downloads_M0"] is None


def test_observed_days_above_month_length_fail_validation():
    row, _ = nd.build_row("a", T0, OLD, full_year())
    row["package_download_days_observed_M1"] = 31  # June has 30 days
    with pytest.raises(ValueError, match="M1"):
        nd.validate([row])


def test_404_gives_null_row_and_logged_reason():
    meta = [{"package_name": "@s/gone", "prediction_date": "2025-08-01T00:00:00.000Z",
             "package_created_at": "2020-01-01T00:00:00.000Z"}]
    rows, nulls, _ = nd.collect(meta, lambda url: (404, {"error": "package not found"}))
    assert all(rows[0][column] is None for column in nd.DOWNLOAD_COLUMNS + nd.DAYS_COLUMNS)
    assert nulls == [("@s/gone", "2025-08-01T00:00:00.000Z", "HTTP 404: package not found")]


def test_fetch_series_uses_bulk_for_unscoped_and_single_for_scoped():
    urls = []
    daily = [{"day": "2025-07-01", "downloads": 1}]

    def fake_get(url):
        urls.append(url)
        if "," in url:
            return 200, {"a": {"downloads": daily}, "b": None}
        return 200, {"downloads": daily}

    result = nd.fetch_series(["a", "b", "@s/c"], "x:y", fake_get)
    assert result == {"a": (daily, None), "b": (None, "null in bulk response"), "@s/c": (daily, None)}
    assert sorted(urls) == [
        f"{config.NPM_DOWNLOADS_URL}/x:y/@s%2Fc",
        f"{config.NPM_DOWNLOADS_URL}/x:y/a,b",
    ]


def test_fetch_series_splits_large_bulk_and_handles_a_single_name():
    urls = []

    def fake_get(url):
        urls.append(url)
        names = url.rsplit("/", 1)[1].split(",")
        if len(names) == 1:
            return 200, {"downloads": []}
        return 200, {name: {"downloads": []} for name in names}

    names = [f"p{i}" for i in range(config.NPM_BULK_MAX_NAMES + 1)]
    assert set(nd.fetch_series(names, "x:y", fake_get)) == set(names)
    assert len(urls) == 2


def test_fetch_series_falls_back_to_singles_when_bulk_is_refused():
    def fake_get(url):
        return (400, {"error": "bad"}) if "," in url else (200, {"downloads": []})

    assert nd.fetch_series(["a", "b"], "x:y", fake_get) == {"a": ([], None), "b": ([], None)}


def test_cross_check_finds_mismatches_and_skips_group_a():
    rows = [nd.build_row(name, T0, OLD, full_year())[0] for name in ("same", "other", "in-a")]
    packages = [
        {"package_name": "same", "selection_downloads_feature_year": str(WINDOW_DAYS)},
        {"package_name": "other", "selection_downloads_feature_year": "9"},
        {"package_name": "in-a", "selection_downloads_feature_year": ""},
    ]
    assert nd.cross_check(rows, packages) == (2, [("other", WINDOW_DAYS, 9)])
