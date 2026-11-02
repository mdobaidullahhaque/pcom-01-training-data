import sqlite3

import pytest

import config
from src import build_dataset as bd
from src import github_activity as ga
from src import run_initial as ri

T0 = config.parse_timestamp(config.PREDICTION_DATES[0])
STAMP = "2025-01-01T00:00:00.000Z"


@pytest.fixture
def columns():
    db = sqlite3.connect(":memory:")
    db.executescript(config.SCHEMA_PATH.read_text(encoding="utf-8"))
    result = bd.schema_columns(db)
    db.close()
    return result


def source_rows(columns, placeholder="0", github=False):
    names = [name for name, _ in columns]
    key = {"package_name": "a", "prediction_date": config.PREDICTION_DATES[0]}
    base = {name: "1" for name in names if name not in bd.TEXT_COLUMNS}
    meta = {**key, **base, "purl": "pkg:npm/a@1.0.0", "package_created_at": STAMP, "published_at": STAMP,
            "is_security_placeholder": placeholder}
    github_rows = [{**key, **{column: "2" for column in ga.SERIES_COLUMNS}}] if github else []
    return [meta], [dict(key)], [dict(key)], github_rows, names


def test_schema_has_225_columns(columns):
    assert len(columns) == bd.EXPECTED_COLUMNS


def test_github_values_used_when_cached_otherwise_null(columns):
    assert bd.assemble(*source_rows(columns, github=True))[0]["issues_opened_M0"] == 2
    assert bd.assemble(*source_rows(columns))[0]["issues_opened_M0"] is None


def test_placeholder_github_columns_are_null(columns):
    assert bd.assemble(*source_rows(columns, placeholder="1", github=True))[0]["issues_opened_M0"] is None


def valid_row(columns):
    return bd.assemble(*source_rows(columns))[0]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"dependency_count": None}, "is NULL"),
        ({"dependency_count": -1}, "negative"),
        ({"published_at": "2025-01-01T00:00:00Z"}, "format"),
        ({"published_at": config.PREDICTION_DATES[0]}, "on or after t0"),
        ({"package_download_days_observed_M1": 31}, "observed days in M1"),
    ],
)
def test_validation_rejects_bad_rows(columns, change, message):
    bd.validate([valid_row(columns)], columns)
    with pytest.raises(ValueError, match=message):
        bd.validate([{**valid_row(columns), **change}], columns)


def test_outputs_go_through_the_schema(columns, tmp_path):
    names = [name for name, _ in columns]
    db_path, csv_path = tmp_path / "d.sqlite", tmp_path / "d.csv"
    bd.write_outputs([valid_row(columns)], names, db_path, csv_path, config.SCHEMA_PATH.read_text(encoding="utf-8"))
    db = sqlite3.connect(db_path)
    assert db.execute(f"SELECT COUNT(*) FROM {bd.TABLE}").fetchone()[0] == 1
    db.close()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].split(",") == names


def test_shuffle_is_reproducible():
    rows = [{"package_name": str(i)} for i in range(50)]
    assert ri.shuffled_packages(rows, 42) == ri.shuffled_packages(rows, 42)
    assert ri.shuffled_packages(rows, 42) != rows


def test_budget_stops_starting_packages_and_failures_are_skipped():
    times = iter(range(100))
    packages = [{"package_name": name} for name in ("gone", "bad", "unreached")]
    answers = {"gone": (404, {}), "bad": (200, {"time": {"unpublished": {}}})}
    rows, failures, reached = ri.collect_meta(
        packages, T0, deadline=2, fetch=lambda name: answers[name], clock=lambda: next(times), started=0
    )
    assert rows == []
    assert reached == 2
    assert [name for name, _ in failures] == ["gone", "bad"]
