import sqlite3
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"
TABLE = "pcom01_training"
T0 = "2025-08-01T00:00:00.000Z"
NULLABLE_FAMILIES = (
    "package_downloads_M",
    "package_download_days_observed_M",
    "package_stable_release_count_M",
    "issues_opened_M",
    "issues_closed_M",
    "issues_reopened_M",
    "pull_requests_opened_M",
    "pull_requests_closed_unmerged_M",
    "pull_requests_merged_M",
    "pull_requests_reopened_M",
    "issues_open_at_month_end_M",
    "pull_requests_open_at_month_end_M",
)


@pytest.fixture
def db(tmp_path):
    connection = sqlite3.connect(tmp_path / "test.sqlite")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    yield connection
    connection.close()


def table_info(db):
    return db.execute(f"PRAGMA table_info({TABLE})").fetchall()


def valid_row(db):
    row = {name: 0 for _, name, column_type, *_ in table_info(db) if column_type == "INTEGER"}
    row.update(
        purl="pkg:npm/react@19.1.0",
        package_created_at="2011-10-26T17:46:21.942Z",
        published_at="2025-03-28T19:59:42.053Z",
        prediction_date=T0,
    )
    return row


def insert(db, row):
    names = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    db.execute(f"INSERT INTO {TABLE} ({names}) VALUES ({marks})", list(row.values()))


def test_column_count(db):
    assert len(table_info(db)) == 225


def test_only_optional_columns_are_nullable(db):
    nullable = {name for _, name, _, notnull, *_ in table_info(db) if not notnull}
    expected = {"unpacked_size", "file_count"} | {
        f"{family}{i}" for family in NULLABLE_FAMILIES for i in range(12)
    }
    assert nullable == expected


def test_valid_row_is_accepted(db):
    insert(db, valid_row(db))
    assert db.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0] == 1


def test_optional_columns_accept_null(db):
    row = valid_row(db)
    row.update(unpacked_size=None, issues_opened_M0=None)
    insert(db, row)


def test_every_count_rejects_negative(db):
    row = valid_row(db)
    counts = [name for name, value in row.items() if isinstance(value, int)]
    assert len(counts) == 221
    for name in counts:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            insert(db, {**row, name: -1})


def test_every_required_column_rejects_null(db):
    row = valid_row(db)
    required = [name for _, name, _, notnull, *_ in table_info(db) if notnull]
    assert len(required) == 225 - 2 - 144
    for name in required:
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL constraint failed"):
            insert(db, {**row, name: None})


def test_duplicate_purl_and_prediction_date_rejected(db):
    row = valid_row(db)
    insert(db, row)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        insert(db, row)


def test_same_purl_at_another_prediction_date_is_accepted(db):
    row = valid_row(db)
    insert(db, row)
    insert(db, {**row, "prediction_date": "2025-11-01T00:00:00.000Z"})


@pytest.mark.parametrize("purl", ["react@19.1.0", "pkg:pypi/react@19.1.0", "pkg:npm/" + "a" * 2041])
def test_invalid_purl_rejected(db, purl):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        insert(db, {**valid_row(db), "purl": purl})


def test_days_observed_above_31_rejected(db):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        insert(db, {**valid_row(db), "package_download_days_observed_M0": 32})
