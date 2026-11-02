import json
import logging
import sqlite3

import pytest

import config
from src import ghsa_features as gf
from src.advisories import CREATE_TABLE, parse_advisory

T0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")
AT_T0 = "2025-08-01T00:00:00.000Z"
AT_END = "2025-10-30T00:00:00.000Z"
JUST_BEFORE_END = "2025-10-29T23:59:59.999Z"
EARLY = "2025-07-15T00:00:00.000Z"
OLD = "2019-03-01T00:00:00.000Z"


def advisory(ghsa_id, published, severity="HIGH", name="pkg", repeat=1, withdrawn=None):
    body = {
        "id": ghsa_id,
        "published": published,
        "modified": "2026-01-01T00:00:00Z",
        "affected": [{"package": {"ecosystem": "npm", "name": name}, "ranges": [{"n": i}]} for i in range(repeat)],
        "database_specific": {"severity": severity},
    }
    if withdrawn:
        body["withdrawn"] = withdrawn
    return body


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "advisories.sqlite"


def store(db_path, tmp_path, snapshot, advisories):
    """Send fake OSV files through the real parser, as Task 2's table was built."""
    folder = tmp_path / snapshot
    folder.mkdir()
    with sqlite3.connect(db_path) as db:
        db.execute(CREATE_TABLE)
        for body in advisories:
            path = folder / f"{body['id']}.json"
            path.write_text(json.dumps(body), encoding="utf-8")
            for row in parse_advisory(path, folder):
                db.execute(
                    "INSERT INTO advisories VALUES (:snapshot, :ghsa_id, :package_name, :published,"
                    " :modified, :withdrawn, :severity, :is_cwe506, :file_path)",
                    {**row, "snapshot": snapshot},
                )
    db.close()


def compute(db_path, tmp_path, at_t0, at_end, names=("pkg",)):
    store(db_path, tmp_path, "t0", at_t0)
    store(db_path, tmp_path, "target_end", at_end)
    features = gf.count_features(gf.load_active(db_path, "t0"), T0)
    targets = gf.count_targets(gf.load_active(db_path, "target_end"), T0)
    rows = gf.build_rows(list(names), features, targets, T0)
    gf.validate(rows)
    return rows


def test_published_exactly_at_t0_is_target_not_feature(db_path, tmp_path):
    item = advisory("GHSA-a", AT_T0)
    row = compute(db_path, tmp_path, [advisory("GHSA-x", OLD, name="other")], [item])[0]
    assert row["target_ghsa_high_count_90d"] == 1
    assert row["ghsa_high_count"] == 0
    item_in_both = compute_again(tmp_path, [item, advisory("GHSA-x", OLD, name="other")], [item])
    assert item_in_both["ghsa_high_count"] == 0


def compute_again(tmp_path, at_t0, at_end):
    folder = tmp_path / "second"
    folder.mkdir()
    return compute(folder / "a.sqlite", folder, at_t0, at_end)[0]


def test_published_exactly_at_window_end_is_neither(db_path, tmp_path):
    row = compute(db_path, tmp_path, [advisory("GHSA-x", OLD, name="other")], [advisory("GHSA-a", AT_END)])[0]
    assert row["target_ghsa_high_count_90d"] == 0
    assert row["ghsa_high_count"] == 0


def test_last_moment_of_window_is_a_target(db_path, tmp_path):
    row = compute(db_path, tmp_path, [advisory("GHSA-x", OLD, name="other")], [advisory("GHSA-a", JUST_BEFORE_END)])[0]
    assert row["target_ghsa_high_count_90d"] == 1


def test_withdrawn_is_not_counted_in_that_snapshot(db_path, tmp_path):
    withdrawn = "2025-07-20T00:00:00Z"
    at_t0 = [advisory("GHSA-w", EARLY, withdrawn=withdrawn), advisory("GHSA-x", OLD, name="other")]
    at_end = [advisory("GHSA-t", AT_T0, withdrawn="2025-09-01T00:00:00Z"), advisory("GHSA-y", OLD, name="other")]
    row = compute(db_path, tmp_path, at_t0, at_end)[0]
    assert row["ghsa_high_count"] == 0
    assert row["ghsa_high_count_M0"] == 0
    assert row["target_ghsa_high_count_90d"] == 0


def test_withdrawn_after_t0_still_counts_as_feature(db_path, tmp_path):
    at_end = [advisory("GHSA-a", EARLY, withdrawn="2025-09-01T00:00:00Z"), advisory("GHSA-y", OLD, name="other")]
    row = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY)], at_end)[0]
    assert row["ghsa_high_count"] == 1
    assert gf.withdrawn_after_t0(db_path) == ["GHSA-a"]


@pytest.mark.parametrize(
    "raw, level",
    [("LOW", "low"), ("MODERATE", "moderate"), ("HIGH", "high"), ("CRITICAL", "critical"), ("Critical", "critical"), (None, "unknown"), ("MEDIUM", "unknown"), ("", "unknown")],
)
def test_severity_levels(db_path, tmp_path, raw, level):
    row = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY, raw)], [advisory("GHSA-b", AT_T0, raw)])[0]
    assert row[f"ghsa_{level}_count"] == 1
    assert row[f"target_ghsa_{level}_count_90d"] == 1


def test_unrecognised_severity_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="ghsa_features"):
        assert gf.normalize_severity("MEDIUM", "GHSA-a") == "unknown"
    assert "MEDIUM" in caplog.text and "GHSA-a" in caplog.text


def test_duplicate_affected_entries_count_once(db_path, tmp_path):
    row = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY, repeat=3)], [advisory("GHSA-b", AT_T0, repeat=3)])[0]
    assert row["ghsa_high_count"] == 1
    assert row["ghsa_high_count_M0"] == 1
    assert row["target_ghsa_high_count_90d"] == 1


def test_only_in_target_end_with_early_publish_is_neither_and_reported(db_path, tmp_path):
    at_t0 = [advisory("GHSA-x", OLD, name="other")]
    at_end = [advisory("GHSA-x", OLD, name="other"), advisory("GHSA-late", EARLY)]
    row = compute(db_path, tmp_path, at_t0, at_end)[0]
    assert row["ghsa_high_count"] == 0
    assert row["target_ghsa_high_count_90d"] == 0
    late = gf.reviewed_after_t0(db_path, T0, {"GHSA-x"})
    assert late["npm_ids_not_in_t0"] == ["GHSA-late"]
    assert late["file_absent_from_t0_snapshot"] == ["GHSA-late"]
    assert late["packages"] == ["pkg"]


def test_advisory_gaining_npm_later_is_separated(db_path, tmp_path):
    compute(db_path, tmp_path, [advisory("GHSA-x", OLD, name="other")], [advisory("GHSA-x", OLD, name="other"), advisory("GHSA-b", EARLY)])
    late = gf.reviewed_after_t0(db_path, T0, {"GHSA-x", "GHSA-b"})
    assert late["file_in_t0_without_npm_entry"] == ["GHSA-b"]
    assert late["file_absent_from_t0_snapshot"] == []


def test_each_count_uses_its_own_snapshots_severity(db_path, tmp_path):
    row = compute(
        db_path, tmp_path,
        [advisory("GHSA-a", EARLY, "LOW")],
        [advisory("GHSA-a", EARLY, "CRITICAL"), advisory("GHSA-b", AT_T0, "HIGH")],
    )[0]
    assert row["ghsa_low_count"] == 1
    assert row["ghsa_critical_count"] == 0
    assert row["target_ghsa_high_count_90d"] == 1
    assert gf.severity_changes(db_path) == {"GHSA-a": ("low", "critical")}


def test_month_edges_and_old_advisories(db_path, tmp_path):
    at_t0 = [
        advisory("GHSA-1", "2025-07-31T23:59:59.999Z"),
        advisory("GHSA-2", "2025-07-01T00:00:00.000Z"),
        advisory("GHSA-3", "2025-06-30T23:59:59.999Z"),
        advisory("GHSA-4", "2024-08-01T00:00:00.000Z"),
        advisory("GHSA-5", "2024-07-31T23:59:59.999Z"),
    ]
    row = compute(db_path, tmp_path, at_t0, [advisory("GHSA-y", OLD, name="other")])[0]
    assert row["ghsa_high_count"] == 5
    assert row["ghsa_high_count_M0"] == 2
    assert row["ghsa_high_count_M1"] == 1
    assert row["ghsa_high_count_M11"] == 1
    assert sum(row[f"ghsa_high_count_M{month}"] for month in gf.MONTHS) == 4


def test_month_direction_comes_from_config(db_path, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "M0_IS_MOST_RECENT", False)
    row = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY)], [advisory("GHSA-y", OLD, name="other")])[0]
    assert row["ghsa_high_count_M11"] == 1
    assert row["ghsa_high_count_M0"] == 0


def test_package_without_advisories_gets_zero_everywhere(db_path, tmp_path):
    rows = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY)], [advisory("GHSA-b", AT_T0)], names=("pkg", "clean"))
    clean = rows[1]
    assert len(clean) == 72
    assert all(clean[column] == 0 for column in gf.COLUMNS[2:])


def test_names_match_exactly_and_case_only_is_reported(db_path, tmp_path):
    rows = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY, name="PKG")], [advisory("GHSA-b", AT_T0, name="PKG")])
    assert rows[0]["ghsa_high_count"] == 0
    assert rows[0]["target_ghsa_high_count_90d"] == 0
    assert gf.case_only_matches(db_path, ["pkg"]) == [
        ("t0", "PKG", ("pkg",)),
        ("target_end", "PKG", ("pkg",)),
    ]


def test_monthly_above_general_fails_validation(db_path, tmp_path):
    row = compute(db_path, tmp_path, [advisory("GHSA-a", EARLY)], [advisory("GHSA-b", AT_T0)])[0]
    row["ghsa_high_count_M3"] = 5
    with pytest.raises(ValueError, match="monthly"):
        gf.validate([row])


def test_sql_recount_agrees_and_detects_differences(db_path, tmp_path):
    at_end = [
        advisory("GHSA-a", AT_T0, "HIGH", repeat=2),
        advisory("GHSA-b", JUST_BEFORE_END, "HIGH"),
        advisory("GHSA-c", AT_T0, None),
        advisory("GHSA-d", AT_END, "HIGH"),
        advisory("GHSA-e", AT_T0, "HIGH", withdrawn="2025-09-01T00:00:00Z"),
    ]
    rows = compute(db_path, tmp_path, [advisory("GHSA-x", OLD, name="other")], at_end)
    recount = gf.recount_targets_sql(db_path, ["pkg"], T0)
    assert recount == {("pkg", "high"): 2, ("pkg", "unknown"): 1}
    assert gf.compare_targets(rows, recount) == []
    rows[0]["target_ghsa_high_count_90d"] = 9
    assert gf.compare_targets(rows, recount) == [(("pkg", "high"), 9, 2)]
