import gzip
import json

import pytest

import config
from src import npm_meta as nm
from src.http_client import cache_path, get_json

T0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")
BEFORE = "2025-01-15T10:00:00.000Z"
AFTER = "2025-09-01T00:00:00.000Z"


def packument(times, manifests=None, created="2020-01-01T00:00:00Z"):
    """Build a fake packument. Every version has a manifest unless manifests is given."""
    if manifests is None:
        manifests = {version: {} for version in times}
    return {"time": {"created": created, "modified": AFTER, **times}, "versions": manifests}


@pytest.mark.parametrize(
    "times, expected",
    [
        ({"1.9.0": BEFORE, "1.10.0": BEFORE}, "1.10.0"),
        ({"1.0.0": BEFORE, "2.0.0-beta.1": BEFORE}, "1.0.0"),
        ({"1.0.0": BEFORE, "2.0.0": AFTER}, "1.0.0"),
        ({"1.0.0": BEFORE, "2.0.0": "2025-08-01T00:00:00.000Z"}, "1.0.0"),
        ({"1.0.0": BEFORE, "1.0.1": "2025-07-31T23:59:59.999Z"}, "1.0.1"),
        ({"1.0.0-alpha.2": BEFORE, "1.0.0-alpha.10": BEFORE, "1.0.0": AFTER}, "1.0.0-alpha.10"),
        ({"0.0.1-security": BEFORE}, "0.0.1-security"),
        ({"1.0.0": BEFORE, "1.0.0beta": BEFORE, "9.9": BEFORE}, "1.0.0"),
    ],
)
def test_version_choice(times, expected):
    assert nm.choose_version(packument(times), T0)[0] == expected


def test_highest_version_with_a_manifest_is_chosen_and_the_other_reported():
    times = {"1.0.0": BEFORE, "2.0.0": BEFORE}
    assert nm.choose_version(packument(times, {"1.0.0": {}}), T0) == ("1.0.0", ["2.0.0"])


@pytest.mark.parametrize(
    "fake, reason",
    [
        (packument({"1.0.0": AFTER}), "no version published before t0"),
        (packument({"1.0beta": BEFORE}), "no strict semver"),
        (packument({"1.0.0": BEFORE}, {}), "no version before t0 still has a manifest"),
        ({"time": {"created": BEFORE, "modified": AFTER, "unpublished": {"time": AFTER}}}, "fully unpublished"),
        ({}, "no version published before t0"),
    ],
)
def test_packages_that_are_skipped(fake, reason):
    with pytest.raises(nm.SkipPackage, match=reason):
        nm.choose_version(fake, T0)


@pytest.mark.parametrize(
    "name, version, expected",
    [
        ("react", "19.1.0", "pkg:npm/react@19.1.0"),
        ("@types/node", "22.0.0", "pkg:npm/%40types/node@22.0.0"),
        ("a", "1.0.0+build.5", "pkg:npm/a@1.0.0%2Bbuild.5"),
        ("a", "1.0.0-beta.1", "pkg:npm/a@1.0.0-beta.1"),
        ("JSONStream", "1.3.5", "pkg:npm/JSONStream@1.3.5"),
    ],
)
def test_purl(name, version, expected):
    assert nm.build_purl(name, version) == expected


@pytest.mark.parametrize(
    "manifest, expected",
    [
        ({"dependencies": {"a": "1", "b": "1"}, "bundleDependencies": True}, 2),
        ({"dependencies": {"a": "1"}, "bundledDependencies": "true"}, 1),
        ({"bundleDependencies": True}, 0),
        ({"dependencies": {"a": "1"}, "bundleDependencies": False}, 0),
        ({"bundleDependencies": ["a", "a", "b"], "bundledDependencies": ["b", "c"]}, 3),
        ({}, 0),
    ],
)
def test_bundled_count(manifest, expected):
    assert nm.dependency_counts(manifest)["bundled_dependency_count"] == expected


def test_other_dependency_counts():
    counts = nm.dependency_counts(
        {
            "dependencies": {"a": "1", "opt": "1"},
            "optionalDependencies": {"opt": "1"},
            "peerDependencies": {"p": "1", "q": "1"},
            "peerDependenciesMeta": {"q": {"optional": True}, "r": {"optional": True}, "s": {"optional": False}},
            "devDependencies": {"d": "1"},
        }
    )
    assert counts == {
        "dependency_count": 2,
        "optional_dependency_count": 1,
        "peer_dependency_count": 3,
        "dev_dependency_count": 1,
        "bundled_dependency_count": 0,
    }
    assert nm.dependency_counts({})["dependency_count"] == 0


def test_legacy_dependency_shapes_do_not_crash():
    assert nm.dependency_counts({"dependencies": ["a", "b"]})["dependency_count"] == 2
    assert nm.dependency_counts({"dependencies": "a"})["dependency_count"] == 0


def test_month_bucketing_at_the_edges():
    counts = nm.stable_release_counts(
        packument(
            {
                "1.0.0": "2025-07-31T23:59:59.999Z",  # last moment of M0
                "1.0.1": "2025-07-01T00:00:00.000Z",  # first moment of M0
                "1.0.2": "2025-06-30T23:59:59.999Z",  # M1
                "1.0.3": "2024-08-01T00:00:00.000Z",  # first moment of M11
                "1.0.4": "2024-07-31T23:59:59.999Z",  # before the window
                "1.0.5": "2025-08-01T00:00:00.000Z",  # t0, excluded
                "1.1.0-rc.1": "2025-07-15T00:00:00.000Z",  # prerelease, not counted
                "1.0.6+build": "2025-07-15T00:00:00.000Z",  # build metadata is still stable
                "1.0beta": "2025-07-15T00:00:00.000Z",  # not strict semver
            }
        ),
        T0,
    )
    assert counts["package_stable_release_count_M0"] == 3
    assert counts["package_stable_release_count_M1"] == 1
    assert counts["package_stable_release_count_M11"] == 1
    assert sum(counts.values()) == 5
    assert len(counts) == 12


def test_versions_unpublished_later_are_still_counted():
    fake = packument({"1.0.0": "2025-07-10T00:00:00Z", "1.0.1": "2025-07-11T00:00:00Z"}, {"1.0.0": {}})
    assert nm.stable_release_counts(fake, T0)["package_stable_release_count_M0"] == 2


def test_month_direction_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "M0_IS_MOST_RECENT", False)
    counts = nm.stable_release_counts(packument({"1.0.0": "2025-07-10T00:00:00Z"}), T0)
    assert counts["package_stable_release_count_M11"] == 1
    assert counts["package_stable_release_count_M0"] == 0


def test_build_row_formats_timestamps_and_keeps_missing_sizes_null():
    manifests = {"1.0.0": {"dist": {"unpackedSize": 500}}}
    row, passed_over = nm.build_row(
        "@s/a", packument({"1.0.0": "2025-01-15T10:00:00Z"}, manifests, created="2020-01-01T00:00:00.5Z"), T0
    )
    assert row["purl"] == "pkg:npm/%40s/a@1.0.0"
    assert row["package_created_at"] == "2020-01-01T00:00:00.500Z"
    assert row["published_at"] == "2025-01-15T10:00:00.000Z"
    assert row["prediction_date"] == "2025-08-01T00:00:00.000Z"
    assert row["unpacked_size"] == 500
    assert row["file_count"] is None
    assert passed_over == []
    assert set(row) == set(nm.COLUMNS)


def test_features_contain_no_post_t0_information():
    early = packument({"1.0.0": BEFORE})
    later = packument({"1.0.0": BEFORE, "2.0.0": AFTER, "3.0.0": "2026-01-01T00:00:00Z"})
    later["dist-tags"] = {"latest": "3.0.0"}
    assert nm.build_row("a", early, T0)[0] == nm.build_row("a", later, T0)[0]


def test_missing_created_is_skipped_not_invented():
    fake = {"time": {"1.0.0": BEFORE}, "versions": {"1.0.0": {}}}
    with pytest.raises(nm.SkipPackage, match="created"):
        nm.build_row("a", fake, T0)


def test_collect_records_registry_failures(monkeypatch):
    monkeypatch.setattr(nm, "fetch_packument", lambda name: (404, {"error": "not found"}))
    rows, failures, _ = nm.collect(["gone"], [T0])
    assert rows == []
    assert failures == [("gone", "2025-08-01T00:00:00.000Z", "registry returned HTTP 404")]


def test_http_client_sends_headers_and_compresses_cache(tmp_path, monkeypatch):
    seen = []

    class Response:
        status_code = 200
        content = b"{}"
        headers = {}

        def json(self):
            return {"time": {"created": BEFORE}}

    def fake_get(url, timeout, headers):
        seen.append(headers)
        return Response()

    monkeypatch.setattr("src.http_client._session.get", fake_get)
    monkeypatch.setattr("src.http_client.time.sleep", lambda seconds: None)
    accept = {"Accept": "application/json"}
    first = get_json("https://example.test/p", tmp_path, accept, compress=True)
    second = get_json("https://example.test/p", tmp_path, accept, compress=True)
    assert first == second == (200, {"time": {"created": BEFORE}})
    assert seen == [accept]
    stored = cache_path("https://example.test/p", tmp_path, True)
    assert json.loads(gzip.decompress(stored.read_bytes()))["status"] == 200
