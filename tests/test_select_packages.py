import pytest

import config
from src import select_packages as sp
from src.http_client import get_json

POOL = [f"pkg-{i}" for i in range(300)] + [f"@scope/pkg-{i}" for i in range(100)]


def fake_fetch(names):
    """Every 7th name is unknown; downloads otherwise depend only on the name."""
    return {name: None if hash_of(name) % 7 == 0 else hash_of(name) * 100 for name in names}


def hash_of(name):
    return sum(ord(character) for character in name)


def select(seed=42, target=40, pool=POOL):
    return sp.select_by_downloads(pool, seed, target, lambda value: value >= 80000, fake_fetch, "T", 16)


def build_all(a=("a1", "@s/a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10")):
    popular = [(f"p{i}", 2_000_000) for i in range(8)] + [("@s/p", 2_000_000)]
    rand = [(f"r{i}", 5000) for i in range(8)] + [("@s/r", 5000)]
    return sp.build_rows(a, popular, rand, "B@1", "C@1", "A@1", "2025-01-01T00:00:00.000Z")


def test_same_seed_gives_same_list_and_order_ignores_input_order():
    assert select() == select()
    assert select()[0] == select(pool=list(reversed(POOL)))[0]


def test_different_seed_gives_different_list():
    assert select(seed=42)[0] != select(seed=43)[0]


def test_stops_at_target_and_records_checked_and_unknown():
    selected, stats = select()
    assert len(selected) == 40
    assert stats["pool"] == len(POOL)
    assert 40 <= stats["checked"] < len(POOL)
    assert stats["unknown"] and all(hash_of(name) % 7 == 0 for name in stats["unknown"])
    assert all(value >= 80000 for _, value in selected)


def test_too_small_a_pool_fails_loudly():
    with pytest.raises(ValueError, match="qualified"):
        select(target=len(POOL))


def test_no_duplicates_and_strata_do_not_overlap():
    rows = build_all()
    names = [row["package_name"] for row in rows]
    assert len(names) == len(set(names))
    with pytest.raises(ValueError, match="Duplicate"):
        build_all(a=("a1", "p0"))


def test_group_a_has_null_downloads_and_others_have_values():
    rows = build_all()
    assert {row["selection_downloads_feature_year"] for row in rows if row["stratum"] == sp.HAD_GHSA} == {""}
    assert all(row["selection_downloads_feature_year"] for row in rows if row["stratum"] != sp.HAD_GHSA)


def test_test_list_composition_and_seed():
    rows = build_all()
    positives = {"a3", "a9", "not-selected"}
    chosen = sp.build_test_list(rows, positives, 42)
    names = [row["package_name"] for row in chosen]
    assert len(names) == len(set(names)) == 20
    for stratum, count in config.TEST_LIST_COUNTS.items():
        assert sum(row["stratum"] == stratum for row in chosen) == count
    assert sum(name.startswith("@") for name in names) >= 2
    assert sum(name in positives for name in names) >= 2
    assert chosen == sp.build_test_list(rows, positives, 42)


def test_test_list_fails_without_positives():
    with pytest.raises(ValueError, match="positives"):
        sp.build_test_list(build_all(), set(), 42)


def test_feature_year_is_the_full_twelve_months():
    t0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")
    assert sp.feature_year(t0) == "2024-08-01:2025-07-31"


def test_high_impact_parser_keeps_exact_scoped_names():
    text = "export const top = [\n  'semver',\n  '@types/node',\n  'JSONStream'\n]\n"
    assert sp.parse_high_impact(text) == ["@types/node", "JSONStream", "semver"]


def test_fetch_downloads_uses_bulk_singles_and_marks_unknown():
    urls = []

    def fake_get(url):
        urls.append(url)
        if "," in url:
            return 200, {"a": {"downloads": 5}, "b": None}
        if "%2F" in url:
            return 404, {"error": "not found"}
        return 200, {"downloads": 9}

    result = sp.fetch_downloads(["a", "b", "@s/c"], "x:y", fake_get)
    assert result == {"a": 5, "b": None, "@s/c": None}
    assert len(urls) == 2
    assert sp.fetch_downloads(["solo"], "x:y", fake_get) == {"solo": 9}


def test_fetch_downloads_falls_back_to_singles_when_bulk_is_refused():
    def fake_get(url):
        return (400, {"error": "bad"}) if "," in url else (200, {"downloads": 1})

    assert sp.fetch_downloads(["a", "b"], "x:y", fake_get) == {"a": 1, "b": 1}


def test_http_client_retries_then_caches(tmp_path, monkeypatch):
    responses = [FakeResponse(429, {}), FakeResponse(404, {"error": "missing"})]
    calls = []

    def fake_get(url, timeout, headers):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr("src.http_client._session.get", fake_get)
    monkeypatch.setattr("src.http_client.time.sleep", lambda seconds: None)
    assert get_json("https://example.test/x", tmp_path) == (404, {"error": "missing"})
    assert get_json("https://example.test/x", tmp_path) == (404, {"error": "missing"})
    assert len(calls) == 2


def test_http_client_gives_up_loudly_and_does_not_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.http_client._session.get", lambda url, timeout, headers: FakeResponse(503, {}))
    monkeypatch.setattr("src.http_client.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Giving up"):
        get_json("https://example.test/y", tmp_path)
    assert not list(tmp_path.iterdir())


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body
        self.headers = {}

    def json(self):
        return self.body


@pytest.mark.skipif(not config.PACKAGES_CSV.exists(), reason="packages.csv not built yet")
def test_built_files_are_valid():
    rows = sp.read_csv(config.PACKAGES_CSV)
    names = [row["package_name"] for row in rows]
    assert len(names) == len(set(names))
    assert {row["stratum"] for row in rows} == set(sp.STRATA)
    for stratum in (sp.POPULAR, sp.RANDOM):
        assert sum(row["stratum"] == stratum for row in rows) == config.SAMPLE_PER_STRATUM
    for row in rows:
        value = row["selection_downloads_feature_year"]
        if row["stratum"] == sp.HAD_GHSA:
            assert value == ""
        elif row["stratum"] == sp.POPULAR:
            assert int(value) >= config.POPULAR_MIN_DOWNLOADS
        else:
            assert config.RANDOM_MIN_DOWNLOADS <= int(value) < config.RANDOM_MAX_DOWNLOADS
    test_names = [row["package_name"] for row in sp.read_csv(config.PACKAGES_TEST_CSV)]
    assert len(test_names) == len(set(test_names)) == 20
    assert set(test_names) <= set(names)
