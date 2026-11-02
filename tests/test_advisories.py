import json
import logging
import sqlite3
import subprocess
from pathlib import Path

import pytest
import requests

import config
from src import advisories
from src.advisories import MissingAdvisoryFiles, build_table, summarize

T0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")


def git_add(folder):
    """Index the fake advisories in the throwaway repo under tmp_path."""
    subprocess.run(["git", "add", "."], cwd=folder, check=True)


def osv(ghsa_id, ecosystem="npm", name="left-pad", published="2025-08-10T12:00:00Z", **extra):
    return {
        "id": ghsa_id,
        "published": published,
        "modified": "2025-09-01T00:00:00Z",
        "affected": [
            {"package": {"ecosystem": ecosystem, "name": name}, "ranges": [{"a": 1}]},
            {"package": {"ecosystem": ecosystem, "name": name}, "ranges": [{"b": 2}]},
        ],
        "database_specific": {"severity": "HIGH"},
        **extra,
    }


@pytest.fixture
def snapshot(tmp_path):
    folder = tmp_path / "latest"
    advisories = folder / config.ADVISORY_SUBDIR / "2025" / "08"
    advisories.mkdir(parents=True)
    fake = [
        osv("GHSA-npm"),
        osv("GHSA-pypi", ecosystem="PyPI", name="django"),
        osv("GHSA-gone", withdrawn="2025-09-01T00:00:00Z"),
        osv("GHSA-early", name="JSONStream", published="2025-01-01T00:00:00Z",
            database_specific={"cwe_ids": ["CWE-79", "CWE-506"]}),
    ]
    for advisory in fake:
        (advisories / f"{advisory['id']}.json").write_text(json.dumps(advisory), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=folder, check=True)
    git_add(folder)
    return folder


def rows(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    result = {row["ghsa_id"]: dict(row) for row in db.execute("SELECT * FROM advisories")}
    db.close()
    return result


def test_keeps_npm_and_drops_other_ecosystems(snapshot, tmp_path):
    db_path = tmp_path / "a.sqlite"
    assert build_table(snapshot, db_path) == (3, 0)
    stored = rows(db_path)
    assert set(stored) == {"GHSA-npm", "GHSA-gone", "GHSA-early"}
    assert stored["GHSA-npm"]["published"] == "2025-08-10T12:00:00.000Z"
    assert stored["GHSA-npm"]["severity"] == "high"
    assert stored["GHSA-npm"]["withdrawn"] is None
    assert stored["GHSA-npm"]["file_path"].endswith("2025/08/GHSA-npm.json")
    assert stored["GHSA-early"]["severity"] is None
    assert stored["GHSA-early"]["package_name"] == "JSONStream"


def test_withdrawn_is_stored_but_not_counted_as_active(snapshot, tmp_path):
    db_path = tmp_path / "a.sqlite"
    build_table(snapshot, db_path)
    assert rows(db_path)["GHSA-gone"]["withdrawn"] == "2025-09-01T00:00:00.000Z"
    summary, severity, non_lowercase, _ = summarize(db_path, "latest", T0)
    assert summary["npm_advisories"] == 3
    assert summary["withdrawn_advisories"] == 1
    assert summary["advisories_in_target_window"] == 1
    assert summary["target_packages_first_time"] == 1
    assert summary["target_packages_with_earlier_advisory"] == 0
    assert severity == [("high", 1)]
    assert non_lowercase == ["JSONStream"]


def test_missing_file_fails_loudly_and_keeps_existing_table(snapshot, tmp_path):
    db_path = tmp_path / "a.sqlite"
    build_table(snapshot, db_path)
    removed = snapshot / config.ADVISORY_SUBDIR / "2025" / "08" / "GHSA-npm.json"
    removed.unlink()
    with pytest.raises(MissingAdvisoryFiles, match=r"(?s)1 advisory file.*GHSA-npm\.json"):
        build_table(snapshot, db_path)
    assert len(rows(db_path)) == 3


def test_file_that_cannot_be_read_fails_loudly(snapshot, tmp_path, monkeypatch):
    def blocked(self, *args, **kwargs):
        if self.name == "GHSA-pypi.json":
            raise OSError(22, "Invalid argument")
        return original(self, *args, **kwargs)

    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", blocked)
    with pytest.raises(MissingAdvisoryFiles, match="GHSA-pypi.json"):
        build_table(snapshot, tmp_path / "a.sqlite")
    assert not (tmp_path / "a.sqlite").exists()


def test_folder_that_is_not_its_own_repo_fails_loudly(snapshot, tmp_path):
    (snapshot / "nested").mkdir()
    with pytest.raises(MissingAdvisoryFiles, match="not its own git repository"):
        build_table(snapshot / "nested", tmp_path / "a.sqlite")


def test_malformed_json_is_counted_not_fatal(snapshot, tmp_path):
    bad = snapshot / config.ADVISORY_SUBDIR / "2025" / "08" / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    git_add(snapshot)
    assert build_table(snapshot, tmp_path / "a.sqlite") == (3, 1)


def test_is_cwe506_flag_keeps_advisory_as_normal_row(snapshot, tmp_path):
    db_path = tmp_path / "a.sqlite"
    build_table(snapshot, db_path)
    stored = rows(db_path)
    assert stored["GHSA-early"]["is_cwe506"] == 1
    assert stored["GHSA-npm"]["is_cwe506"] == 0
    assert summarize(db_path, "latest", T0)[0]["cwe506_advisories"] == 1


@pytest.fixture
def allowed(snapshot, monkeypatch):
    """Track an allow-listed advisory whose local content must never be used."""
    path = snapshot / config.ADVISORY_SUBDIR / "2025" / "08" / "GHSA-allowed.json"
    path.write_text("{this file must not be opened", encoding="utf-8")
    git_add(snapshot)
    monkeypatch.setattr(config, "ALLOWED_BLOCKED_ADVISORIES", frozenset({"GHSA-allowed"}))
    monkeypatch.setattr(advisories, "allowed_advisory_ecosystems", lambda ghsa_id: ["composer"])
    return path


def test_allowed_file_on_disk_is_not_opened_and_is_logged(snapshot, allowed, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="advisories"):
        assert build_table(snapshot, tmp_path / "a.sqlite") == (3, 0)
    assert "allowed GHSA-allowed (on disk, ecosystems: composer)" in caplog.text


def test_allowed_file_may_be_missing_and_is_logged(snapshot, allowed, tmp_path, caplog):
    allowed.unlink()
    with caplog.at_level(logging.INFO, logger="advisories"):
        assert build_table(snapshot, tmp_path / "a.sqlite") == (3, 0)
    assert "allowed GHSA-allowed (missing, ecosystems: composer)" in caplog.text


def test_other_missing_file_still_fails_with_allow_list(snapshot, allowed, tmp_path):
    allowed.unlink()
    (allowed.parent / "GHSA-npm.json").unlink()
    with pytest.raises(MissingAdvisoryFiles, match="GHSA-npm") as caught:
        build_table(snapshot, tmp_path / "a.sqlite")
    assert "GHSA-allowed" not in str(caught.value)


@pytest.mark.parametrize("ecosystems", [["composer", "npm"], ["NPM"], []])
def test_allowed_file_fails_if_npm_or_unconfirmed(snapshot, allowed, tmp_path, monkeypatch, ecosystems):
    monkeypatch.setattr(advisories, "allowed_advisory_ecosystems", lambda ghsa_id: ecosystems)
    with pytest.raises(MissingAdvisoryFiles, match="cannot be skipped"):
        build_table(snapshot, tmp_path / "a.sqlite")
    assert not (tmp_path / "a.sqlite").exists()


def test_allowed_file_fails_if_api_is_unavailable(snapshot, allowed, tmp_path, monkeypatch):
    def unavailable(ghsa_id):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(advisories, "allowed_advisory_ecosystems", unavailable)
    with pytest.raises(requests.ConnectionError):
        build_table(snapshot, tmp_path / "a.sqlite")


def test_committed_file_answers_every_allowed_id_without_npm():
    assert config.ALLOWED_BLOCKED_FILES_JSON == config.DATA_DIR / "allowed_blocked_files.json"
    answers = json.loads(config.ALLOWED_BLOCKED_FILES_JSON.read_text(encoding="utf-8"))
    assert set(answers) == config.ALLOWED_BLOCKED_ADVISORIES
    assert all(ecosystems and "npm" not in ecosystems for ecosystems in answers.values())


def test_allowed_ecosystems_use_cache_and_fetch_only_new_ids(tmp_path, monkeypatch):
    cache = tmp_path / "allowed_blocked_files.json"
    cache.write_text(json.dumps({"GHSA-cached": ["composer"]}), encoding="utf-8")
    monkeypatch.setattr(config, "ALLOWED_BLOCKED_FILES_JSON", cache)
    requested = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"details": "not cached", "vulnerabilities": [{"package": {"ecosystem": "pip"}}]}

    def fake_get(url, timeout):
        requested.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    assert advisories.allowed_advisory_ecosystems("GHSA-cached") == ["composer"]
    assert requested == []
    assert advisories.allowed_advisory_ecosystems("GHSA-new") == ["pip"]
    assert advisories.allowed_advisory_ecosystems("GHSA-new") == ["pip"]
    assert len(requested) == 1
    assert json.loads(cache.read_text(encoding="utf-8")) == {
        "GHSA-cached": ["composer"],
        "GHSA-new": ["pip"],
    }


def test_rerun_replaces_only_its_own_snapshot(snapshot, tmp_path):
    db_path = tmp_path / "a.sqlite"
    build_table(snapshot, db_path)
    assert build_table(snapshot, db_path) == (3, 0)
    assert len(rows(db_path)) == 3
