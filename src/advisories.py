"""Parse a reviewed-advisory snapshot into SQLite and report on its npm advisories.

Usage: python -m src.advisories data/cache/advisory-db/latest
"""
import argparse
import json
import logging
import sqlite3
import subprocess
from pathlib import Path

import requests

import config

log = logging.getLogger("advisories")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS advisories (
    snapshot TEXT NOT NULL,
    ghsa_id TEXT NOT NULL,
    package_name TEXT NOT NULL,
    published TEXT,
    modified TEXT,
    withdrawn TEXT,
    severity TEXT,
    is_cwe506 INTEGER NOT NULL CHECK (is_cwe506 IN (0, 1)),
    file_path TEXT NOT NULL,
    PRIMARY KEY (snapshot, ghsa_id, package_name)
)
"""


def normalize_timestamp(value):
    """Return the project's timestamp format, or None if missing. Bad values raise ValueError."""
    if not value:
        return None
    return config.format_timestamp(config.parse_timestamp(value))


def parse_advisory(path, snapshot_dir):
    """Return one row per distinct npm package named by an advisory file."""
    advisory = json.loads(path.read_text(encoding="utf-8"))
    names = {
        entry["package"]["name"]
        for entry in advisory.get("affected") or []
        if entry.get("package", {}).get("ecosystem") == "npm"
    }
    database_specific = advisory.get("database_specific") or {}
    severity = database_specific.get("severity")
    is_cwe506 = int("CWE-506" in (database_specific.get("cwe_ids") or []))
    return [
        {
            "ghsa_id": advisory["id"],
            "package_name": name,
            "published": normalize_timestamp(advisory.get("published")),
            "modified": normalize_timestamp(advisory.get("modified")),
            "withdrawn": normalize_timestamp(advisory.get("withdrawn")),
            "severity": severity.lower() if severity else None,
            "is_cwe506": is_cwe506,
            "file_path": path.relative_to(snapshot_dir).as_posix(),
        }
        for name in sorted(names)
    ]


class MissingAdvisoryFiles(Exception):
    """Files tracked by the snapshot are absent or unreadable, so counts would be too low."""


def tracked_files(snapshot_dir):
    """Advisory paths in the snapshot's own git index. Fails if the folder is not its own repo."""

    def git(*args):
        command = ["git", *args]
        return subprocess.run(
            command, cwd=snapshot_dir, check=True, capture_output=True, text=True, encoding="utf-8"
        ).stdout

    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != Path(snapshot_dir).resolve():
        raise MissingAdvisoryFiles(f"{snapshot_dir} is not its own git repository")
    paths = [path for path in git("ls-files", "-z", "--", config.ADVISORY_SUBDIR).split("\0") if path]
    if not paths:
        raise MissingAdvisoryFiles(f"{snapshot_dir} tracks no files under {config.ADVISORY_SUBDIR}")
    return paths


def is_allowed_blocked(path):
    return Path(path).stem in config.ALLOWED_BLOCKED_ADVISORIES


def allowed_advisory_ecosystems(ghsa_id):
    """Ecosystems GitHub lists for an allowed advisory, without reading its local file.

    The network is used only for an id not yet in the cache file. Only the ecosystem names are
    cached: the full response quotes the text Defender flags.
    """
    path = config.ALLOWED_BLOCKED_FILES_JSON
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if ghsa_id not in cached:
        response = requests.get(f"{config.GITHUB_ADVISORIES_API}/{ghsa_id}", timeout=60)
        response.raise_for_status()
        cached[ghsa_id] = sorted(
            {item["package"]["ecosystem"] for item in response.json().get("vulnerabilities") or []}
        )
        path.write_text(json.dumps(cached, indent=2) + "\n", encoding="utf-8")
    return cached[ghsa_id]


def verify_allowed_files(snapshot_dir):
    """Log the allowed files this snapshot tracks. Raise if one is npm or cannot be confirmed."""
    allowed = [path for path in tracked_files(snapshot_dir) if is_allowed_blocked(path)]
    log.info("%s: %d allowed Defender-blocked file(s), not opened", snapshot_dir, len(allowed))
    for path in allowed:
        ghsa_id = Path(path).stem
        ecosystems = allowed_advisory_ecosystems(ghsa_id)
        state = "on disk" if (Path(snapshot_dir) / path).is_file() else "missing"
        log.info("  allowed %s (%s, ecosystems: %s)", ghsa_id, state, ", ".join(ecosystems))
        if not ecosystems or "npm" in {ecosystem.lower() for ecosystem in ecosystems}:
            raise MissingAdvisoryFiles(
                f"Allowed file {path} cannot be skipped: GitHub reports ecosystems {ecosystems}"
            )


def check_no_missing_files(snapshot_dir, unreadable=()):
    """Raise, listing every file, unless all tracked files not on the allow list are available."""
    problems = sorted(
        {
            path
            for path in tracked_files(snapshot_dir)
            if not is_allowed_blocked(path) and not (Path(snapshot_dir) / path).is_file()
        }
        | set(unreadable)
    )
    if problems:
        listing = "\n  ".join(problems)
        raise MissingAdvisoryFiles(
            f"{len(problems)} advisory file(s) missing or unreadable in {snapshot_dir}; "
            f"table not updated. Restore with: git checkout <sha> -- <path>\n  {listing}"
        )


def build_table(snapshot_dir, db_path):
    """Replace this snapshot's rows, named after its folder. Returns (rows, failed files).

    Raises MissingAdvisoryFiles before touching the table if any tracked file is unavailable.
    The check runs again after parsing because antivirus can remove files when they are read.
    """
    snapshot_dir = Path(snapshot_dir)
    verify_allowed_files(snapshot_dir)
    check_no_missing_files(snapshot_dir)
    files = sorted(
        path
        for path in (snapshot_dir / config.ADVISORY_SUBDIR).rglob("*.json")
        if not is_allowed_blocked(path)
    )
    rows, failed, unreadable = [], 0, []
    for path in files:
        try:
            rows += parse_advisory(path, snapshot_dir)
        except OSError as error:
            unreadable.append(path.relative_to(snapshot_dir).as_posix())
            log.error("Could not read %s: %r", path, error)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            failed += 1
            log.warning("Could not parse %s: %r", path, error)
    check_no_missing_files(snapshot_dir, unreadable)
    snapshot = snapshot_dir.name
    with sqlite3.connect(db_path) as db:
        # The table is a rebuildable cache, so an older layout is simply dropped.
        columns = [row[1] for row in db.execute("PRAGMA table_info(advisories)")]
        if columns and "is_cwe506" not in columns:
            db.execute("DROP TABLE advisories")
        db.execute(CREATE_TABLE)
        db.execute("DELETE FROM advisories WHERE snapshot = ?", (snapshot,))
        db.executemany(
            "INSERT INTO advisories VALUES (:snapshot, :ghsa_id, :package_name, :published,"
            " :modified, :withdrawn, :severity, :is_cwe506, :file_path)",
            [{**row, "snapshot": snapshot} for row in rows],
        )
    return len(rows), failed


def summarize(db_path, snapshot, t0):
    """Inventory counts. Package and window counts use non-withdrawn advisories only."""
    t0_text = config.format_timestamp(t0)
    end_text = config.format_timestamp(config.target_window(t0)[1])
    db = sqlite3.connect(db_path)

    def scalar(where, select="COUNT(DISTINCT ghsa_id)", params=()):
        query = f"SELECT {select} FROM advisories WHERE snapshot = ? AND {where}"
        return db.execute(query, (snapshot, *params)).fetchone()[0]

    active = "withdrawn IS NULL"
    before = f"{active} AND published < ?"
    in_window = f"{active} AND published >= ? AND published < ?"
    packages = "COUNT(DISTINCT package_name)"
    pairs = "COUNT(*)"
    window = (t0_text, end_text)
    overlap = scalar(
        f"{in_window} AND package_name IN (SELECT package_name FROM advisories"
        f" WHERE snapshot = ? AND {before})",
        packages,
        (*window, snapshot, t0_text),
    )
    window_packages = scalar(in_window, packages, window)
    summary = {
        "npm_advisories": scalar("1"),
        "npm_pairs": scalar("1", pairs),
        "npm_packages": scalar("1", packages),
        "withdrawn_advisories": scalar("withdrawn IS NOT NULL"),
        "active_advisories": scalar(active),
        "active_pairs": scalar(active, pairs),
        "target_window_pairs": scalar(in_window, pairs, window),
        "cwe506_advisories": scalar("is_cwe506 = 1"),
        "cwe506_active_advisories": scalar(f"{active} AND is_cwe506 = 1"),
        "cwe506_target_window_advisories": scalar(f"{in_window} AND is_cwe506 = 1", params=window),
        "modified_after_published": scalar("modified > published"),
        "modified_on_or_after_t0_published_before": scalar(
            "published < ? AND modified >= ?", params=(t0_text, t0_text)
        ),
        "packages_active": scalar(active, packages),
        "packages_with_advisory_before_t0": scalar(before, packages, (t0_text,)),
        "advisories_in_target_window": scalar(in_window, params=window),
        "packages_in_target_window": window_packages,
        "target_packages_with_earlier_advisory": overlap,
        "target_packages_first_time": window_packages - overlap,
        "advisories_missing_published": scalar("published IS NULL"),
    }
    severity = db.execute(
        "SELECT COALESCE(severity, 'NULL'), COUNT(DISTINCT ghsa_id) FROM advisories"
        f" WHERE snapshot = ? AND {in_window} GROUP BY 1 ORDER BY 2 DESC",
        (snapshot, *window),
    ).fetchall()
    non_lowercase = [
        row[0]
        for row in db.execute(
            "SELECT DISTINCT package_name FROM advisories WHERE snapshot = ?"
            " AND package_name <> LOWER(package_name) ORDER BY 1",
            (snapshot,),
        )
    ]
    examples = db.execute(
        "SELECT ghsa_id, package_name, published, modified, withdrawn, severity, is_cwe506, file_path"
        " FROM advisories WHERE snapshot = ? ORDER BY ghsa_id LIMIT 5",
        (snapshot,),
    ).fetchall()
    db.close()
    return summary, severity, non_lowercase, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--db", type=Path, default=config.ADVISORIES_SQLITE)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.LOG_DIR / "advisories.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    rows, failed = build_table(args.snapshot_dir, args.db)
    print(f"Rows written: {rows}; files that failed to parse: {failed}")

    t0 = config.parse_timestamp(config.PREDICTION_DATES[0])
    summary, severity, non_lowercase, examples = summarize(args.db, args.snapshot_dir.name, t0)
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"target window severity: {severity}")
    print(f"non-lowercase package names ({len(non_lowercase)}): {non_lowercase}")
    print("examples:")
    for example in examples:
        print(f"  {example}")


if __name__ == "__main__":
    main()
