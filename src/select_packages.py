"""Build data/packages.csv and data/packages_test.csv.

Selection uses only information from before t0. Target-window advisories are used only for the
information-only report and to make sure the pipeline test list contains some positives.

Usage: python -u -m src.select_packages
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import random
import re
import sqlite3
import tarfile
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

import config
from src.http_client import get_json

log = logging.getLogger("select_packages")

HAD_GHSA, POPULAR, RANDOM = "had_ghsa", "popular_no_ghsa", "random_no_ghsa"
STRATA = [HAD_GHSA, POPULAR, RANDOM]
COLUMNS = ["package_name", "stratum", "source", "selected_at", "selection_downloads_feature_year"]


def feature_year(t0):
    """Return the feature window as inclusive dates for the npm API."""
    windows = config.feature_months(t0).values()
    start = min(window[0] for window in windows)
    end = max(window[1] for window in windows) - timedelta(days=1)
    return f"{start:%Y-%m-%d}:{end:%Y-%m-%d}"


def source_label(source):
    return f"{source['name']}@{source['version']}"


def fetch_source(source):
    """Return the path of a pinned npm tarball, verified against its recorded checksum."""
    label = f"{source['name']}-{source['version']}.tgz"
    path = config.SOURCES_DIR / label
    if not path.exists():
        url = f"{config.NPM_REGISTRY_URL}/{source['name']}/-/{label}"
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if hashlib.sha1(path.read_bytes()).hexdigest() != source["shasum"]:
        raise ValueError(f"{path} does not match the checksum in config.py")
    return path


def read_tar_member(path, member):
    """Read one file as data. The archive is never extracted or executed."""
    with tarfile.open(path, "r:gz") as archive:
        return archive.extractfile(member).read()


def parse_high_impact(text):
    return sorted(set(re.findall(r"'((?:@[^'\s/]+/)?[^'\s/]+)'", text)))


def load_popular_pool():
    content = read_tar_member(fetch_source(config.POPULAR_SOURCE), "package/lib/top.js")
    return parse_high_impact(content.decode("utf-8"))


def load_name_list():
    return sorted(set(json.loads(read_tar_member(fetch_source(config.RANDOM_SOURCE), "package/names.json"))))


def had_ghsa_packages(db_path, t0):
    """Packages with a qualifying advisory before t0, as known in the t0 snapshot."""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT DISTINCT package_name FROM advisories"
        " WHERE snapshot = 't0' AND withdrawn IS NULL AND published < ?",
        (config.format_timestamp(t0),),
    ).fetchall()
    db.close()
    if not rows:
        raise ValueError(f"No t0 advisories in {db_path}; run src.advisories on the t0 snapshot")
    return sorted(row[0] for row in rows)


def target_window_packages(db_path, t0):
    """For reporting and the test list only; must never feed into the main selection."""
    start, end = config.target_window(t0)
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT DISTINCT package_name FROM advisories WHERE snapshot = 'target_end'"
        " AND withdrawn IS NULL AND published >= ? AND published < ?",
        (config.format_timestamp(start), config.format_timestamp(end)),
    ).fetchall()
    db.close()
    return {row[0] for row in rows}


def fetch_one(name, window, get=get_json):
    status, body = get(f"{config.NPM_DOWNLOADS_POINT_URL}/{window}/{quote(name, safe='@')}")
    return (body or {}).get("downloads") if status == 200 else None


def fetch_downloads(names, window, get=get_json):
    """Return {name: downloads or None}. Scoped names cannot use npm's bulk endpoint."""
    unscoped = [name for name in names if not name.startswith("@")]
    result = {}
    if len(unscoped) > 1:
        joined = ",".join(quote(name, safe="") for name in unscoped)
        status, body = get(f"{config.NPM_DOWNLOADS_POINT_URL}/{window}/{joined}")
        if status == 200:
            for name in unscoped:
                result[name] = ((body or {}).get(name) or {}).get("downloads")
        else:
            log.warning("Bulk lookup returned %s; asking for %d names singly", status, len(unscoped))
            result.update({name: fetch_one(name, window, get) for name in unscoped})
    elif unscoped:
        result[unscoped[0]] = fetch_one(unscoped[0], window, get)
    for name in names:
        if name.startswith("@"):
            result[name] = fetch_one(name, window, get)
    return result


def select_by_downloads(pool, seed, target, qualifies, fetch, label="", batch_size=None):
    """Shuffle the pool, then check names in that order until target names qualify."""
    batch_size = batch_size or config.NPM_BULK_MAX_NAMES
    order = sorted(pool)
    random.Random(seed).shuffle(order)
    selected, unknown = [], []
    checked = 0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        downloads = fetch(batch)
        for name in batch:
            checked += 1
            value = downloads.get(name)
            if value is None:
                unknown.append(name)
            elif qualifies(value):
                selected.append((name, value))
                if len(selected) == target:
                    return selected, {"pool": len(order), "checked": checked, "unknown": unknown}
        print(f"  {label}: checked {checked}, qualified {len(selected)}/{target}, unknown {len(unknown)}")
    raise ValueError(f"{label}: only {len(selected)} of {target} qualified in the whole pool")


def build_rows(had_ghsa, popular, random_rows, popular_source, random_source, a_source, selected_at):
    rows = [
        {"package_name": name, "stratum": HAD_GHSA, "source": a_source,
         "selected_at": selected_at, "selection_downloads_feature_year": ""}
        for name in sorted(had_ghsa)
    ]
    for stratum, source, selected in (
        (POPULAR, popular_source, popular),
        (RANDOM, random_source, random_rows),
    ):
        rows += [
            {"package_name": name, "stratum": stratum, "source": source,
             "selected_at": selected_at, "selection_downloads_feature_year": downloads}
            for name, downloads in selected
        ]
    validate_rows(rows)
    return rows


def validate_rows(rows):
    names = [row["package_name"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1}) if len(set(names)) != len(names) else []
    if duplicates:
        raise ValueError(f"Duplicate packages across strata: {duplicates[:20]}")


def build_test_list(rows, positives, seed):
    """Pick the pipeline test list. Uses outcome information, so never train on this list."""
    rng = random.Random(seed)
    chosen = []
    for stratum, count in config.TEST_LIST_COUNTS.items():
        members = [row for row in rows if row["stratum"] == stratum]
        rng.shuffle(members)
        if stratum == HAD_GHSA:
            required = [row for row in members if row["package_name"] in positives][:2]
        else:
            required = [row for row in members if row["package_name"].startswith("@")][:1]
        others = [row for row in members if row not in required]
        chosen += required + others[:count - len(required)]
    names = [row["package_name"] for row in chosen]
    if sum(name.startswith("@") for name in names) < 2 or sum(name in positives for name in names) < 2:
        raise ValueError("Test list needs at least 2 scoped packages and 2 target-window positives")
    return chosen


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    """Keep the existing file, and its selected_at, when the selection has not changed."""
    def key(items):
        return [(row["package_name"], row["stratum"], row["source"],
                 str(row["selection_downloads_feature_year"])) for row in items]

    if path.exists() and key(read_csv(path)) == key(rows):
        print(f"{path.name}: selection unchanged, file kept")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: wrote {len(rows)} rows")


def report(rows, test_rows, stats, positives):
    print("\n=== Report ===")
    for stratum in STRATA:
        members = [row for row in rows if row["stratum"] == stratum]
        hits = sum(row["package_name"] in positives for row in members)
        print(f"{stratum}: {len(members)} packages, source {members[0]['source']}, "
              f"target-window positives (information only) {hits}")
    print(f"selected positives {sum(row['package_name'] in positives for row in rows)} "
          f"of {len(positives)} target-window packages overall")
    for stratum, values in stats.items():
        print(f"{stratum}: pool {values['pool']}, checked {values['checked']}, "
              f"unknown downloads {len(values['unknown'])}")
    print("Test list:")
    for row in test_rows:
        print(f"  {row['stratum']:<16} {row['package_name']} "
              f"downloads={row['selection_downloads_feature_year'] or 'NULL'} "
              f"target_window={'yes' if row['package_name'] in positives else 'no'}")
    print("10 example rows:")
    rng = random.Random(config.RANDOM_SEED)
    for stratum, count in zip(STRATA, (4, 3, 3)):
        for row in rng.sample([row for row in rows if row["stratum"] == stratum], count):
            print(f"  {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=config.SAMPLE_PER_STRATUM)
    parser.add_argument("--trial", action="store_true", help="fetch and report, write no output files")
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "select_packages.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    t0 = config.parse_timestamp(config.PREDICTION_DATES[0])
    window = feature_year(t0)
    print(f"t0 {config.format_timestamp(t0)}, feature year {window}, seed {config.RANDOM_SEED}")
    snapshots = json.loads(config.ADVISORY_SNAPSHOTS_JSON.read_text(encoding="utf-8"))
    a_source = f"github-advisory-database@{snapshots['t0']['sha'][:7]}"

    had_ghsa = had_ghsa_packages(config.ADVISORIES_SQLITE, t0)
    excluded = set(had_ghsa)
    popular_names = load_popular_pool()
    all_names = load_name_list()
    print(f"A {len(had_ghsa)}; high-impact list {len(popular_names)}; name list {len(all_names)}; "
          f"A not in name list {len(excluded - set(all_names))}")
    popular_pool = [name for name in popular_names if name not in excluded]
    random_pool = sorted(set(all_names) - excluded - set(popular_names))
    print(f"B pool {len(popular_pool)}; C pool {len(random_pool)}")

    fetch = lambda names: fetch_downloads(names, window)
    popular, popular_stats = select_by_downloads(
        popular_pool, config.RANDOM_SEED, args.sample_size,
        lambda value: value >= config.POPULAR_MIN_DOWNLOADS, fetch, "B",
    )
    selected_random, random_stats = select_by_downloads(
        random_pool, config.RANDOM_SEED, args.sample_size,
        lambda value: config.RANDOM_MIN_DOWNLOADS <= value < config.RANDOM_MAX_DOWNLOADS, fetch, "C",
    )
    stats = {POPULAR: popular_stats, RANDOM: random_stats}
    for stratum, values in stats.items():
        log.info("%s: checked %d of %d, %d unknown", stratum, values["checked"], values["pool"], len(values["unknown"]))
        for name in values["unknown"]:
            log.info("%s unknown downloads (possible survivorship): %s", stratum, name)

    selected_at = config.format_timestamp(datetime.now(timezone.utc))
    rows = build_rows(
        had_ghsa, popular, selected_random, source_label(config.POPULAR_SOURCE),
        source_label(config.RANDOM_SOURCE), a_source, selected_at,
    )
    positives = target_window_packages(config.ADVISORIES_SQLITE, t0)
    test_rows = build_test_list(rows, positives, config.RANDOM_SEED)
    if not args.trial:
        write_csv(config.PACKAGES_CSV, rows)
        write_csv(config.PACKAGES_TEST_CSV, test_rows)
        config.SELECTION_STATS_JSON.write_text(
            json.dumps(
                {"seed": config.RANDOM_SEED, "feature_year": window,
                 "sources": {POPULAR: config.POPULAR_SOURCE, RANDOM: config.RANDOM_SOURCE},
                 "strata": stats},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    report(rows, test_rows, stats, positives)


if __name__ == "__main__":
    main()
