"""Compute npm metadata features from full packuments, as known before each prediction date.

A packument is today's state, so everything published on or after t0 is ignored. Today's
latest version is used only for the printed report and never reaches the feature file.

Usage: python -u -m src.npm_meta [--packages data/packages_test.csv]
"""
import argparse
import csv
import logging
from pathlib import Path
from urllib.parse import quote

import semver

import config
from src.http_client import get_json

log = logging.getLogger("npm_meta")

TIME_METADATA_KEYS = {"created", "modified", "unpublished"}
MONTHS = range(config.FEATURE_MONTHS)
RELEASE_COLUMNS = [f"package_stable_release_count_M{month}" for month in MONTHS]
COLUMNS = [
    "package_name", "prediction_date", "chosen_version", "purl", "package_created_at", "published_at",
    "unpacked_size", "file_count", "dependency_count", "optional_dependency_count",
    "peer_dependency_count", "dev_dependency_count", "bundled_dependency_count", *RELEASE_COLUMNS,
]


class SkipPackage(Exception):
    """The package cannot produce a row; the message is the reason that gets logged."""


def parse_version(version):
    """Return a semver Version, or None for a string that is not strict semver."""
    try:
        return semver.Version.parse(version)
    except (TypeError, ValueError):
        return None


def version_times(packument):
    """Return {version: publish datetime} from "time", including versions unpublished later."""
    times = {}
    for version, value in (packument.get("time") or {}).items():
        if version in TIME_METADATA_KEYS:
            continue
        try:
            times[version] = config.parse_timestamp(value)
        except (AttributeError, TypeError, ValueError):
            log.warning("Ignoring unreadable publish time for version %r: %r", version, value)
    return times


def choose_version(packument, t0):
    """Return (version, versions passed over for lacking a manifest).

    Highest stable version before t0, else highest version before t0. A manifest is needed for
    the required dependency counts, so versions unpublished later cannot be chosen.
    """
    if "unpublished" in (packument.get("time") or {}):
        raise SkipPackage("fully unpublished")
    before_t0 = {v: published for v, published in version_times(packument).items() if published < t0}
    if not before_t0:
        raise SkipPackage("no version published before t0")

    parsed = {v: parse_version(v) for v in before_t0}
    candidates = [v for v, value in parsed.items() if value is not None]
    if not candidates:
        raise SkipPackage(f"no strict semver version before t0: {sorted(before_t0)[:5]}")
    manifests = packument.get("versions") or {}

    def highest(versions):
        return max(versions, key=lambda v: (parsed[v], before_t0[v], v))

    stable = [v for v in candidates if parsed[v].prerelease is None]
    pool = stable or candidates
    usable = [v for v in stable if v in manifests] or [v for v in candidates if v in manifests]
    if not usable:
        raise SkipPackage("no version before t0 still has a manifest")
    chosen = highest(usable)
    passed_over = sorted(
        v for v in pool if v not in manifests and (parsed[v], before_t0[v], v) > (parsed[chosen], before_t0[chosen], chosen)
    )
    return chosen, passed_over


def build_purl(name, version):
    """Build pkg:npm/name@version, percent-encoding each part. The exact npm name is kept."""
    encoded_name = "/".join(quote(part, safe="") for part in name.split("/"))
    return f"pkg:npm/{encoded_name}@{quote(version, safe='')}"


def declared_names(value, field=""):
    """Return the distinct names in a dependency field. Missing means none."""
    if value is None:
        return set()
    if isinstance(value, dict):
        return set(value)
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    log.warning("Unexpected %s value treated as empty: %r", field, value)
    return set()


def dependency_counts(manifest):
    dependencies = declared_names(manifest.get("dependencies"), "dependencies")
    peers = declared_names(manifest.get("peerDependencies"), "peerDependencies")
    peer_meta = manifest.get("peerDependenciesMeta")
    if isinstance(peer_meta, dict):
        peers |= {name for name, meta in peer_meta.items() if isinstance(meta, dict) and meta.get("optional") is True}
    bundled = set()
    for field in ("bundleDependencies", "bundledDependencies"):
        value = manifest.get(field)
        if value is True or value == "true":
            bundled |= dependencies
        elif value is not False:
            bundled |= declared_names(value, field)
    return {
        "dependency_count": len(dependencies),
        "optional_dependency_count": len(declared_names(manifest.get("optionalDependencies"), "optionalDependencies")),
        "peer_dependency_count": len(peers),
        "dev_dependency_count": len(declared_names(manifest.get("devDependencies"), "devDependencies")),
        "bundled_dependency_count": len(bundled),
    }


def non_negative_int(value, field):
    """Return a size from dist, or None if it is missing or unusable."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    log.warning("Unusable %s stored as NULL: %r", field, value)
    return None


def stable_release_counts(packument, t0):
    """Count distinct stable versions per feature month, from "time" so unpublished ones count."""
    windows = config.feature_months(t0)
    counts = {column: 0 for column in RELEASE_COLUMNS}
    for version, published in version_times(packument).items():
        parsed = parse_version(version)
        if parsed is None:
            log.warning("Not strict semver, so not counted as a stable release: %r", version)
            continue
        if parsed.prerelease is not None:
            continue
        for month, (start, end) in windows.items():
            if start <= published < end:
                counts[f"package_stable_release_count_M{month}"] += 1
                break
    return counts


def build_row(name, packument, t0):
    """Return (feature row, versions passed over). Raises SkipPackage with the reason."""
    version, passed_over = choose_version(packument, t0)
    times = packument["time"]
    if not times.get("created"):
        raise SkipPackage("time.created is missing")
    manifest = packument["versions"][version]
    dist = manifest.get("dist") or {}
    row = {
        "package_name": name,
        "prediction_date": config.format_timestamp(t0),
        "chosen_version": version,
        "purl": build_purl(name, version),
        "package_created_at": config.format_timestamp(config.parse_timestamp(times["created"])),
        "published_at": config.format_timestamp(config.parse_timestamp(times[version])),
        "unpacked_size": non_negative_int(dist.get("unpackedSize"), "unpackedSize"),
        "file_count": non_negative_int(dist.get("fileCount"), "fileCount"),
        **dependency_counts(manifest),
        **stable_release_counts(packument, t0),
    }
    return row, passed_over


def fetch_packument(name):
    """Fetch the full packument; the abbreviated format has no "time" field."""
    url = f"{config.NPM_REGISTRY_URL}/{quote(name, safe='@')}"
    return get_json(url, config.PACKUMENTS_CACHE_DIR, {"Accept": "application/json"}, compress=True)


def collect(names, t0_values):
    rows, failures, latest = [], [], {}
    for index, name in enumerate(names, 1):
        status, packument = fetch_packument(name)
        if status != 200 or not isinstance(packument, dict):
            reason = f"registry returned HTTP {status}"
            for t0 in t0_values:
                failures.append((name, config.format_timestamp(t0), reason))
                log.warning("SKIP %s: %s", name, reason)
            continue
        latest[name] = (packument.get("dist-tags") or {}).get("latest")
        for t0 in t0_values:
            try:
                row, passed_over = build_row(name, packument, t0)
            except SkipPackage as error:
                failures.append((name, config.format_timestamp(t0), str(error)))
                log.warning("SKIP %s at %s: %s", name, config.format_timestamp(t0), error)
                continue
            if passed_over:
                log.info("%s: chose %s, passing over %s (no manifest)", name, row["chosen_version"], passed_over)
            rows.append(row)
        print(f"  {index}/{len(names)} {name}: {len(packument.get('versions') or {})} manifests")
    return rows, failures, latest


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(rows, failures, latest):
    print(f"\n=== Report: {len(rows)} rows, {len(failures)} failures ===")
    print("Chosen version vs today's latest:")
    for row in rows:
        today = latest.get(row["package_name"])
        note = "same" if row["chosen_version"] == today else "differs"
        print(f"  {row['package_name']}: chosen {row['chosen_version']} ({row['published_at']}), latest {today} [{note}]")
    print("Failures:")
    for failure in failures:
        print(f"  {failure}")
    print("NULL counts:", {column: sum(row[column] is None for row in rows) for column in ("unpacked_size", "file_count")})
    scoped = [row for row in rows if row["package_name"].startswith("@")][:1]
    unscoped = [row for row in rows if not row["package_name"].startswith("@")][:2]
    print("Example rows:")
    for row in scoped + unscoped:
        print(f"  {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=config.PACKAGES_TEST_CSV)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "npm_meta.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    with open(args.packages, newline="", encoding="utf-8") as handle:
        names = [row["package_name"] for row in csv.DictReader(handle)]
    t0_values = [config.parse_timestamp(value) for value in config.PREDICTION_DATES]
    print(f"{len(names)} packages from {args.packages}, prediction dates {config.PREDICTION_DATES}")
    rows, failures, latest = collect(names, t0_values)
    write_rows(config.NPM_META_CSV, rows)
    report(rows, failures, latest)


if __name__ == "__main__":
    main()
