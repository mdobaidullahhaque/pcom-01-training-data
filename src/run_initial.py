"""Initial version: run npm metadata, downloads and GHSA features under a time budget.

Packages are shuffled with the seed across all strata, so the stratum mix stays fair wherever
the budget ends. Packages not reached in time are left out; cached answers make a rerun fast.

Usage: python -u -m src.run_initial [--budget-minutes 40]
"""
import argparse
import csv
import json
import logging
import random
import time
from collections import Counter

import config
from src import ghsa_features as gf
from src import npm_downloads as nd
from src import npm_meta as nm
from src.select_packages import read_csv

log = logging.getLogger("run_initial")
PROGRESS_EVERY = 50
INCLUDED_COLUMNS = ["package_name", "stratum", "source", "purl", "chosen_version", "is_security_placeholder"]


def shuffled_packages(rows, seed):
    order = list(rows)
    random.Random(seed).shuffle(order)
    return order


def collect_meta(packages, t0, deadline, fetch=nm.fetch_packument, clock=time.time, started=None):
    """Return (meta rows, failures, reached). Stops starting new packages at the deadline."""
    started = started if started is not None else clock()
    rows, failures, reached = [], [], 0
    for package in packages:
        if clock() >= deadline:
            break
        name = package["package_name"]
        reached += 1
        try:
            status, packument = fetch(name)
            if status != 200 or not isinstance(packument, dict):
                raise nm.SkipPackage(f"registry returned HTTP {status}")
            row, passed_over = nm.build_row(name, packument, t0)
            if passed_over:
                log.info("%s: chose %s, passing over %s (no manifest)", name, row["chosen_version"], passed_over)
            rows.append(row)
        except nm.SkipPackage as error:
            failures.append((name, str(error)))
            log.warning("SKIP %s: %s", name, error)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            failures.append((name, f"unusable metadata or request failure: {error!r}"))
            log.warning("SKIP %s: %r", name, error)
        if reached % PROGRESS_EVERY == 0:
            print(f"  metadata: {reached} reached, {len(rows)} ok, {len(failures)} skipped, "
                  f"{(clock() - started) / 60:.1f} min")
    return rows, failures, reached


def folder_megabytes(folder):
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file()) / 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-minutes", type=float, default=config.INITIAL_BUDGET_MINUTES)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(config.LOG_DIR / "run_initial.log", encoding="utf-8"), console],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    t0 = config.parse_timestamp(config.PREDICTION_DATES[0])
    started = time.time()
    package_rows = read_csv(config.PACKAGES_CSV)
    packages = shuffled_packages(package_rows, config.RANDOM_SEED)
    print(f"{len(packages)} packages shuffled with seed {config.RANDOM_SEED}; budget {args.budget_minutes} min")

    meta_rows, failures, reached = collect_meta(packages, t0, started + args.budget_minutes * 60, started=started)
    nm.write_rows(config.NPM_META_CSV, meta_rows)
    meta_minutes = (time.time() - started) / 60
    print(f"Metadata done in {meta_minutes:.1f} min: reached {reached}, ok {len(meta_rows)}, skipped {len(failures)}, "
          f"not reached {len(packages) - reached}; packument cache {folder_megabytes(config.PACKUMENTS_CACHE_DIR):.0f} MB")

    by_name = {row["package_name"]: row for row in package_rows}
    with open(config.PACKAGES_INCLUDED_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, INCLUDED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in meta_rows:
            source = by_name[row["package_name"]]
            writer.writerow({
                "package_name": row["package_name"], "stratum": source["stratum"], "source": source["source"],
                "purl": row["purl"], "chosen_version": row["chosen_version"],
                "is_security_placeholder": row["is_security_placeholder"],
            })

    # Downloads read the metadata as text, the way npm_downloads reads its CSV.
    stored = read_csv(config.NPM_META_CSV)
    download_rows, nulls, removed = [], [], {}
    for start in range(0, len(stored), PROGRESS_EVERY):
        rows, slice_nulls, slice_removed = nd.collect(stored[start:start + PROGRESS_EVERY])
        download_rows += rows
        nulls += slice_nulls
        removed.update(slice_removed)
        print(f"  downloads: {len(download_rows)}/{len(stored)}, {(time.time() - started) / 60:.1f} min")
    nd.validate(download_rows)
    nd.write_rows(config.NPM_DOWNLOADS_CSV, download_rows)
    checked, mismatches = nd.cross_check(download_rows, package_rows)
    print(f"Downloads done: {len(nulls)} packages with no data; cross-check {checked} checked, {len(mismatches)} mismatches")
    for mismatch in mismatches[:20]:
        log.warning("Download cross-check mismatch: %s", mismatch)
        print(f"  mismatch {mismatch}")

    names = [row["package_name"] for row in stored]
    features = gf.count_features(gf.load_active(config.ADVISORIES_SQLITE, gf.FEATURE_SNAPSHOT), t0)
    targets = gf.count_targets(gf.load_active(config.ADVISORIES_SQLITE, gf.TARGET_SNAPSHOT), t0)
    ghsa_rows = gf.build_rows(names, features, targets, t0)
    gf.validate(ghsa_rows)
    gf.write_rows(config.GHSA_FEATURES_CSV, ghsa_rows)
    differences = gf.compare_targets(ghsa_rows, gf.recount_targets_sql(config.ADVISORIES_SQLITE, names, t0))
    if differences:
        raise ValueError(f"Target recount differs: {differences[:10]}")
    print(f"GHSA done: {sum(gf.is_positive(row) for row in ghsa_rows)} positive packages, recount differences 0")

    reasons = Counter(reason.split(":")[0] for _, reason in failures)
    config.INITIAL_RUN_STATS_JSON.write_text(
        json.dumps(
            {
                "seed": config.RANDOM_SEED,
                "budget_minutes": args.budget_minutes,
                "metadata_minutes": round(meta_minutes, 1),
                "packages_total": len(packages),
                "packages_reached": reached,
                "packages_included": len(meta_rows),
                "packages_not_reached": len(packages) - reached,
                "skipped": len(failures),
                "skip_reasons": dict(reasons),
                "skipped_packages": [{"package_name": name, "reason": reason} for name, reason in failures],
                "packages_without_download_data": len(nulls),
                "download_cross_check": {"checked": checked, "mismatches": len(mismatches)},
                "packument_cache_megabytes": round(folder_megabytes(config.PACKUMENTS_CACHE_DIR)),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Skip reasons: {dict(reasons)}")
    print(f"Total {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
