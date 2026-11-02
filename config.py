"""Central configuration for the PCOM-01 training dataset. Standard library only."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Prediction setup ---------------------------------------------------------
# Add more dates here later; the pipeline produces one row per package per t0.
PREDICTION_DATES = ["2025-08-01T00:00:00.000Z"]
FEATURE_MONTHS = 12
TARGET_DAYS = 90

# True: M0 = most recent full month before t0 (t0 = 2025-08-01 gives M0 = July 2025,
# M11 = August 2024). False flips the index (M0 = August 2024, M11 = July 2025).
M0_IS_MOST_RECENT = True

# --- Package selection --------------------------------------------------------
RANDOM_SEED = 42
SAMPLE_PER_STRATUM = 500
POPULAR_MIN_DOWNLOADS = 1_000_000  # feature-year downloads, at least
RANDOM_MIN_DOWNLOADS = 1_000  # feature-year downloads, at least
RANDOM_MAX_DOWNLOADS = 1_000_000  # feature-year downloads, under
TEST_LIST_COUNTS = {"had_ghsa": 8, "popular_no_ghsa": 6, "random_no_ghsa": 6}
# Name lists published before t0, so that choosing candidates uses no later information.
POPULAR_SOURCE = {
    "name": "npm-high-impact",
    "version": "1.11.0",
    "published_at": "2025-06-05T06:37:15.400Z",
    "shasum": "12c22a769f9f452d9750bd71cf065f3271f4873e",
}
RANDOM_SOURCE = {
    "name": "all-the-package-names",
    "version": "2.0.2159",
    "published_at": "2025-07-31T00:10:49.958Z",
    "shasum": "af5e529462f6aff895e53e4325d1be2edd4d5b09",
}

# --- Vulnerabilities ----------------------------------------------------------
SEVERITIES = ["low", "moderate", "high", "critical", "unknown"]
ADVISORY_DB_URL = "https://github.com/github/advisory-database"
ADVISORY_SUBDIR = "advisories/github-reviewed"
ADVISORY_COMMITS_API = "https://api.github.com/repos/github/advisory-database/commits"
SNAPSHOT_SIZE_LIMIT_BYTES = 1024**3  # 1 GB per snapshot
GITHUB_ADVISORIES_API = "https://api.github.com/advisories"
# Windows Defender quarantines these advisory files on this machine when they are read, even
# with a folder exclusion. GitHub's API lists all 7 as non-npm (composer or pip), so losing them
# cannot change npm counts. src/advisories.py never opens them, logs them on every run, and
# fails if the API reports npm for one. Any other missing file is still a hard failure.
ALLOWED_BLOCKED_ADVISORIES = frozenset({
    "GHSA-88xg-v53p-fpvf",
    "GHSA-v22v-xwh7-2vrm",
    "GHSA-3pgc-xqg9-cfr6",
    "GHSA-vf3q-frmr-vrr9",
    "GHSA-vp2f-cqqp-478j",
    "GHSA-wx9m-wx4f-4cmg",
    "GHSA-gxmj-r5rf-ggwq",
})

# --- External services --------------------------------------------------------
NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/range"
NPM_DOWNLOADS_POINT_URL = "https://api.npmjs.org/downloads/point"
NPM_BULK_MAX_NAMES = 128
USER_AGENT = "pcom-01-training-data (university coursework)"
REQUEST_DELAY_SECONDS = 1.0  # 0.25 s drew HTTP 429 from the npm downloads API
MAX_RETRIES = 6
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
# Name of the environment variable only. The token itself is never stored in a file.
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_PAGE_SIZE = 100
GITHUB_MIN_POINTS = 100  # wait for the rate-limit reset below this many remaining points
GITHUB_MAX_REPO_MINUTES = 30  # larger repos are held back until the user approves them
GITHUB_DEFAULT_SECONDS_PER_PAGE = 3.0  # until a real rate has been measured in the run

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schema" / "schema.sql"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ADVISORY_SNAPSHOT_DIR = CACHE_DIR / "advisory-db"
ADVISORY_SNAPSHOTS_JSON = CACHE_DIR / "advisory_snapshots.json"
ADVISORIES_SQLITE = CACHE_DIR / "advisories.sqlite"
ALLOWED_BLOCKED_FILES_JSON = DATA_DIR / "allowed_blocked_files.json"  # committed, public info only
LOG_DIR = ROOT / "logs"
HTTP_CACHE_DIR = CACHE_DIR / "http"
SOURCES_DIR = CACHE_DIR / "sources"
PACKUMENTS_CACHE_DIR = CACHE_DIR / "packuments"
FEATURES_DIR = CACHE_DIR / "features"
NPM_META_CSV = FEATURES_DIR / "npm_meta.csv"
NPM_DOWNLOADS_CSV = FEATURES_DIR / "npm_downloads.csv"
GHSA_FEATURES_CSV = FEATURES_DIR / "ghsa_features.csv"
GITHUB_ACTIVITY_CSV = FEATURES_DIR / "github_activity.csv"
GITHUB_CACHE_DIR = CACHE_DIR / "github"
PACKAGES_CSV = DATA_DIR / "packages.csv"
PACKAGES_TEST_CSV = DATA_DIR / "packages_test.csv"
SELECTION_STATS_JSON = DATA_DIR / "packages_selection_stats.json"
OUTPUT_SQLITE = DATA_DIR / "pcom01_training.sqlite"
OUTPUT_CSV = DATA_DIR / "pcom01_training.csv"


# --- Date helpers -------------------------------------------------------------
def parse_timestamp(value):
    """Parse an RFC 3339 timestamp into an aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_timestamp(value):
    """Format as RFC 3339 UTC with exactly 3 fractional digits."""
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def feature_months(t0):
    """Return {month_index: (start, end)} with end exclusive, for the full months before t0."""
    year, month = t0.year, t0.month
    windows = []
    for _ in range(FEATURE_MONTHS):
        end = datetime(year, month, 1, tzinfo=timezone.utc)
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
        windows.append((datetime(year, month, 1, tzinfo=timezone.utc), end))
    if not M0_IS_MOST_RECENT:
        windows.reverse()
    return dict(enumerate(windows))


def target_window(t0):
    """Return the target window (t0, t0 + TARGET_DAYS), with the end exclusive."""
    return t0, t0 + timedelta(days=TARGET_DAYS)
