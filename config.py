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
TEST_SAMPLE_SIZE = 20
TARGET_PACKAGE_COUNT = 1000

# --- Vulnerabilities ----------------------------------------------------------
SEVERITIES = ["low", "moderate", "high", "critical", "unknown"]
ADVISORY_DB_URL = "https://github.com/github/advisory-database"
ADVISORY_SUBDIR = "advisories/github-reviewed"

# --- External services --------------------------------------------------------
NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/range"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
# Name of the environment variable only. The token itself is never stored in a file.
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schema" / "schema.sql"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ADVISORY_DB_DIR = CACHE_DIR / "advisory-database"
LOG_DIR = ROOT / "logs"
PACKAGES_CSV = DATA_DIR / "packages.csv"
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
