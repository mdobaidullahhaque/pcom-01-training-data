"""Compute monthly issue and pull request series from GitHub's GraphQL API.

The repository comes from the chosen version's manifest, so it reflects the package before t0.
Items are fetched oldest first and fetching stops at t0, so later history costs nothing. Raw
results are cached per page in data/cache/github/, so a stopped repo resumes where it was.

Known limitation: issues deleted, or transferred to other repos, after t0 are invisible today.
Issues-enabled is today's repo setting, and monorepo packages share their whole repo's series.

Usage: python -u -m src.github_activity [--packages data/packages_test.csv] [--allow-large o/r ...]
"""
import argparse
import csv
import gzip
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import config
from src.npm_meta import fetch_packument
from src.select_packages import read_csv

log = logging.getLogger("github_activity")

MONTHS = range(config.FEATURE_MONTHS)
ISSUE_FAMILIES = ["issues_opened", "issues_closed", "issues_reopened", "issues_open_at_month_end"]
PR_FAMILIES = [
    "pull_requests_opened", "pull_requests_closed_unmerged", "pull_requests_merged",
    "pull_requests_reopened", "pull_requests_open_at_month_end",
]
ISSUE_COLUMNS = [f"{family}_M{month}" for family in ISSUE_FAMILIES for month in MONTHS]
PR_COLUMNS = [f"{family}_M{month}" for family in PR_FAMILIES for month in MONTHS]
SERIES_COLUMNS = ISSUE_COLUMNS + PR_COLUMNS
COLUMNS = ["package_name", "prediction_date", "github_repo", *SERIES_COLUMNS]
GRAPHQL_RETRIES = 5
MERGE_CLOSE_SECONDS = 60
NOT_REPO_PATHS = {"orgs", "users", "sponsors", "settings", "marketplace"}

ISSUE_EVENT_FIELDS = """
  pageInfo { hasNextPage endCursor }
  nodes {
    __typename
    ... on ClosedEvent { createdAt }
    ... on ReopenedEvent { createdAt }
  }
"""
PR_EVENT_FIELDS = """
  pageInfo { hasNextPage endCursor }
  nodes {
    __typename
    ... on ClosedEvent { createdAt }
    ... on ReopenedEvent { createdAt }
    ... on MergedEvent { createdAt }
  }
"""
RATE_FIELDS = "rateLimit { cost remaining resetAt }"
ITEM_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  %(rate)s
  repository(owner: $owner, name: $name) {
    connection: %(connection)s(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id createdAt closedAt %(extra)s
        timelineItems(first: 20, itemTypes: [%(types)s]) { %(events)s }
      }
    }
  }
}
"""
TIMELINE_QUERY = """
query($id: ID!, $after: String) {
  %(rate)s
  node(id: $id) {
    ... on Issue { issueEvents: timelineItems(first: 100, after: $after, itemTypes: [CLOSED_EVENT, REOPENED_EVENT]) { %(issue_events)s } }
    ... on PullRequest { prEvents: timelineItems(first: 100, after: $after, itemTypes: [CLOSED_EVENT, REOPENED_EVENT, MERGED_EVENT]) { %(pr_events)s } }
  }
}
""" % {"rate": RATE_FIELDS, "issue_events": ISSUE_EVENT_FIELDS, "pr_events": PR_EVENT_FIELDS}
RESOLVE_QUERY = """
query($owner: String!, $name: String!) {
  %(rate)s
  repository(owner: $owner, name: $name) {
    nameWithOwner hasIssuesEnabled
    issues { totalCount }
    pullRequests { totalCount }
  }
}
""" % {"rate": RATE_FIELDS}
COUNT_QUERY = """
query($issues: String!, $prs: String!) {
  %(rate)s
  issuesBefore: search(query: $issues, type: ISSUE, first: 0) { issueCount }
  prsBefore: search(query: $prs, type: ISSUE, first: 0) { issueCount }
}
""" % {"rate": RATE_FIELDS}
KINDS = {
    "issues": ITEM_QUERY % {
        "rate": RATE_FIELDS, "connection": "issues", "extra": "",
        "types": "CLOSED_EVENT, REOPENED_EVENT", "events": ISSUE_EVENT_FIELDS,
    },
    "prs": ITEM_QUERY % {
        "rate": RATE_FIELDS, "connection": "pullRequests", "extra": "mergedAt isDraft",
        "types": "CLOSED_EVENT, REOPENED_EVENT, MERGED_EVENT", "events": PR_EVENT_FIELDS,
    },
}


class RepoTooLong(Exception):
    """The repo passed the time limit; its finished pages stay cached for resuming."""


class TransientFailure(Exception):
    """GitHub kept failing a query that may succeed if it asks for less."""


class InvalidQuery(Exception):
    """GitHub rejected the query itself, so retrying cannot help."""


# --- Repository URLs ------------------------------------------------------------------------
def parse_repository(field):
    """Return ("owner/name", None) for a GitHub repo, or (None, reason)."""
    if isinstance(field, dict):
        field = field.get("url")
    if isinstance(field, list):
        field = field[0] if field else None
        return parse_repository(field)
    if not field or not isinstance(field, str):
        return None, "no repository field"
    text = field.strip().split("#")[0].split("?")[0].strip()
    match = re.fullmatch(r"github:([^/\s:]+)/([^/\s]+?)/?", text)
    if match:
        return clean_slug(match[1], match[2], field)
    if re.fullmatch(r"(gitlab|bitbucket|gist):.*", text):
        return None, f"not GitHub: {field}"
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+?)/?", text)
    if match:
        return clean_slug(match[1], match[2], field)
    match = re.fullmatch(
        r"(?:[a-z+]+://)?(?:[^@/\s]+@)?(?:www\.)?github\.com(?::\d+)?[/:]+([^/\s:]+)/([^/\s]+)(?:/.*)?",
        text, re.IGNORECASE,
    )
    if match:
        return clean_slug(match[1], match[2], field)
    return None, f"not GitHub: {field}"


def clean_slug(owner, name, field):
    name = re.sub(r"\.git$", "", name, flags=re.IGNORECASE).rstrip("/")
    valid = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", owner) and re.fullmatch(r"[A-Za-z0-9._-]+", name)
    if not valid or owner.lower() in NOT_REPO_PATHS:
        return None, f"not a usable GitHub repository: {field}"
    return f"{owner}/{name}", None


# --- GraphQL --------------------------------------------------------------------------------
class GitHubClient:
    """GraphQL client that waits on rate limits. The token is only ever put in a request header."""

    def __init__(self, token, sleep=time.sleep):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"bearer {token}"
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.sleep = sleep
        self.points = 0
        self.remaining = None

    def wait_until(self, reset_at, why):
        seconds = max(0, (config.parse_timestamp(reset_at) - datetime.now(timezone.utc)).total_seconds()) + 5
        print(f"  waiting {seconds:.0f} s until {reset_at}: {why}")
        self.sleep(seconds)

    def query(self, text, variables):
        """Return the data block. Waits on limits; raises TransientFailure if GitHub keeps failing."""
        failure = ""
        for attempt in range(GRAPHQL_RETRIES):
            try:
                response = self.session.post(
                    config.GITHUB_GRAPHQL_URL, json={"query": text, "variables": variables}, timeout=120
                )
            except requests.RequestException as error:
                failure = type(error).__name__
                self.sleep(2 ** (attempt + 2))
                continue
            status = response.status_code
            if status == 401:
                raise RuntimeError("GitHub rejected the token (HTTP 401)")
            if status in (403, 429):
                wait = float(response.headers.get("Retry-After") or 60)
                print(f"  HTTP {status}: secondary rate limit, waiting {wait:.0f} s")
                self.sleep(wait)
                failure = f"HTTP {status}"
                continue
            if status != 200:
                failure = f"HTTP {status}"
                self.sleep(2 ** (attempt + 2))
                continue
            body = response.json()
            errors = body.get("errors") or []
            limit = (body.get("data") or {}).get("rateLimit")
            if limit:
                self.points += limit["cost"]
                self.remaining = limit["remaining"]
            if any(error.get("type") == "RATE_LIMITED" for error in errors):
                self.wait_until(limit["resetAt"] if limit else self.next_hour(), "rate limited")
                failure = "RATE_LIMITED"
                continue
            serious = [error for error in errors if error.get("type") != "NOT_FOUND"]
            if serious:
                failure = str(serious[0].get("message"))[:200]
                if (serious[0].get("extensions") or {}).get("code"):
                    raise InvalidQuery(failure)
                log.warning("GraphQL error, attempt %d: %s", attempt + 1, failure)
                self.sleep(2 ** (attempt + 2))
                continue
            if limit and limit["remaining"] < config.GITHUB_MIN_POINTS:
                self.wait_until(limit["resetAt"], f"{limit['remaining']} points left")
            return body["data"]
        raise TransientFailure(failure)

    @staticmethod
    def next_hour():
        return config.format_timestamp(datetime.now(timezone.utc) + timedelta(hours=1))


# --- Cache ----------------------------------------------------------------------------------
def repo_dir(slug):
    return config.GITHUB_CACHE_DIR / slug.replace("/", "__")


def read_json(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
    os.replace(temporary, path)


def resolve_repo(client, slug, cutoff):
    """Return the cached {"repo", "issues_enabled", counts} or {"repo": None, "reason"}.

    The repository lookup follows renames but search does not, so counts use the current name.
    """
    path = config.GITHUB_CACHE_DIR / "resolved" / f"{slug.replace('/', '__')}.json"
    if path.exists():
        cached = read_json(path)
        if not cached["repo"] or "counted_as" in cached:
            return cached
    owner, name = slug.split("/")
    repository = client.query(RESOLVE_QUERY, {"owner": owner, "name": name}).get("repository")
    if not repository:
        result = {"repo": None, "reason": f"deleted, renamed away or private: {slug}"}
    else:
        current = repository["nameWithOwner"]
        stamp = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        counts = client.query(COUNT_QUERY, {
            "issues": f"repo:{current} is:issue created:<{stamp}",
            "prs": f"repo:{current} is:pr created:<{stamp}",
        })
        result = {
            "repo": current,
            "counted_as": current,
            "issues_enabled": repository["hasIssuesEnabled"],
            "issues_total_today": repository["issues"]["totalCount"],
            "prs_total_today": repository["pullRequests"]["totalCount"],
            "issues_before": counts["issuesBefore"]["issueCount"],
            "prs_before": counts["prsBefore"]["issueCount"],
        }
    write_json(path, result)
    return result


# --- Fetching -------------------------------------------------------------------------------
def compact_item(node):
    return {
        "createdAt": node["createdAt"],
        "closedAt": node.get("closedAt"),
        "mergedAt": node.get("mergedAt"),
        "isDraft": node.get("isDraft"),
        "events": [[event["__typename"], event["createdAt"]] for event in node["timelineItems"]["nodes"]],
    }


def complete_timeline(client, node):
    """Fetch the remaining timeline pages for an item with an unusually long timeline."""
    info = node["timelineItems"]["pageInfo"]
    while info["hasNextPage"]:
        data = client.query(TIMELINE_QUERY, {"id": node["id"], "after": info["endCursor"]})
        extra = data["node"].get("issueEvents") or data["node"]["prEvents"]
        node["timelineItems"]["nodes"].extend(extra["nodes"])
        info = extra["pageInfo"]


def fetch_kind(client, slug, kind, cutoff, stats, deadline=None):
    """Return all items of a kind created before the cutoff, resuming from cached pages."""
    folder = repo_dir(slug)
    owner, name = slug.split("/")
    pages = sorted(folder.glob(f"{kind}_*.json.gz"))
    items, cursor, finished = [], None, False
    for path in pages:
        page = read_json(path)
        items += page["items"]
        cursor, finished = page["cursor"], page["done"]
    first = config.GITHUB_PAGE_SIZE
    number = len(pages)
    while not finished:
        if deadline and time.time() > deadline:
            raise RepoTooLong(f"{slug} {kind}: passed the time limit after {number} pages")
        started, points_before = time.time(), client.points
        try:
            data = client.query(KINDS[kind], {"owner": owner, "name": name, "first": first, "after": cursor})
        except TransientFailure as error:
            if first <= 10:
                raise
            first //= 2
            log.warning("%s %s: asking for %d items per page after: %s", slug, kind, first, error)
            continue
        connection = data["repository"]["connection"]
        kept = []
        for node in connection["nodes"]:
            if config.parse_timestamp(node["createdAt"]) >= cutoff:
                finished = True
                break
            complete_timeline(client, node)
            kept.append(compact_item(node))
        finished = finished or not connection["pageInfo"]["hasNextPage"]
        cursor = connection["pageInfo"]["endCursor"]
        number += 1
        write_json(folder / f"{kind}_{number:06d}.json.gz", {"cursor": cursor, "done": finished, "items": kept})
        items += kept
        stats["pages"] += 1
        stats["points"] += client.points - points_before
        stats["seconds"] += time.time() - started
        print(f"  {slug} {kind}: page {number}, {len(items)} items, {client.remaining} points left")
    return items


def fetch_repo(client, slug, issues_enabled, cutoff, deadline=None):
    """Return (issues or None, prs, stats). Stats add up over resumed runs."""
    meta_path = repo_dir(slug) / "meta.json"
    stats = read_json(meta_path) if meta_path.exists() else {"pages": 0, "points": 0, "seconds": 0.0}
    try:
        issues = fetch_kind(client, slug, "issues", cutoff, stats, deadline) if issues_enabled else None
        prs = fetch_kind(client, slug, "prs", cutoff, stats, deadline)
    finally:
        write_json(meta_path, stats)
    return issues, prs, stats


# --- Series ---------------------------------------------------------------------------------
def timed_events(item, kinds):
    """Return (time, kind) in time order; timeline order breaks ties."""
    events = [
        (config.parse_timestamp(event_time), kind)
        for kind, event_time in item["events"] if kind in kinds
    ]
    return sorted(events, key=lambda event: event[0])


def with_closed_at_fallback(item, events):
    """An item closed today with no close event (rare, old data) still closed at closedAt."""
    if item.get("closedAt") and not any(kind in ("ClosedEvent", "MergedEvent") for _, kind in events):
        events = sorted([*events, (config.parse_timestamp(item["closedAt"]), "ClosedEvent")], key=lambda e: e[0])
    return events


def open_at(created, events, moment):
    """Whether an item is open just before a moment, replaying its events in order."""
    if created >= moment:
        return False
    is_open = True
    for event_time, kind in events:
        if event_time >= moment:
            break
        is_open = kind == "ReopenedEvent"
    return is_open


def add_count(row, family, windows, moment):
    for month, (start, end) in windows.items():
        if start <= moment < end:
            row[f"{family}_M{month}"] += 1
            return


def issue_series(items, t0):
    """Return the issue columns; None means the repo has issues disabled, so all are NULL."""
    if items is None:
        return {column: None for column in ISSUE_COLUMNS}
    windows = config.feature_months(t0)
    row = {column: 0 for column in ISSUE_COLUMNS}
    for item in items:
        created = config.parse_timestamp(item["createdAt"])
        events = with_closed_at_fallback(item, timed_events(item, {"ClosedEvent", "ReopenedEvent"}))
        add_count(row, "issues_opened", windows, created)
        for event_time, kind in events:
            family = "issues_closed" if kind == "ClosedEvent" else "issues_reopened"
            add_count(row, family, windows, event_time)
        for month, (_, end) in windows.items():
            row[f"issues_open_at_month_end_M{month}"] += open_at(created, events, end)
    return row


def is_merge_closure(close_time, events, merged_at):
    """GitHub records a close event with each merge; only that one is not an unmerged closure."""
    if merged_at is None or abs((merged_at - close_time).total_seconds()) > MERGE_CLOSE_SECONDS:
        return False
    return not any(kind == "ReopenedEvent" and close_time < event_time < merged_at for event_time, kind in events)


def pr_series(items, t0):
    windows = config.feature_months(t0)
    row = {column: 0 for column in PR_COLUMNS}
    for item in items:
        created = config.parse_timestamp(item["createdAt"])
        events = timed_events(item, {"ClosedEvent", "ReopenedEvent", "MergedEvent"})
        merge_times = [event_time for event_time, kind in events if kind == "MergedEvent"]
        if not merge_times and item.get("mergedAt"):
            merge_times = [config.parse_timestamp(item["mergedAt"])]
            events = sorted([*events, (merge_times[0], "MergedEvent")], key=lambda e: e[0])
        events = with_closed_at_fallback(item, events)
        merged_at = merge_times[0] if merge_times else None
        add_count(row, "pull_requests_opened", windows, created)
        for event_time, kind in events:
            if kind == "MergedEvent":
                add_count(row, "pull_requests_merged", windows, event_time)
            elif kind == "ReopenedEvent":
                add_count(row, "pull_requests_reopened", windows, event_time)
            elif not is_merge_closure(event_time, events, merged_at):
                add_count(row, "pull_requests_closed_unmerged", windows, event_time)
        for month, (_, end) in windows.items():
            row[f"pull_requests_open_at_month_end_M{month}"] += open_at(created, events, end)
    return row


def null_row(name, t0, repo=None):
    return {"package_name": name, "prediction_date": config.format_timestamp(t0), "github_repo": repo,
            **{column: None for column in SERIES_COLUMNS}}


# --- Run ------------------------------------------------------------------------------------
def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def estimated_seconds(info, seconds_per_page):
    """Estimate fetch time. A zero search count falls back to today's total, the safe upper bound."""
    issues = info["issues_before"] or info.get("issues_total_today", 0)
    prs = info["prs_before"] or info.get("prs_total_today", 0)
    items = (issues if info["issues_enabled"] else 0) + prs
    return (items / config.GITHUB_PAGE_SIZE + 2) * seconds_per_page


def repository_field(name, version):
    """Return the repository field from the chosen version's manifest, not today's."""
    status, packument = fetch_packument(name)
    if status != 200:
        return None
    return ((packument.get("versions") or {}).get(version) or {}).get("repository")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=config.PACKAGES_TEST_CSV)
    parser.add_argument("--allow-large", nargs="*", default=[], help="repos approved to pass the time limit")
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "github_activity.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    log.info("Known limitation: issues deleted or transferred after t0 are invisible today.")
    token = os.environ.get(config.GITHUB_TOKEN_ENV)
    if not token:
        sys.exit(f"{config.GITHUB_TOKEN_ENV} is not set. Create a read-only token and set it first.")
    client = GitHubClient(token)

    names = {row["package_name"] for row in read_csv(args.packages)}
    meta_rows = [row for row in read_csv(config.NPM_META_CSV) if row["package_name"] in names]
    t0_values = sorted({row["prediction_date"] for row in meta_rows})
    cutoff = max(config.parse_timestamp(value) for value in t0_values)

    mapping = {}
    for row in meta_rows:
        slug, reason = parse_repository(repository_field(row["package_name"], row["chosen_version"]))
        mapping[row["package_name"], row["prediction_date"]] = {"slug": slug, "reason": reason, "resolved": None}
    infos = {}
    for entry in mapping.values():
        if entry["slug"]:
            infos.setdefault(entry["slug"], resolve_repo(client, entry["slug"], cutoff))
            entry["resolved"] = infos[entry["slug"]]
            if not entry["resolved"]["repo"]:
                entry["reason"] = entry["resolved"]["reason"]

    repos = {info["repo"]: info for info in infos.values() if info["repo"]}
    fetched, deferred, measures = {}, {}, {}
    total_pages, total_seconds = 0, 0.0
    allowed = {repo.lower() for repo in args.allow_large}
    for repo, info in sorted(repos.items(), key=lambda item: estimated_seconds(item[1], 1)):
        rate = total_seconds / total_pages if total_pages >= 20 else config.GITHUB_DEFAULT_SECONDS_PER_PAGE
        estimate = estimated_seconds(info, rate)
        limit_seconds = config.GITHUB_MAX_REPO_MINUTES * 60
        if estimate > limit_seconds and repo.lower() not in allowed:
            deferred[repo] = {"estimate_minutes": round(estimate / 60), **info}
            print(f"HELD BACK {repo}: about {estimate / 60:.0f} min at {rate:.1f} s per page")
            continue
        print(f"{repo}: {info['issues_before']} issues, {info['prs_before']} PRs before t0, about {estimate / 60:.1f} min")
        deadline = None if repo.lower() in allowed else time.time() + limit_seconds
        try:
            issues, prs, stats = fetch_repo(client, repo, info["issues_enabled"], cutoff, deadline)
        except RepoTooLong as error:
            deferred[repo] = {"estimate_minutes": "passed the limit", **info}
            print(f"HELD BACK {error}")
            continue
        if not info["issues_enabled"]:
            log.info("%s has issues disabled: issue columns are NULL, PR columns computed", repo)
        fetched[repo] = (issues, prs)
        measures[repo] = {"items": len(issues or []) + len(prs), **stats}
        total_pages += stats["pages"]
        total_seconds += stats["seconds"]

    rows, held = [], []
    series_cache = {}
    for row in meta_rows:
        name, prediction_date = row["package_name"], row["prediction_date"]
        entry = mapping[name, prediction_date]
        t0 = config.parse_timestamp(prediction_date)
        repo = (entry["resolved"] or {}).get("repo")
        if repo in deferred:
            held.append((name, repo))
            continue
        output = null_row(name, t0, repo)
        if repo:
            key = (repo, prediction_date)
            if key not in series_cache:
                issues, prs = fetched[repo]
                series_cache[key] = {**issue_series(issues, t0), **pr_series(prs, t0)}
            output.update(series_cache[key])
        else:
            log.warning("All GitHub columns NULL for %s: %s", name, entry["reason"])
        rows.append(output)
    write_rows(config.GITHUB_ACTIVITY_CSV, rows)

    print(f"\n=== Report: {len(rows)} rows written, {len(held)} packages held back ===")
    print("Repo mapping:")
    for row in meta_rows:
        entry = mapping[row["package_name"], row["prediction_date"]]
        repo = (entry["resolved"] or {}).get("repo")
        moved = " (renamed from " + entry["slug"] + ")" if repo and repo.lower() != entry["slug"].lower() else ""
        print(f"  {row['package_name']}: {repo or 'NULL'}{moved}{' -- ' + entry['reason'] if entry['reason'] else ''}")
    print("Measured per repo:")
    for repo, measure in measures.items():
        print(f"  {repo}: {measure}")
    print("Held back:")
    for repo, values in deferred.items():
        print(f"  {repo}: {values}")
    print(f"Issues disabled: {[repo for repo, info in repos.items() if not info['issues_enabled']]}")
    print(f"Total points used this run: {client.points}; points left: {client.remaining}")
    print(f"Measured seconds per page: {total_seconds / total_pages if total_pages else 0:.2f}")
    print("All-NULL rows:", sum(row[SERIES_COLUMNS[-1]] is None for row in rows))


if __name__ == "__main__":
    main()
