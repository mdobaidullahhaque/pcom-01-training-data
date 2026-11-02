import pytest
import requests

import config
from src import github_activity as ga

T0 = config.parse_timestamp("2025-08-01T00:00:00.000Z")


def item(created, *events, closed_at=None, merged_at=None, draft=False):
    return {"createdAt": created, "closedAt": closed_at, "mergedAt": merged_at, "isDraft": draft,
            "events": [list(event) for event in events]}


CLOSED, REOPENED, MERGED = "ClosedEvent", "ReopenedEvent", "MergedEvent"


@pytest.mark.parametrize(
    "field, expected",
    [
        ("git+https://github.com/u/r.git", "u/r"),
        ("https://github.com/u/r", "u/r"),
        ("https://github.com/u/r/", "u/r"),
        ("http://www.github.com/u/r.git", "u/r"),
        ("git://github.com/u/r.git", "u/r"),
        ("ssh://git@github.com/u/r.git", "u/r"),
        ("git+ssh://git@github.com/u/r.git", "u/r"),
        ("git@github.com:u/r.git", "u/r"),
        ("github:u/r", "u/r"),
        ("u/r", "u/r"),
        ("git+https://github.com/u/r.git#main", "u/r"),
        ("https://github.com/u/r/tree/main/packages/x", "u/r"),
        ("https://GitHub.com/U/R.js.git", "U/R.js"),
        ({"type": "git", "url": "git+https://github.com/u/r.git", "directory": "packages/x"}, "u/r"),
        (["https://github.com/u/r"], "u/r"),
    ],
)
def test_repo_url_parsing(field, expected):
    assert ga.parse_repository(field) == (expected, None)


@pytest.mark.parametrize(
    "field, reason",
    [
        (None, "no repository field"),
        ("", "no repository field"),
        ({"type": "git"}, "no repository field"),
        ("https://gitlab.com/u/r.git", "not GitHub"),
        ("gitlab:u/r", "not GitHub"),
        ("bitbucket:u/r", "not GitHub"),
        ("git+https://bitbucket.org/u/r.git", "not GitHub"),
        ("https://github.com/orgs/u", "not a usable"),
        ("https://github.com/orgs/u/x", "not a usable"),
    ],
)
def test_repo_urls_that_are_not_usable(field, reason):
    slug, why = ga.parse_repository(field)
    assert slug is None
    assert reason in why


def test_month_end_open_with_close_and_reopen_inside_a_month():
    row = ga.issue_series(
        [item("2025-05-10T00:00:00Z", (CLOSED, "2025-07-05T00:00:00Z"), (REOPENED, "2025-07-20T00:00:00Z"))], T0
    )
    assert row["issues_opened_M2"] == 1
    assert row["issues_closed_M0"] == 1
    assert row["issues_reopened_M0"] == 1
    assert row["issues_open_at_month_end_M0"] == 1
    assert row["issues_open_at_month_end_M1"] == 1
    assert row["issues_open_at_month_end_M2"] == 1
    assert row["issues_open_at_month_end_M3"] == 0


def test_issue_closed_and_not_reopened_in_month_is_not_open_at_end():
    row = ga.issue_series([item("2025-05-10T00:00:00Z", (CLOSED, "2025-07-05T00:00:00Z"))], T0)
    assert row["issues_open_at_month_end_M0"] == 0
    assert row["issues_open_at_month_end_M1"] == 1


def test_issue_closed_exactly_at_a_month_boundary():
    row = ga.issue_series([item("2025-05-10T00:00:00Z", (CLOSED, "2025-07-01T00:00:00.000Z"))], T0)
    assert row["issues_closed_M0"] == 1
    assert row["issues_closed_M1"] == 0
    assert row["issues_open_at_month_end_M1"] == 1  # still open just before June ended
    assert row["issues_open_at_month_end_M0"] == 0


def test_issue_created_exactly_at_a_month_boundary():
    row = ga.issue_series([item("2025-07-01T00:00:00.000Z")], T0)
    assert row["issues_opened_M0"] == 1
    assert row["issues_opened_M1"] == 0
    assert row["issues_open_at_month_end_M1"] == 0
    assert row["issues_open_at_month_end_M0"] == 1


def test_item_from_before_the_window_counts_as_open_only():
    row = ga.issue_series([item("2019-01-01T00:00:00Z")], T0)
    assert sum(row[f"issues_opened_M{month}"] for month in ga.MONTHS) == 0
    assert all(row[f"issues_open_at_month_end_M{month}"] == 1 for month in ga.MONTHS)


def test_events_on_or_after_t0_are_ignored():
    row = ga.issue_series([item("2025-07-10T00:00:00Z", (CLOSED, "2025-08-01T00:00:00Z"))], T0)
    assert sum(row[f"issues_closed_M{month}"] for month in ga.MONTHS) == 0
    assert row["issues_open_at_month_end_M0"] == 1


def test_repeated_closures_after_reopening_all_count():
    row = ga.issue_series(
        [item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-02T00:00:00Z"), (REOPENED, "2025-07-03T00:00:00Z"), (CLOSED, "2025-07-04T00:00:00Z"))], T0
    )
    assert row["issues_closed_M0"] == 2
    assert row["issues_opened_M0"] == 1


def test_timeline_order_breaks_same_second_ties():
    row = ga.issue_series(
        [item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-02T00:00:00Z"), (REOPENED, "2025-07-02T00:00:00Z"))], T0
    )
    assert row["issues_open_at_month_end_M0"] == 1


def test_issue_closed_without_a_close_event_uses_closed_at():
    row = ga.issue_series([item("2025-05-01T00:00:00Z", closed_at="2025-06-10T00:00:00Z")], T0)
    assert row["issues_closed_M1"] == 1
    assert row["issues_open_at_month_end_M1"] == 0


def test_issues_disabled_gives_null_issue_columns_only():
    row = ga.issue_series(None, T0)
    assert len(row) == 48
    assert all(value is None for value in row.values())


def test_pr_closed_unmerged_then_reopened_and_merged():
    merged_time = "2025-07-20T00:00:00Z"
    row = ga.pr_series(
        [item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-05T00:00:00Z"), (REOPENED, "2025-07-10T00:00:00Z"),
              (MERGED, merged_time), (CLOSED, merged_time), merged_at=merged_time, closed_at=merged_time)], T0
    )
    assert row["pull_requests_opened_M0"] == 1
    assert row["pull_requests_closed_unmerged_M0"] == 1
    assert row["pull_requests_merged_M0"] == 1
    assert row["pull_requests_reopened_M0"] == 1
    assert row["pull_requests_open_at_month_end_M0"] == 0


def test_merge_close_event_a_few_seconds_apart_is_not_unmerged():
    row = ga.pr_series(
        [item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-20T00:00:03Z"), (MERGED, "2025-07-20T00:00:00Z"), merged_at="2025-07-20T00:00:00Z")], T0
    )
    assert row["pull_requests_closed_unmerged_M0"] == 0
    assert row["pull_requests_merged_M0"] == 1


def test_pr_closed_unmerged_stays_closed():
    row = ga.pr_series([item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-05T00:00:00Z"), closed_at="2025-07-05T00:00:00Z")], T0)
    assert row["pull_requests_closed_unmerged_M0"] == 1
    assert row["pull_requests_merged_M0"] == 0
    assert row["pull_requests_open_at_month_end_M0"] == 0


def test_pr_closed_before_t0_and_merged_after_t0_is_unmerged_before_t0():
    row = ga.pr_series(
        [item("2025-07-01T00:00:00Z", (CLOSED, "2025-07-05T00:00:00Z"), (REOPENED, "2025-08-10T00:00:00Z"),
              (MERGED, "2025-08-20T00:00:00Z"), (CLOSED, "2025-08-20T00:00:00Z"), merged_at="2025-08-20T00:00:00Z")], T0
    )
    assert row["pull_requests_closed_unmerged_M0"] == 1
    assert sum(row[f"pull_requests_merged_M{month}"] for month in ga.MONTHS) == 0
    assert row["pull_requests_open_at_month_end_M0"] == 0


def test_draft_counts_as_opened_and_open():
    row = ga.pr_series([item("2025-07-01T00:00:00Z", draft=True)], T0)
    assert row["pull_requests_opened_M0"] == 1
    assert row["pull_requests_open_at_month_end_M0"] == 1


def test_merged_at_without_a_merge_event_still_counts_once():
    row = ga.pr_series([item("2025-07-01T00:00:00Z", merged_at="2025-07-02T00:00:00Z", closed_at="2025-07-02T00:00:00Z")], T0)
    assert row["pull_requests_merged_M0"] == 1
    assert row["pull_requests_closed_unmerged_M0"] == 0
    assert row["pull_requests_open_at_month_end_M0"] == 0


def test_month_direction_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "M0_IS_MOST_RECENT", False)
    row = ga.issue_series([item("2025-07-10T00:00:00Z")], T0)
    assert row["issues_opened_M11"] == 1
    assert row["issues_opened_M0"] == 0


def test_column_counts():
    assert len(ga.SERIES_COLUMNS) == 108
    assert len(ga.null_row("a", T0)) == 111


def node(created, node_id="n", events=(), more=False):
    return {"id": node_id, "createdAt": created, "closedAt": None,
            "timelineItems": {"pageInfo": {"hasNextPage": more, "endCursor": "t1"},
                              "nodes": [{"__typename": kind, "createdAt": moment} for kind, moment in events]}}


class FakeClient:
    """Serves prepared answers; counts how many item queries were sent."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.points = 0
        self.remaining = 4000
        self.item_queries = 0
        self.seen_after = []

    def query(self, text, variables):
        if "$id" in text:
            return {"node": {"issueEvents": {"pageInfo": {"hasNextPage": False, "endCursor": None},
                                             "nodes": [{"__typename": CLOSED, "createdAt": "2025-07-09T00:00:00Z"}]}}}
        self.item_queries += 1
        self.seen_after.append(variables["after"])
        answer = self.pages.pop(0)
        if isinstance(answer, Exception):
            raise answer
        self.points += 2
        nodes, more, cursor = answer
        return {"repository": {"connection": {"pageInfo": {"hasNextPage": more, "endCursor": cursor}, "nodes": nodes}}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GITHUB_CACHE_DIR", tmp_path)
    return tmp_path


def new_stats():
    return {"pages": 0, "points": 0, "seconds": 0.0}


def test_fetch_stops_at_t0_and_rerun_uses_the_cache(cache):
    client = FakeClient([([node("2025-01-01T00:00:00Z")], True, "c1"),
                         ([node("2025-07-01T00:00:00Z"), node("2025-08-01T00:00:00Z")], True, "c2")])
    stats = new_stats()
    items = ga.fetch_kind(client, "u/r", "issues", T0, stats)
    assert [entry["createdAt"] for entry in items] == ["2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"]
    assert client.item_queries == 2
    assert stats["pages"] == 2 and stats["points"] == 4
    again = FakeClient([])
    assert ga.fetch_kind(again, "u/r", "issues", T0, new_stats()) == items
    assert again.item_queries == 0


def test_interrupted_fetch_resumes_from_the_last_cursor(cache):
    first = FakeClient([([node("2025-01-01T00:00:00Z")], True, "c1"), ga.TransientFailure("boom")])
    first_pages = first.pages
    with pytest.raises(ga.TransientFailure):
        # A failure at the smallest page size is final; earlier pages stay cached.
        original = config.GITHUB_PAGE_SIZE
        config.GITHUB_PAGE_SIZE = 10
        try:
            ga.fetch_kind(first, "u/r", "issues", T0, new_stats())
        finally:
            config.GITHUB_PAGE_SIZE = original
    assert first_pages == []
    second = FakeClient([([node("2025-02-01T00:00:00Z")], False, "c2")])
    items = ga.fetch_kind(second, "u/r", "issues", T0, new_stats())
    assert second.seen_after == ["c1"]
    assert len(items) == 2


def test_page_size_halves_when_github_keeps_failing(cache):
    client = FakeClient([ga.TransientFailure("502"), ([node("2025-01-01T00:00:00Z")], False, "c1")])
    assert len(ga.fetch_kind(client, "u/r", "issues", T0, new_stats())) == 1
    assert client.seen_after == [None, None]


def test_long_timeline_is_completed(cache):
    client = FakeClient([([node("2025-07-01T00:00:00Z", more=True)], False, "c1")])
    items = ga.fetch_kind(client, "u/r", "issues", T0, new_stats())
    assert items[0]["events"] == [[CLOSED, "2025-07-09T00:00:00Z"]]


def test_time_limit_holds_the_repo_back_and_keeps_pages(cache):
    client = FakeClient([])
    with pytest.raises(ga.RepoTooLong):
        ga.fetch_kind(client, "u/r", "issues", T0, new_stats(), deadline=1)


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self.body = body or {}
        self.headers = headers or {}

    def json(self):
        return self.body


def client_with(monkeypatch, responses):
    sleeps = []
    client = ga.GitHubClient("secret-token", sleep=sleeps.append)
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: responses.pop(0))
    return client, sleeps


def rate(remaining=4000, cost=2):
    return {"cost": cost, "remaining": remaining, "resetAt": "2099-01-01T00:00:00Z"}


def test_low_rate_limit_waits_until_reset_instead_of_failing(monkeypatch):
    client, sleeps = client_with(monkeypatch, [FakeResponse(200, {"data": {"rateLimit": rate(remaining=5)}})])
    assert client.query("q", {})["rateLimit"]["remaining"] == 5
    assert len(sleeps) == 1 and sleeps[0] > 1000
    assert client.points == 2


def test_rate_limited_error_waits_then_retries(monkeypatch):
    client, sleeps = client_with(monkeypatch, [
        FakeResponse(200, {"data": {"rateLimit": rate()}, "errors": [{"type": "RATE_LIMITED", "message": "x"}]}),
        FakeResponse(200, {"data": {"rateLimit": rate()}}),
    ])
    client.query("q", {})
    assert len(sleeps) == 1


def test_secondary_limit_waits_for_retry_after(monkeypatch):
    client, sleeps = client_with(monkeypatch, [
        FakeResponse(403, headers={"Retry-After": "77"}),
        FakeResponse(200, {"data": {"rateLimit": rate()}}),
    ])
    client.query("q", {})
    assert sleeps == [77.0]


def test_not_found_is_returned_not_raised(monkeypatch):
    client, _ = client_with(monkeypatch, [
        FakeResponse(200, {"data": {"rateLimit": rate(), "repository": None}, "errors": [{"type": "NOT_FOUND", "message": "x"}]})
    ])
    assert client.query("q", {})["repository"] is None


def test_repeated_server_errors_raise_without_exposing_the_token(monkeypatch):
    client, _ = client_with(monkeypatch, [FakeResponse(502) for _ in range(ga.GRAPHQL_RETRIES)])
    with pytest.raises(ga.TransientFailure) as caught:
        client.query("q", {})
    assert "secret-token" not in str(caught.value)


def test_rejected_token_fails_without_exposing_it(monkeypatch):
    client, _ = client_with(monkeypatch, [FakeResponse(401)])
    with pytest.raises(RuntimeError) as caught:
        client.query("q", {})
    assert "secret-token" not in str(caught.value)


def test_estimate_ignores_issues_when_disabled():
    info = {"issues_enabled": False, "issues_before": 10000, "prs_before": 100}
    assert ga.estimated_seconds(info, 1.0) == 3.0


def test_estimate_uses_todays_total_when_search_finds_nothing():
    info = {"issues_enabled": True, "issues_before": 0, "prs_before": 0,
            "issues_total_today": 5000, "prs_total_today": 5000}
    assert ga.estimated_seconds(info, 1.0) == 102.0


def test_query_error_fails_at_once_without_retrying(monkeypatch):
    responses = [FakeResponse(200, {"errors": [{"message": "bad fragment", "extensions": {"code": "fragmentSpreadTypeMismatch"}}]})]
    client, sleeps = client_with(monkeypatch, responses)
    with pytest.raises(ga.InvalidQuery, match="bad fragment"):
        client.query("q", {})
    assert sleeps == []


def test_issue_query_does_not_ask_for_merge_events():
    assert "MergedEvent" not in ga.KINDS["issues"]
    assert "MERGED_EVENT" not in ga.KINDS["issues"]
    assert "MergedEvent" in ga.KINDS["prs"]


class ResolveClient:
    def __init__(self, repository):
        self.repository = repository
        self.searches = []

    def query(self, text, variables):
        if "search" in text:
            self.searches.append(variables)
            return {"issuesBefore": {"issueCount": 7}, "prsBefore": {"issueCount": 9}}
        return {"repository": self.repository}


def test_counts_search_the_current_name_after_a_rename(cache):
    client = ResolveClient({"nameWithOwner": "new/name", "hasIssuesEnabled": True,
                            "issues": {"totalCount": 10}, "pullRequests": {"totalCount": 20}})
    info = ga.resolve_repo(client, "old/name", T0)
    assert info["repo"] == "new/name"
    assert (info["issues_before"], info["prs_before"]) == (7, 9)
    assert client.searches[0]["issues"].startswith("repo:new/name ")
    assert ga.resolve_repo(client, "old/name", T0) == info
    assert len(client.searches) == 1


def test_older_cached_counts_are_replaced(cache):
    ga.write_json(cache / "resolved" / "old__name.json", {"repo": "new/name", "issues_before": 0, "prs_before": 0})
    client = ResolveClient({"nameWithOwner": "new/name", "hasIssuesEnabled": True,
                            "issues": {"totalCount": 10}, "pullRequests": {"totalCount": 20}})
    assert ga.resolve_repo(client, "old/name", T0)["issues_before"] == 7


def test_deleted_repo_gives_a_reason_and_no_search(cache):
    client = ResolveClient(None)
    info = ga.resolve_repo(client, "gone/repo", T0)
    assert info["repo"] is None and "deleted" in info["reason"]
    assert client.searches == []
