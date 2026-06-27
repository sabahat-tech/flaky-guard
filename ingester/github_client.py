"""
Thin client around the GitHub REST API for two jobs:

1. Pull workflow run history for a repo (to later extract per-test pass/fail/duration)
2. Mine issues labeled as flaky-test reports (to build ground-truth labels)

Note: GitHub Actions does not expose per-test results natively via the API —
that data lives inside test report artifacts (e.g. JUnit XML) attached to each
run. This client fetches run metadata + artifact links; parsing the JUnit XML
itself happens in `ingester/test_results.py` (next step).
"""
import time
import requests
from . import config


class RateLimitExceeded(Exception):
    """Raised when GitHub's rate limit is hit and a token should be set."""


def _get(url, params=None, max_wait_seconds=120):
    resp = requests.get(url, headers=config.REQUEST_HEADERS, params=params, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - time.time(), 1)
        if not config.GITHUB_TOKEN or wait > max_wait_seconds:
            raise RateLimitExceeded(
                "GitHub rate limit exceeded. Set the GITHUB_TOKEN env var "
                "(a classic PAT with public_repo scope) to raise your quota "
                "from 60/hr to 5000/hr."
            )
        print(f"[rate limit] sleeping {wait:.0f}s")
        time.sleep(wait)
        return _get(url, params, max_wait_seconds)
    resp.raise_for_status()
    return resp


def fetch_workflow_runs(repo: str, max_runs: int = 200):
    """Fetch recent workflow runs for a repo. Returns list of run dicts."""
    runs = []
    page = 1
    while len(runs) < max_runs:
        url = f"{config.GITHUB_API_BASE}/repos/{repo}/actions/runs"
        resp = _get(url, params={"per_page": 100, "page": page})
        batch = resp.json().get("workflow_runs", [])
        if not batch:
            break
        runs.extend(batch)
        page += 1
        if len(batch) < 100:
            break
    return runs[:max_runs]


def fetch_run_artifacts(repo: str, run_id: int):
    """Fetch artifact metadata (e.g. JUnit test reports) for a single run."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/actions/runs/{run_id}/artifacts"
    resp = _get(url)
    return resp.json().get("artifacts", [])


def fetch_flaky_labeled_issues(repo: str, labels: list[str] = None, max_issues: int = 200):
    """Mine issues tagged with flaky-test labels for ground truth."""
    labels = labels or config.FLAKY_LABELS
    found = []
    for label in labels:
        url = f"{config.GITHUB_API_BASE}/search/issues"
        query = f"repo:{repo} label:{label}"
        resp = _get(url, params={"q": query, "per_page": 100})
        found.extend(resp.json().get("items", []))
        if len(found) >= max_issues:
            break
    # de-dupe by issue number
    seen = set()
    deduped = []
    for issue in found:
        if issue["number"] not in seen:
            seen.add(issue["number"])
            deduped.append(issue)
    return deduped[:max_issues]


if __name__ == "__main__":
    # Quick smoke test against a small public repo. Without GITHUB_TOKEN set,
    # the shared 60-req/hr unauthenticated quota will likely already be used
    # up (e.g. by other traffic on this network) -- that's expected, not a
    # bug. Set GITHUB_TOKEN for real development.
    import json
    try:
        runs = fetch_workflow_runs("octocat/Hello-World", max_runs=5)
        print(f"Fetched {len(runs)} runs")
        print(json.dumps([r.get("name") for r in runs], indent=2))
    except RateLimitExceeded as e:
        print(f"[expected without a token] {e}")
