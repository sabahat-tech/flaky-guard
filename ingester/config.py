"""Configuration for FlakyGuard ingestion."""
import os

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"

# Repos to mine for workflow run history + ground-truth flaky labels.
# Start with one repo while developing, expand once the pipeline works.
TARGET_REPOS = [
    "apache/kafka",
    "elastic/elasticsearch",
]

# Issue label(s) used to mine ground-truth flaky test reports.
# Different repos use different conventions — adjust per repo as needed.
FLAKY_LABELS = ["flaky-test", "flaky", "test-flakiness"]

# How many workflow runs to pull per repo (start small to respect rate limits
# on an unauthenticated/low-quota token; raise once you have a real PAT).
MAX_RUNS_PER_REPO = 200

REQUEST_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    REQUEST_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
