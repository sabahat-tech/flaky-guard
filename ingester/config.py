"""Configuration for FlakyGuard ingestion."""
import os

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"

# Repos to mine for workflow run history + ground-truth flaky labels.
# The seed repo (replace YOUR_USERNAME) gives guaranteed positive examples
# even before real-world mining from the large repos below finds enough
# naturally-occurring flaky cases.
TARGET_REPOS = [
    "elastic/elasticsearch",
]

# Issue label(s) used to mine ground-truth flaky test reports.
# Different repos use different conventions — adjust per repo as needed.
FLAKY_LABELS = ["flaky-test", "flaky", "test-flakiness"]

# How many workflow runs to pull per repo. Raised to increase the chance of
# catching commits that got CI-rerun (the source of "reran_commits" evidence
# the detector needs). This will take notably longer to run for large repos
# -- expect tens of minutes for apache/kafka at this size.
MAX_RUNS_PER_REPO = 2000

REQUEST_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    REQUEST_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
