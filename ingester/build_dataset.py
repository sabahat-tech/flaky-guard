"""
End-to-end dataset builder.

For each repo in config.TARGET_REPOS:
  1. Fetch recent workflow runs
  2. For each run, fetch its artifacts
  3. Download + extract any artifact that looks like a test report
  4. Parse JUnit XML into per-test records
  5. Append everything into one CSV: data/test_results.csv

Run this with GITHUB_TOKEN set, or it will fail fast with a clear error
once the unauthenticated rate limit (60 req/hr) is hit.

NOTE: not every CI workflow uploads a JUnit XML artifact -- many projects
only print results to the log, or use a different report format/name.
This script skips runs with no matching artifact and reports a summary at
the end so you know your real "usable run" rate. That number is itself
worth recording in your project writeup (it's a known practical limitation
of mining real-world CI data).
"""
import os
import csv
from tqdm import tqdm

from . import config
from . import github_client as gh
from . import artifact_downloader as dl
from . import test_results as tr

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "test_results.csv")

# Artifact names commonly used for test reports across ecosystems.
# Extend this list as you discover naming conventions in your target repos.
LIKELY_TEST_ARTIFACT_KEYWORDS = ["test", "junit", "report", "results"]


def looks_like_test_artifact(artifact_name: str) -> bool:
    name = artifact_name.lower()
    return any(kw in name for kw in LIKELY_TEST_ARTIFACT_KEYWORDS)


def load_existing_records(csv_path: str) -> list[dict]:
    """Load previously saved records so a new run merges with, rather than
    overwrites, prior progress (e.g. a separate Kafka-only run before this
    one targets Elasticsearch)."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def dedupe_records(records: list[dict]) -> list[dict]:
    """De-duplicate by (run_id, classname, test_name) so re-running on
    overlapping data (e.g. partial reruns) doesn't create duplicate rows."""
    seen = set()
    deduped = []
    for r in records:
        key = (r.get("run_id"), r.get("classname"), r.get("test_name"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def build_dataset(repos=None, max_runs_per_repo=None, checkpoint_every=20, merge=True):
    repos = repos or config.TARGET_REPOS
    max_runs_per_repo = max_runs_per_repo or config.MAX_RUNS_PER_REPO

    all_records = load_existing_records(OUTPUT_CSV) if merge else []
    if all_records:
        print(f"Loaded {len(all_records)} existing rows from {OUTPUT_CSV} -- "
              f"new data will be merged in, not overwrite them.")

    stats = {"runs_seen": 0, "runs_with_test_artifact": 0, "xml_files_parsed": 0}

    def save_checkpoint():
        """Write whatever has been collected so far (existing + new merged).
        Called periodically AND at the end, so a crash/forced shutdown never
        loses more than `checkpoint_every` runs' worth of new work."""
        if not all_records:
            return
        deduped = dedupe_records(all_records)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=deduped[0].keys())
            writer.writeheader()
            writer.writerows(deduped)

    for repo in repos:
        print(f"\n=== {repo} ===")
        runs = gh.fetch_workflow_runs(repo, max_runs=max_runs_per_repo)
        print(f"Found {len(runs)} recent runs")

        for i, run in enumerate(tqdm(runs, desc=f"{repo} runs")):
            stats["runs_seen"] += 1
            run_id = run["id"]
            commit_sha = run.get("head_sha")

            try:
                artifacts = gh.fetch_run_artifacts(repo, run_id)
            except Exception as e:
                print(f"  [skip] run {run_id}: could not list artifacts ({e})")
                continue

            test_artifacts = [a for a in artifacts if looks_like_test_artifact(a["name"])]
            if not test_artifacts:
                continue
            stats["runs_with_test_artifact"] += 1

            for artifact in test_artifacts:
                print(f"  downloading artifact {artifact['id']} ({artifact['name']}, "
                      f"{artifact.get('size_in_bytes', 0) / 1e6:.1f}MB)...")
                try:
                    extract_path = dl.download_artifact(repo, artifact["id"])
                except Exception as e:
                    print(f"  [skip] artifact {artifact['id']}: download failed ({e})")
                    continue

                xml_files = dl.find_junit_xml_files(extract_path)
                for xml_path in xml_files:
                    try:
                        results = tr.parse_junit_xml(xml_path, run_id=run_id, commit_sha=commit_sha)
                        all_records.extend(tr.results_to_records(results))
                        stats["xml_files_parsed"] += 1
                    except Exception as e:
                        print(f"  [skip] could not parse {xml_path}: {e}")

            # Checkpoint periodically -- the key fix for laptops that
            # shut down mid-run (e.g. thermal protection).
            if (i + 1) % checkpoint_every == 0:
                save_checkpoint()
                print(f"  [checkpoint] saved {len(all_records)} rows so far")

    save_checkpoint()
    if all_records:
        print(f"\nWrote {len(all_records)} test result rows to {OUTPUT_CSV}")
    else:
        print("\nNo test result records collected -- see stats below for why.")

    print(f"\nSummary: {stats}")
    return all_records, stats


if __name__ == "__main__":
    if not config.GITHUB_TOKEN:
        print("WARNING: GITHUB_TOKEN not set. You will likely hit the 60 "
              "req/hr unauthenticated limit almost immediately on a real "
              "repo like apache/kafka. Set it and re-run.")
    build_dataset()
