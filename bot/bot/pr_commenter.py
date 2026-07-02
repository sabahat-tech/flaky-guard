"""
FlakyGuard GitHub Bot.

Posts a comment on a Pull Request when failing tests are identified as
likely flaky by the detector, so developers know not to treat the failure
as a real regression.

Two modes of operation:

1. MANUAL (local testing):
   python -m bot.pr_commenter --repo sabahat-tech/flaky-seed-repo \
                               --pr 1 \
                               --scores data/test_results_flakiness_scores.csv \
                               --failed-tests "test_truly_flaky,test_stable_fail"

2. GITHUB ACTIONS (automated):
   Add the workflow in bot/workflows/flaky_comment.yml to your repo.
   It triggers on every push/PR, runs FlakyGuard, and posts comments
   automatically -- no manual intervention needed.

Requires GITHUB_TOKEN with repo scope set as an environment variable.
"""
import os
import sys
import json
import argparse
import requests
import pandas as pd

GITHUB_API = "https://api.github.com"
FLAKY_THRESHOLD = 0.3
BOT_SIGNATURE = "\n\n---\n*🛡️ Posted by [FlakyGuard](https://github.com/sabahat-tech/flaky-guard) — automated flaky test detection*"


def get_headers():
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def load_quarantine_list(scores_csv: str, threshold: float = FLAKY_THRESHOLD) -> dict:
    """
    Load the flakiness scores CSV and return a dict of
    {test_name: score} for all tests above the threshold
    with rerun evidence.
    """
    if not os.path.exists(scores_csv):
        print(f"[warn] scores file not found: {scores_csv}")
        return {}
    df = pd.read_csv(scores_csv)
    flagged = df[
        (df["flakiness_score"].astype(float) >= threshold) &
        (df["evidence_level"] == "rerun_observed")
    ]
    return {
        row["test_name"]: {
            "score": float(row["flakiness_score"]),
            "fail_rate": float(row["fail_rate"]),
            "same_commit_inconsistency": float(row["same_commit_inconsistency"]),
            "classname": row["classname"],
        }
        for _, row in flagged.iterrows()
    }


def find_flaky_failures(failed_tests: list[str], quarantine: dict) -> list[dict]:
    """
    Cross-reference a list of failing test names against the quarantine list.
    Returns details for any failing tests that are known flaky.
    """
    flaky_failures = []
    for test in failed_tests:
        # Match by test_name (exact) or by partial match (handles
        # parameterized test names that vary by run)
        if test in quarantine:
            flaky_failures.append({"test_name": test, **quarantine[test]})
        else:
            # Try partial match for parameterized tests
            for quarantined_name, details in quarantine.items():
                if quarantined_name in test or test in quarantined_name:
                    flaky_failures.append({
                        "test_name": test,
                        "matched_quarantine": quarantined_name,
                        **details
                    })
                    break
    return flaky_failures


def build_pr_comment(flaky_failures: list[dict], run_url: str = None) -> str:
    """Build the markdown comment body for a PR."""
    if not flaky_failures:
        return None

    lines = ["## ⚠️ FlakyGuard: Likely Flaky Test Failures Detected\n"]
    lines.append(
        f"**{len(flaky_failures)} failing test(s)** in this run are in the "
        f"FlakyGuard quarantine list — they may not represent real regressions.\n"
    )

    lines.append("| Test | Flakiness score | Fail rate | Same-commit inconsistency |")
    lines.append("|------|----------------|-----------|--------------------------|")
    for f in flaky_failures:
        score = f.get("score", 0)
        fail_rate = f.get("fail_rate", 0)
        sci = f.get("same_commit_inconsistency", 0)
        lines.append(
            f"| `{f['test_name']}` | **{score:.2f}** | {fail_rate:.0%} | {sci:.2f} |"
        )

    lines.append(
        "\n**What this means:** These tests have been observed failing "
        "nondeterministically on the same commit in previous CI runs. "
        "Before blocking this PR, consider re-running the CI to see if "
        "these failures persist."
    )

    if run_url:
        lines.append(f"\n[View CI run]({run_url})")

    lines.append(BOT_SIGNATURE)
    return "\n".join(lines)


def post_pr_comment(repo: str, pr_number: int, comment_body: str) -> bool:
    """Post a comment on a GitHub PR. Returns True if successful."""
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(
        url,
        headers=get_headers(),
        json={"body": comment_body},
        timeout=30,
    )
    if resp.status_code == 201:
        print(f"✅ Comment posted on {repo}#{pr_number}")
        print(f"   URL: {resp.json().get('html_url')}")
        return True
    else:
        print(f"❌ Failed to post comment: {resp.status_code} {resp.text}")
        return False


def delete_previous_bot_comments(repo: str, pr_number: int):
    """
    Delete any previous FlakyGuard comments on this PR to avoid
    cluttering the thread with repeated identical comments on re-runs.
    """
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.get(url, headers=get_headers(), timeout=30)
    if resp.status_code != 200:
        return
    for comment in resp.json():
        if BOT_SIGNATURE.strip() in comment.get("body", ""):
            del_url = f"{GITHUB_API}/repos/{repo}/issues/comments/{comment['id']}"
            requests.delete(del_url, headers=get_headers(), timeout=30)
            print(f"  Deleted previous bot comment {comment['id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Post a FlakyGuard comment on a GitHub PR"
    )
    parser.add_argument("--repo", required=True,
                         help="GitHub repo in owner/name format")
    parser.add_argument("--pr", type=int, required=True,
                         help="PR number to comment on")
    parser.add_argument("--scores", default="data/test_results_flakiness_scores.csv",
                         help="Path to flakiness scores CSV")
    parser.add_argument("--failed-tests", default="",
                         help="Comma-separated list of failing test names from this run")
    parser.add_argument("--run-url", default=None,
                         help="URL to the CI run (shown in comment)")
    parser.add_argument("--threshold", type=float, default=FLAKY_THRESHOLD,
                         help="Flakiness score threshold for quarantine")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print comment without posting it")
    args = parser.parse_args()

    # Parse failed tests
    failed_tests = [t.strip() for t in args.failed_tests.split(",") if t.strip()]
    if not failed_tests:
        print("No failing tests provided — nothing to check.")
        return

    print(f"Checking {len(failed_tests)} failing test(s) against quarantine list...")
    quarantine = load_quarantine_list(args.scores, args.threshold)
    print(f"Quarantine list has {len(quarantine)} tests (score >= {args.threshold})")

    flaky_failures = find_flaky_failures(failed_tests, quarantine)

    if not flaky_failures:
        print("✅ No failing tests matched the quarantine list — likely a real failure.")
        return

    print(f"⚠️  {len(flaky_failures)} failing test(s) are in the quarantine list:")
    for f in flaky_failures:
        print(f"   - {f['test_name']} (score: {f['score']:.2f})")

    comment = build_pr_comment(flaky_failures, args.run_url)

    if args.dry_run:
        print("\n--- DRY RUN: Comment that would be posted ---")
        print(comment)
        print("--- End of comment ---")
        return

    if not os.environ.get("GITHUB_TOKEN"):
        print("⚠️  GITHUB_TOKEN not set — cannot post comment. Use --dry-run to preview.")
        return

    delete_previous_bot_comments(args.repo, args.pr)
    post_pr_comment(args.repo, args.pr, comment)


if __name__ == "__main__":
    main()
