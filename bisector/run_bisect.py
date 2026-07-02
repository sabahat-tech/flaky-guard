"""
CLI for running a bisection.

Example:
    python -m bisector.run_bisect /path/to/repo abc123 def456 -- pytest tests/test_foo.py::test_bar
"""
import argparse
import sys

from . import bisect_engine as be


def main():
    parser = argparse.ArgumentParser(
        description="Bisect a git repo to find which commit broke a test"
    )
    parser.add_argument("repo_path")
    parser.add_argument("good_sha", help="Last known-good commit SHA")
    parser.add_argument("bad_sha", help="Known-bad (failing) commit SHA")
    parser.add_argument("test_command", nargs="+",
                         help="Command to run the specific failing test, e.g. "
                              "-- pytest tests/test_foo.py::test_bar")
    parser.add_argument("--verbose", action="store_true",
                         help="Print full test output for each step (for debugging)")
    args = parser.parse_args()

    def progress(commit, outcome, n, output):
        print(f"  [{n}] {commit[:8]} -> {outcome}")
        if args.verbose:
            print(f"      --- output ---")
            for line in output.strip().splitlines():
                print(f"      {line}")
            print(f"      --------------")

    print(f"Bisecting {args.repo_path}")
    print(f"  good: {args.good_sha[:8]}  bad: {args.bad_sha[:8]}")
    print(f"  test: {' '.join(args.test_command)}\n")

    result = be.bisect(
        args.repo_path, args.good_sha, args.bad_sha, args.test_command,
        progress_callback=progress,
    )

    print(f"\nChecked {result.total_commits_checked} commits "
          f"(out of range size, via binary search)")
    print(f"Culprit commit: {result.culprit_commit}")


if __name__ == "__main__":
    main()
