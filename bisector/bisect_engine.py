"""
Automated bisection: given a known-good commit, a known-bad commit, and a
way to run a specific test, binary-search the commit history between them
to find the exact commit that introduced the failure.

This mirrors `git bisect`, but is driven programmatically so it can be
triggered automatically by FlakyGuard when the detector confirms a failure
is NOT flaky (i.e. a genuine regression worth bisecting, as opposed to a
flaky test where bisection would be meaningless/misleading).

Usage assumes a local clone of the target repo. For CI-triggered bisection
in a real deployment, you'd swap `run_test_at_commit` to trigger a CI job
per commit instead of running locally -- the binary-search logic itself
doesn't change.
"""
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BisectStep:
    commit_sha: str
    commit_short: str
    result: str  # "pass" | "fail" | "skip" (e.g. couldn't build at that commit)


@dataclass
class BisectResult:
    culprit_commit: str = None
    steps: list = field(default_factory=list)
    total_commits_checked: int = 0


def run_command(cmd: list[str], cwd: str) -> tuple[int, str]:
    """Run a shell command, return (exit_code, combined_output)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def get_commits_between(repo_path: str, good_sha: str, bad_sha: str) -> list[str]:
    """
    Return the list of commit SHAs from good_sha (exclusive) to bad_sha
    (inclusive), oldest first -- the candidate range to bisect through.
    """
    code, out = run_command(
        ["git", "rev-list", "--reverse", f"{good_sha}..{bad_sha}"], cwd=repo_path
    )
    if code != 0:
        raise RuntimeError(f"git rev-list failed: {out}")
    commits = [c for c in out.strip().splitlines() if c]
    return commits


def checkout_commit(repo_path: str, commit_sha: str) -> bool:
    code, out = run_command(["git", "checkout", "--force", commit_sha], cwd=repo_path)
    return code == 0


def clear_python_cache(repo_path: str):
    """
    Delete all __pycache__ directories and .pyc files in the repo before
    each test run. Without this, stale compiled bytecode from a PREVIOUS
    checkout can silently get reused even after `git checkout` changes the
    .py source -- git doesn't reliably bump file mtimes in a way Python's
    cache invalidation always catches, which can make bisection report the
    wrong commit. This was caught via a controlled test where the known
    culprit commit and the bisector's reported commit didn't match.
    """
    repo = Path(repo_path)
    for cache_dir in repo.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for pyc_file in repo.rglob("*.pyc"):
        pyc_file.unlink(missing_ok=True)


def resolve_python_command(test_command: list[str]) -> list[str]:
    """
    Replace a bare "python" or "python3" in the test command with
    sys.executable -- the exact interpreter currently running this script.

    Without this, subprocess.run(["python", ...]) on Windows can resolve
    "python" via PATH to a DIFFERENT interpreter than the one the user
    activated (e.g. a global install instead of the venv), even when the
    parent shell's prompt shows the venv is active. This was confirmed via
    a controlled test where bisection consistently failed with "No module
    named pytest" despite pytest being correctly installed in the active
    venv -- the subprocess was silently using a different, global Python.
    """
    resolved = list(test_command)
    if resolved and resolved[0] in ("python", "python3"):
        resolved[0] = sys.executable
    return resolved


def run_test_at_commit(repo_path: str, commit_sha: str, test_command: list[str]) -> tuple[str, str]:
    """
    Checkout the given commit and run the test command.
    Returns (outcome, output) where outcome is "pass", "fail", or "skip"
    (checkout itself failed) and output is the full combined stdout+stderr,
    useful for diagnosing WHY a step failed instead of guessing.
    """
    if not checkout_commit(repo_path, commit_sha):
        return "skip", "(checkout failed)"

    clear_python_cache(repo_path)
    resolved_command = resolve_python_command(test_command)
    code, output = run_command(resolved_command, cwd=repo_path)
    return ("pass" if code == 0 else "fail"), output


def bisect(repo_path: str,
           good_sha: str,
           bad_sha: str,
           test_command: list[str],
           progress_callback=None) -> BisectResult:
    """
    Binary search between good_sha (known passing) and bad_sha (known
    failing) to find the first commit where test_command starts failing.

    test_command should run ONLY the specific test in question (e.g.
    ["pytest", "tests/test_flaky_demo.py::test_stable_fail", "-x"]) --
    bisecting against the whole suite would conflate unrelated breakage.
    """
    commits = get_commits_between(repo_path, good_sha, bad_sha)
    result = BisectResult()

    if not commits:
        # bad_sha IS the immediate next commit after good_sha
        result.culprit_commit = bad_sha
        return result

    lo, hi = 0, len(commits) - 1
    culprit_index = len(commits) - 1  # default: bad_sha itself, if nothing else fails

    while lo <= hi:
        mid = (lo + hi) // 2
        commit = commits[mid]
        outcome, output = run_test_at_commit(repo_path, commit, test_command)
        result.steps.append(BisectStep(commit, commit[:8], outcome))
        result.total_commits_checked += 1

        if progress_callback:
            progress_callback(commit, outcome, result.total_commits_checked, output)

        if outcome == "skip":
            # Can't draw a conclusion here -- shrink the search conservatively
            # by treating it like "pass" (move forward) since we have no
            # better signal; a more advanced version could try the adjacent
            # commit instead.
            lo = mid + 1
        elif outcome == "fail":
            culprit_index = mid
            hi = mid - 1
        else:  # pass
            lo = mid + 1

    result.culprit_commit = commits[culprit_index]
    return result


if __name__ == "__main__":
    print(__doc__)
    print("Run via the CLI entrypoint instead: python -m bisector.run_bisect "
          "<repo_path> <good_sha> <bad_sha> -- <test_command...>")
