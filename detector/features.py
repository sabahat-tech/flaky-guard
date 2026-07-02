"""
Turns the raw per-run test_results.csv into per-test aggregate features that
signal flakiness.

Key insight: a test that simply fails consistently (e.g. broken code) is NOT
flaky -- it's just failing. A flaky test is one whose outcome is inconsistent
*for the same code* (i.e. same commit_sha, different status across runs/
reruns), or whose failure rate sits in a suspicious middle ground across many
distinct commits with no fail-rate clustering near 0% or 100%.

Features computed per (classname, test_name):
  - total_runs: how many times we've observed this test
  - distinct_commits: how many distinct commits it ran under
  - fail_rate: overall failure rate (failed+error / total)
  - same_commit_inconsistency: fraction of commits where this test had BOTH
    a pass and a fail recorded (the strongest single flakiness signal --
    same code, different outcome)
  - duration_cv: coefficient of variation of duration (flaky tests often
    have erratic timing, e.g. timeouts, race conditions)
"""
import pandas as pd
import numpy as np


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_fail"] = df["status"].isin(["failed", "error"]).astype(int)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["classname", "test_name"])

    rows = []
    for (classname, test_name), g in grouped:
        total_runs = len(g)
        distinct_commits = g["commit_sha"].nunique()
        fail_rate = g["is_fail"].mean()

        # Same-commit inconsistency: for each commit, did we see BOTH a pass
        # and a fail for this exact test? That's same code, different result
        # -- the clearest flakiness signal available from CI history alone.
        per_commit = g.groupby("commit_sha")["is_fail"].agg(["min", "max"])
        inconsistent_commits = (per_commit["min"] != per_commit["max"]).sum()
        same_commit_inconsistency = (
            inconsistent_commits / len(per_commit) if len(per_commit) > 0 else 0.0
        )

        # How many commits did we see this test run on MORE THAN ONCE?
        # This is the prerequisite for same_commit_inconsistency to mean
        # anything at all -- if every commit only has 1 run, that column
        # is 0 by construction (no evidence either way), not because the
        # test is well-behaved.
        runs_per_commit = g.groupby("commit_sha").size()
        reran_commits = int((runs_per_commit >= 2).sum())

        durations = g["duration_s"].dropna()
        duration_cv = (
            durations.std() / durations.mean()
            if len(durations) > 1 and durations.mean() > 0
            else 0.0
        )

        rows.append({
            "classname": classname,
            "test_name": test_name,
            "total_runs": total_runs,
            "distinct_commits": distinct_commits,
            "reran_commits": reran_commits,
            "fail_rate": round(fail_rate, 4),
            "same_commit_inconsistency": round(same_commit_inconsistency, 4),
            "duration_cv": round(duration_cv, 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Synthetic smoke test: 3 tests with distinct behavior patterns.
    import io

    sample_csv = """test_name,classname,status,duration_s,run_id,commit_sha
testStable,com.x.A,passed,0.1,1,c1
testStable,com.x.A,passed,0.1,2,c2
testStable,com.x.A,passed,0.1,3,c3
testBroken,com.x.A,failed,0.2,1,c1
testBroken,com.x.A,failed,0.2,2,c2
testBroken,com.x.A,failed,0.2,3,c3
testFlaky,com.x.A,passed,0.5,1,c1
testFlaky,com.x.A,failed,4.8,1,c1
testFlaky,com.x.A,passed,0.4,2,c2
testFlaky,com.x.A,failed,5.1,2,c2
testFlaky,com.x.A,passed,0.6,3,c3
"""
    df = load_results(io.StringIO(sample_csv))
    features = compute_features(df)
    print(features.to_string(index=False))
