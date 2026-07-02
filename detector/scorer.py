"""
Combines the per-test features from features.py into a single flakiness
score in [0, 1], and provides a CLI to rank the most-flaky tests in a
real test_results.csv dataset.

This is a heuristic baseline -- intentionally simple and explainable, which
is a defensible choice for an FYP: it gives you something to beat with a
learned model later (e.g. logistic regression / random forest on these same
features) and a clear story for your evaluation chapter: "heuristic baseline
vs learned model, here's the precision/recall delta."

Scoring rationale:
  - same_commit_inconsistency is weighted highest: it is the strongest
    possible signal (same code, different outcome) and has near-zero false
    positive rate by construction.
  - fail_rate contributes only in the "suspicious middle" (neither ~0% nor
    ~100%), via a simple penalty for values near the extremes.
  - duration_cv contributes a smaller weight as a secondary signal (timing
    erraticism often co-occurs with flakiness, e.g. timeouts/races) but is
    noisy alone, so it should never dominate the score.
"""
import argparse
import numpy as np
import pandas as pd

from . import features as feat


def fail_rate_suspicion(fail_rate: float) -> float:
    """
    Peaks at fail_rate=0.5 (most suspicious), drops to 0 at the extremes
    (0% or 100% -- consistent behavior, not flaky).
    """
    return 1 - abs(fail_rate - 0.5) * 2


def score_tests(features_df: pd.DataFrame,
                 w_inconsistency: float = 0.7,
                 w_fail_rate: float = 0.3,
                 w_duration_bonus: float = 0.1) -> pd.DataFrame:
    """
    Scoring philosophy: same_commit_inconsistency and fail_rate_suspicion
    are the ONLY signals trusted to drive the score, because both require
    actual evidence (a real failure occurred, or the same commit produced
    different outcomes). duration_cv is demoted to a small bonus added ONLY
    on top of tests that already have at least one rerun signal -- it can
    nudge an already-suspicious test higher, but it can no longer manufacture
    a high score on its own out of pure timing noise.

    Tests with reran_commits == 0 have NO evidence either way for the
    inconsistency signal (not "proven stable" -- simply unobserved), so they
    are marked evidence_level="insufficient" and excluded from the main
    ranked list rather than silently scored on duration alone.
    """
    df = features_df.copy()
    df["fail_rate_suspicion"] = df["fail_rate"].apply(fail_rate_suspicion)
    df["duration_signal"] = df["duration_cv"].clip(0, 2) / 2

    df["evidence_level"] = np.where(df["reran_commits"] > 0, "rerun_observed", "insufficient")

    base_score = (
        w_inconsistency * df["same_commit_inconsistency"]
        + w_fail_rate * df["fail_rate_suspicion"]
    )
    # Duration only adds on top when the test ALREADY shows actual
    # inconsistency (same_commit_inconsistency > 0) -- not merely "this
    # commit happened to be rerun." A test that reran and passed both
    # times consistently is not flaky no matter how noisy its timing is;
    # gating on reran_commits alone let exactly that case slip back in.
    duration_bonus = np.where(
        df["same_commit_inconsistency"] > 0,
        w_duration_bonus * df["duration_signal"],
        0.0,
    )
    df["flakiness_score"] = (base_score + duration_bonus).round(4)

    # Tests with zero evidence get scored on fail_rate_suspicion alone
    # (still a real signal -- a 30% fail rate across many commits is
    # meaningful even without rerun data) but flagged so you know to
    # treat them with more caution than rerun-confirmed ones.
    df.loc[df["evidence_level"] == "insufficient", "flakiness_score"] = (
        w_fail_rate * df.loc[df["evidence_level"] == "insufficient", "fail_rate_suspicion"]
    ).round(4)

    return df.sort_values(
        ["evidence_level", "flakiness_score"], ascending=[True, False]
    )


def main():
    parser = argparse.ArgumentParser(description="Rank tests by flakiness score")
    parser.add_argument("csv_path", help="Path to test_results.csv")
    parser.add_argument("--top", type=int, default=20, help="Show top N flakiest tests")
    parser.add_argument("--threshold", type=float, default=0.3,
                         help="Score above which a test is flagged as likely flaky")
    args = parser.parse_args()

    df = feat.load_results(args.csv_path)
    features_df = feat.compute_features(df)
    scored = score_tests(features_df)

    n_with_evidence = (scored["evidence_level"] == "rerun_observed").sum()
    n_insufficient = (scored["evidence_level"] == "insufficient").sum()
    flagged = scored[(scored["evidence_level"] == "rerun_observed")
                      & (scored["flakiness_score"] >= args.threshold)]

    print(f"Loaded {len(df)} test result rows, {len(scored)} unique tests")
    print(f"  {n_with_evidence} tests have rerun evidence (same commit observed 2+ times)")
    print(f"  {n_insufficient} tests have insufficient evidence (no commit reran) -- "
          f"scored on fail_rate alone, treat with caution")
    print(f"Flagged {len(flagged)} tests as likely flaky (score >= {args.threshold}, "
          f"rerun-evidence only)\n")
    print(scored.head(args.top).to_string(index=False))

    out_path = args.csv_path.replace(".csv", "_flakiness_scores.csv")
    scored.to_csv(out_path, index=False)
    print(f"\nFull ranked results written to {out_path}")


if __name__ == "__main__":
    main()
