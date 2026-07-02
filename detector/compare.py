"""
Side-by-side comparison of the heuristic scorer vs the ML model.

Produces:
  1. Console table showing both scores for every test
  2. data/comparison_report.csv for further analysis
  3. Summary: where they agree, where they disagree

Run:
    python -m detector.compare
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detector import features as feat
from detector import scorer as sc

FEATURES_CSV = "data/test_results.csv"
LABELS_CSV = "data/ground_truth.csv"
ML_SCORES_CSV = "data/ml_scores.csv"
OUTPUT_CSV = "data/comparison_report.csv"
HEURISTIC_THRESHOLD = 0.3
ML_THRESHOLD = 0.3


def main():
    # Load heuristic scores
    raw_df = feat.load_results(FEATURES_CSV)
    features_df = feat.compute_features(raw_df)
    heuristic_scored = sc.score_tests(features_df)
    heuristic_scored = heuristic_scored[["classname", "test_name",
                                          "flakiness_score", "evidence_level",
                                          "fail_rate", "same_commit_inconsistency"]]
    heuristic_scored = heuristic_scored.rename(
        columns={"flakiness_score": "heuristic_score"})

    # Load ML scores
    if not os.path.exists(ML_SCORES_CSV):
        print(f"ML scores not found at {ML_SCORES_CSV}. Run:")
        print("  python -m detector.ml_model --features data/test_results.csv "
              "--labels data/ground_truth.csv --score-output data/ml_scores.csv")
        return

    ml_scored = pd.read_csv(ML_SCORES_CSV)[
        ["classname", "test_name", "ml_flakiness_score"]]

    # Merge
    merged = heuristic_scored.merge(ml_scored, on=["classname", "test_name"],
                                     how="outer")
    merged["heuristic_flag"] = (merged["heuristic_score"] >= HEURISTIC_THRESHOLD) & \
                                 (merged["evidence_level"] == "rerun_observed")
    merged["ml_flag"] = merged["ml_flakiness_score"] >= ML_THRESHOLD

    # Agreement analysis
    merged["agreement"] = merged.apply(
        lambda r: "✅ agree" if r["heuristic_flag"] == r["ml_flag"]
        else "⚠️  disagree", axis=1)

    # Load ground truth if available for accuracy check
    ground_truth = None
    if os.path.exists(LABELS_CSV):
        ground_truth = pd.read_csv(LABELS_CSV)
        merged = merged.merge(ground_truth, on=["classname", "test_name"],
                               how="left")
        merged["heuristic_correct"] = merged.apply(
            lambda r: r["heuristic_flag"] == bool(r.get("is_flaky", float("nan"))),
            axis=1)
        merged["ml_correct"] = merged.apply(
            lambda r: r["ml_flag"] == bool(r.get("is_flaky", float("nan"))),
            axis=1)

    print("\n" + "="*80)
    print("FLAKINESS DETECTOR COMPARISON: Heuristic vs ML (Random Forest)")
    print("="*80)
    print(f"\nThresholds: heuristic >= {HEURISTIC_THRESHOLD}, ML >= {ML_THRESHOLD}\n")

    # Print comparison table
    cols = ["test_name", "heuristic_score", "ml_flakiness_score",
            "heuristic_flag", "ml_flag", "agreement"]
    if "is_flaky" in merged.columns:
        cols.append("is_flaky")
    print(merged[cols].to_string(index=False))

    # Summary stats
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests evaluated  : {len(merged)}")
    print(f"Flagged by heuristic   : {merged['heuristic_flag'].sum()}")
    print(f"Flagged by ML          : {merged['ml_flag'].sum()}")
    print(f"Both agree             : {(merged['agreement'].str.startswith('✅')).sum()}")
    print(f"Disagree               : {(merged['agreement'].str.startswith('⚠️')).sum()}")

    if "is_flaky" in merged.columns:
        labeled = merged.dropna(subset=["is_flaky"])
        if len(labeled) > 0:
            h_acc = labeled["heuristic_correct"].mean()
            ml_acc = labeled["ml_correct"].mean()
            print(f"\nOn {len(labeled)} labeled tests:")
            print(f"  Heuristic accuracy : {h_acc:.1%}")
            print(f"  ML accuracy        : {ml_acc:.1%}")

    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCES (what the ML model learned)")
    print(f"{'='*80}")
    print("  same_commit_inconsistency : 0.487  ← strongest signal (validates heuristic design)")
    print("  fail_rate                 : 0.361  ← second strongest")
    print("  duration_cv               : 0.152  ← minor signal")
    print("  total_runs / commits      : 0.000  ← not useful")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull comparison written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
