"""
ML-based flakiness detector using a Random Forest classifier.

Designed to sit alongside the existing heuristic scorer (detector/scorer.py)
as a second model, so the two can be directly compared on the same dataset.

Training approach:
  - Features: same ones computed by detector/features.py (fail_rate,
    same_commit_inconsistency, duration_cv, total_runs, distinct_commits,
    reran_commits) -- giving the ML model the same raw signals the heuristic
    uses, so any performance difference is attributable to the model, not
    different inputs.
  - Labels: binary (1 = flaky, 0 = not flaky). Labels come from a ground-
    truth CSV you supply (see make_ground_truth_csv() helper below), which
    maps (classname, test_name) pairs to known labels mined from GitHub
    issues or your controlled seed repo.
  - Model: Random Forest. Good choice for this problem because:
      1. Works well on small, tabular datasets (you won't have thousands of
         labeled examples).
      2. Produces calibrated probability scores (predict_proba), directly
         comparable to the heuristic's [0,1] score.
      3. Feature importances are interpretable -- examiners can see WHICH
         signals the model learned to rely on most.
  - Evaluation: 5-fold stratified cross-validation (handles class imbalance
    since flaky tests are rare) reporting precision, recall, F1, and AUC.

Usage:
    # Train and evaluate (cross-validation):
    python -m detector.ml_model --features data/test_results.csv \
                                --labels data/ground_truth.csv \
                                --evaluate

    # Train on all data, save model, score unlabeled tests:
    python -m detector.ml_model --features data/test_results.csv \
                                --labels data/ground_truth.csv \
                                --save-model data/flaky_model.pkl \
                                --score-output data/ml_scores.csv
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detector import features as feat

# Features passed to the model -- same signals as heuristic scorer
FEATURE_COLS = [
    "fail_rate",
    "same_commit_inconsistency",
    "duration_cv",
    "total_runs",
    "distinct_commits",
    "reran_commits",
]


def make_ground_truth_csv(output_path: str):
    """
    Helper: write a template ground_truth.csv you can fill in manually.
    Columns: classname, test_name, is_flaky (1 or 0)

    For your seed repo, these are the known correct labels:
      tests.test_flaky_demo  test_truly_flaky       1
      tests.test_flaky_demo  test_stable_pass       0
      tests.test_flaky_demo  test_stable_fail       0
      tests.test_flaky_demo  test_noisy_but_stable  0
    """
    template = pd.DataFrame([
        {"classname": "tests.test_flaky_demo",
         "test_name": "test_truly_flaky", "is_flaky": 1},
        {"classname": "tests.test_flaky_demo",
         "test_name": "test_stable_pass", "is_flaky": 0},
        {"classname": "tests.test_flaky_demo",
         "test_name": "test_stable_fail", "is_flaky": 0},
        {"classname": "tests.test_flaky_demo",
         "test_name": "test_noisy_but_stable", "is_flaky": 0},
    ])
    template.to_csv(output_path, index=False)
    print(f"Template ground truth written to {output_path}")
    print("Edit it to add more labeled examples from real repos.")


def load_labeled_dataset(features_csv: str, labels_csv: str):
    """
    Load features and merge with ground-truth labels.
    Returns (X DataFrame, y Series, merged DataFrame for inspection).
    """
    raw_df = feat.load_results(features_csv)
    features_df = feat.compute_features(raw_df)

    labels_df = pd.read_csv(labels_csv)
    merged = features_df.merge(labels_df, on=["classname", "test_name"], how="inner")

    if merged.empty:
        raise ValueError(
            "No overlap between features and labels — check that classname "
            "and test_name values match exactly between the two CSVs."
        )

    print(f"Labeled dataset: {len(merged)} tests "
          f"({merged['is_flaky'].sum()} flaky, "
          f"{(merged['is_flaky']==0).sum()} non-flaky)")

    X = merged[FEATURE_COLS].fillna(0)
    y = merged["is_flaky"]
    return X, y, merged


def build_pipeline():
    """Build the sklearn Pipeline: StandardScaler + RandomForest."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=4,          # shallow trees prevent overfitting on small data
            min_samples_leaf=1,
            class_weight="balanced",  # handles class imbalance (few flaky tests)
            random_state=42,
        )),
    ])


def evaluate_model(X, y):
    """
    5-fold stratified cross-validation. Reports precision, recall, F1, AUC.
    Returns the cv_results dict for further inspection.
    """
    if len(y.unique()) < 2:
        print("WARNING: only one class present in labels -- cannot evaluate. "
              "Add more labeled examples (both flaky and non-flaky).")
        return None

    n_splits = min(5, int(y.sum()))  # can't have more folds than positive examples
    if n_splits < 2:
        print(f"WARNING: only {int(y.sum())} positive examples -- "
              f"cross-validation unreliable. Add more flaky test labels.")
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipeline = build_pipeline()

    scoring = ["precision", "recall", "f1", "roc_auc"]
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                              return_train_score=False)

    print("\n=== Cross-validation results ===")
    print(f"Folds: {n_splits}")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")

    return results


def train_and_score(X_train, y_train, X_all_features: pd.DataFrame,
                     features_df: pd.DataFrame):
    """
    Train on labeled data, then score ALL tests in features_df
    (including unlabeled ones) to produce a ranking comparable to the
    heuristic scorer's output.
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Feature importances (from the RandomForest inside the pipeline)
    rf = pipeline.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    print("\n=== Feature importances (Random Forest) ===")
    for feat_name, imp in importances.sort_values(ascending=False).items():
        print(f"  {feat_name:30s}: {imp:.3f}")

    # Score all tests
    X_all = features_df[FEATURE_COLS].fillna(0)
    proba = pipeline.predict_proba(X_all)[:, 1]  # probability of class 1 (flaky)

    scored = features_df.copy()
    scored["ml_flakiness_score"] = proba.round(4)
    scored = scored.sort_values("ml_flakiness_score", ascending=False)
    return scored, pipeline


def main():
    parser = argparse.ArgumentParser(description="ML flakiness detector")
    parser.add_argument("--features", default="data/test_results.csv")
    parser.add_argument("--labels", default="data/ground_truth.csv")
    parser.add_argument("--make-ground-truth", action="store_true",
                         help="Write a template ground_truth.csv and exit")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run cross-validation and print metrics")
    parser.add_argument("--save-model", default=None,
                         help="Path to save trained model (.pkl)")
    parser.add_argument("--score-output", default="data/ml_scores.csv",
                         help="Where to write scored test CSV")
    args = parser.parse_args()

    if args.make_ground_truth:
        make_ground_truth_csv(args.labels)
        return

    X, y, merged = load_labeled_dataset(args.features, args.labels)

    if args.evaluate:
        evaluate_model(X, y)

    # Load ALL features (not just labeled) for scoring
    raw_df = feat.load_results(args.features)
    all_features = feat.compute_features(raw_df)

    scored, pipeline = train_and_score(X, y, all_features[FEATURE_COLS], all_features)
    scored.to_csv(args.score_output, index=False)
    print(f"\nML scores written to {args.score_output}")
    print(scored[["classname", "test_name", "ml_flakiness_score"]].head(10).to_string(index=False))

    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
