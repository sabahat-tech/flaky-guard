"""
Unit tests for detector/features.py and detector/scorer.py.
"""
import io
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detector.features import load_results, compute_features
from detector.scorer import score_tests, fail_rate_suspicion

SAMPLE_CSV = """test_name,classname,status,duration_s,run_id,commit_sha
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
testNoisy,com.x.A,passed,0.1,1,c1
testNoisy,com.x.A,passed,9.5,2,c2
"""

def get_features():
    df = load_results(io.StringIO(SAMPLE_CSV))
    return compute_features(df)

class TestFeatureComputation:
    def test_stable_pass_fail_rate_zero(self):
        features = get_features()
        row = features[features["test_name"] == "testStable"].iloc[0]
        assert row["fail_rate"] == 0.0

    def test_broken_fail_rate_one(self):
        features = get_features()
        row = features[features["test_name"] == "testBroken"].iloc[0]
        assert row["fail_rate"] == 1.0

    def test_flaky_same_commit_inconsistency_high(self):
        features = get_features()
        row = features[features["test_name"] == "testFlaky"].iloc[0]
        assert row["same_commit_inconsistency"] > 0.5

    def test_stable_same_commit_inconsistency_zero(self):
        features = get_features()
        row = features[features["test_name"] == "testStable"].iloc[0]
        assert row["same_commit_inconsistency"] == 0.0

    def test_broken_same_commit_inconsistency_zero(self):
        features = get_features()
        row = features[features["test_name"] == "testBroken"].iloc[0]
        assert row["same_commit_inconsistency"] == 0.0

    def test_noisy_has_high_duration_cv(self):
        features = get_features()
        row = features[features["test_name"] == "testNoisy"].iloc[0]
        assert row["duration_cv"] > 1.0

    def test_reran_commits_counted_correctly(self):
        features = get_features()
        row = features[features["test_name"] == "testFlaky"].iloc[0]
        assert row["reran_commits"] == 2

    def test_all_tests_present(self):
        features = get_features()
        names = set(features["test_name"].tolist())
        assert "testStable" in names
        assert "testBroken" in names
        assert "testFlaky" in names
        assert "testNoisy" in names

    def test_total_runs_correct(self):
        features = get_features()
        row = features[features["test_name"] == "testFlaky"].iloc[0]
        assert row["total_runs"] == 5

class TestFailRateSuspicion:
    def test_zero_fail_rate_returns_zero(self):
        assert fail_rate_suspicion(0.0) == pytest.approx(0.0)

    def test_full_fail_rate_returns_zero(self):
        assert fail_rate_suspicion(1.0) == pytest.approx(0.0)

    def test_fifty_percent_fail_rate_returns_one(self):
        assert fail_rate_suspicion(0.5) == pytest.approx(1.0)

    def test_quarter_fail_rate_between_zero_and_one(self):
        val = fail_rate_suspicion(0.25)
        assert 0.0 < val < 1.0

    def test_symmetric_around_fifty_percent(self):
        assert fail_rate_suspicion(0.3) == pytest.approx(fail_rate_suspicion(0.7))

class TestScorer:
    def test_flaky_scores_highest_among_rerun_observed(self):
        features = get_features()
        scored = score_tests(features)
        rerun = scored[scored["evidence_level"] == "rerun_observed"]
        assert rerun.iloc[0]["test_name"] == "testFlaky"

    def test_stable_scores_zero(self):
        features = get_features()
        scored = score_tests(features)
        row = scored[scored["test_name"] == "testStable"].iloc[0]
        assert row["flakiness_score"] == pytest.approx(0.0)

    def test_broken_scores_zero(self):
        features = get_features()
        scored = score_tests(features)
        row = scored[scored["test_name"] == "testBroken"].iloc[0]
        assert row["flakiness_score"] == pytest.approx(0.0)

    def test_noisy_scores_zero_without_inconsistency(self):
        features = get_features()
        scored = score_tests(features)
        row = scored[scored["test_name"] == "testNoisy"].iloc[0]
        assert row["flakiness_score"] == pytest.approx(0.0)

    def test_evidence_level_set(self):
        features = get_features()
        scored = score_tests(features)
        row = scored[scored["test_name"] == "testFlaky"].iloc[0]
        assert row["evidence_level"] == "rerun_observed"

    def test_scores_in_valid_range(self):
        features = get_features()
        scored = score_tests(features)
        assert (scored["flakiness_score"] >= 0.0).all()
        assert (scored["flakiness_score"] <= 1.5).all()

    def test_output_sorted_descending_within_evidence_group(self):
        features = get_features()
        scored = score_tests(features)
        rerun = scored[scored["evidence_level"] == "rerun_observed"]["flakiness_score"].tolist()
        assert rerun == sorted(rerun, reverse=True)
