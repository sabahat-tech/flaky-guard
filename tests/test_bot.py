"""
Unit tests for bot/pr_commenter.py.
Tests cover: quarantine list loading, flaky failure detection,
comment building, and partial name matching for parameterized tests.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot.pr_commenter import (
    load_quarantine_list, find_flaky_failures, build_pr_comment
)

SCORES_CSV = """classname,test_name,total_runs,distinct_commits,reran_commits,fail_rate,same_commit_inconsistency,duration_cv,fail_rate_suspicion,duration_signal,evidence_level,flakiness_score
tests.demo,test_truly_flaky,15,1,1,0.2,1.0,3.87,0.6,0.5,rerun_observed,0.92
tests.demo,test_stable_pass,15,1,1,0.0,0.0,3.87,0.0,0.5,rerun_observed,0.0
tests.demo,test_stable_fail,15,1,1,1.0,0.0,0.4,0.0,0.5,rerun_observed,0.0
tests.demo,test_noisy_but_stable,15,1,1,0.0,0.0,0.48,0.0,0.5,rerun_observed,0.0
"""


def write_scores_csv():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(SCORES_CSV)
    f.close()
    return f.name


class TestLoadQuarantineList:

    def test_loads_only_tests_above_threshold(self):
        path = write_scores_csv()
        quarantine = load_quarantine_list(path, threshold=0.3)
        assert "test_truly_flaky" in quarantine
        assert "test_stable_pass" not in quarantine
        assert "test_stable_fail" not in quarantine
        os.unlink(path)

    def test_returns_correct_score(self):
        path = write_scores_csv()
        quarantine = load_quarantine_list(path, threshold=0.3)
        assert quarantine["test_truly_flaky"]["score"] == pytest.approx(0.92)
        os.unlink(path)

    def test_empty_if_threshold_too_high(self):
        path = write_scores_csv()
        quarantine = load_quarantine_list(path, threshold=0.99)
        assert len(quarantine) == 0
        os.unlink(path)

    def test_returns_empty_if_file_missing(self):
        quarantine = load_quarantine_list("/nonexistent/path.csv")
        assert quarantine == {}


class TestFindFlakyFailures:

    def setup_method(self):
        path = write_scores_csv()
        self.quarantine = load_quarantine_list(path, threshold=0.3)
        os.unlink(path)

    def test_finds_exact_match(self):
        failures = find_flaky_failures(["test_truly_flaky"], self.quarantine)
        assert len(failures) == 1
        assert failures[0]["test_name"] == "test_truly_flaky"

    def test_no_match_for_stable_tests(self):
        failures = find_flaky_failures(
            ["test_stable_pass", "test_stable_fail"], self.quarantine
        )
        assert len(failures) == 0

    def test_partial_match_for_parameterized_test(self):
        """Parameterized tests have names like 'test_truly_flaky[param=1]'."""
        failures = find_flaky_failures(
            ["test_truly_flaky[param=value]"], self.quarantine
        )
        assert len(failures) == 1

    def test_empty_failing_list_returns_empty(self):
        failures = find_flaky_failures([], self.quarantine)
        assert failures == []

    def test_mixed_flaky_and_real_failures(self):
        """Only flaky tests should appear in results, not genuinely broken ones."""
        failures = find_flaky_failures(
            ["test_truly_flaky", "test_stable_fail"], self.quarantine
        )
        names = [f["test_name"] for f in failures]
        assert "test_truly_flaky" in names
        assert "test_stable_fail" not in names


class TestBuildPrComment:

    def test_returns_none_for_empty_failures(self):
        assert build_pr_comment([]) is None

    def test_contains_test_name(self):
        failures = [{"test_name": "test_truly_flaky", "score": 0.92,
                     "fail_rate": 0.2, "same_commit_inconsistency": 1.0}]
        comment = build_pr_comment(failures)
        assert "test_truly_flaky" in comment

    def test_contains_flakiness_score(self):
        failures = [{"test_name": "test_truly_flaky", "score": 0.92,
                     "fail_rate": 0.2, "same_commit_inconsistency": 1.0}]
        comment = build_pr_comment(failures)
        assert "0.92" in comment

    def test_contains_bot_signature(self):
        failures = [{"test_name": "test_truly_flaky", "score": 0.92,
                     "fail_rate": 0.2, "same_commit_inconsistency": 1.0}]
        comment = build_pr_comment(failures)
        assert "FlakyGuard" in comment

    def test_contains_run_url_when_provided(self):
        failures = [{"test_name": "test_truly_flaky", "score": 0.92,
                     "fail_rate": 0.2, "same_commit_inconsistency": 1.0}]
        comment = build_pr_comment(failures, run_url="https://github.com/actions/runs/123")
        assert "https://github.com/actions/runs/123" in comment

    def test_multiple_failures_all_appear(self):
        failures = [
            {"test_name": "test_a", "score": 0.9, "fail_rate": 0.3, "same_commit_inconsistency": 1.0},
            {"test_name": "test_b", "score": 0.7, "fail_rate": 0.2, "same_commit_inconsistency": 0.8},
        ]
        comment = build_pr_comment(failures)
        assert "test_a" in comment
        assert "test_b" in comment
