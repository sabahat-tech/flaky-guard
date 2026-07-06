"""
Unit tests for ingester/test_results.py (JUnit XML parser).
Tests cover: basic parsing, status detection, parameterized tests,
missing attributes, and both <testsuite> root and <testsuites> wrapper formats.
"""
import io
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ingester.test_results import parse_junit_xml, results_to_records


def write_xml(content: str) -> str:
    """Write XML content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestParseJUnitXml:

    def test_basic_passed_test(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.example.Foo" name="testAdd" time="0.1"/>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert len(results) == 1
        assert results[0].test_name == "testAdd"
        assert results[0].classname == "com.example.Foo"
        assert results[0].status == "passed"
        assert results[0].duration_s == pytest.approx(0.1)
        os.unlink(path)

    def test_failed_test(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="testBad" time="0.2">
            <failure message="assert failed">stack trace</failure>
          </testcase>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert results[0].status == "failed"
        os.unlink(path)

    def test_error_test(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="testErr" time="0.0">
            <error message="NullPointerException"/>
          </testcase>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert results[0].status == "error"
        os.unlink(path)

    def test_skipped_test(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="testSkip" time="0.0">
            <skipped/>
          </testcase>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert results[0].status == "skipped"
        os.unlink(path)

    def test_multiple_tests(self):
        path = write_xml("""
        <testsuite name="S" tests="3">
          <testcase classname="com.x.A" name="t1" time="0.1"/>
          <testcase classname="com.x.A" name="t2" time="0.2">
            <failure/>
          </testcase>
          <testcase classname="com.x.A" name="t3" time="0.3">
            <skipped/>
          </testcase>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert len(results) == 3
        statuses = [r.status for r in results]
        assert statuses == ["passed", "failed", "skipped"]
        os.unlink(path)

    def test_testsuites_wrapper(self):
        """Handles <testsuites> root wrapping multiple <testsuite> elements."""
        path = write_xml("""
        <testsuites>
          <testsuite name="S1"><testcase classname="A" name="t1" time="0.1"/></testsuite>
          <testsuite name="S2"><testcase classname="B" name="t2" time="0.2"/></testsuite>
        </testsuites>""")
        results = parse_junit_xml(path)
        assert len(results) == 2
        assert results[0].classname == "A"
        assert results[1].classname == "B"
        os.unlink(path)

    def test_run_id_and_commit_sha_attached(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="t1" time="0.1"/>
        </testsuite>""")
        results = parse_junit_xml(path, run_id=999, commit_sha="abc123")
        assert results[0].run_id == 999
        assert results[0].commit_sha == "abc123"
        os.unlink(path)

    def test_missing_time_defaults_to_zero(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="t1"/>
        </testsuite>""")
        results = parse_junit_xml(path)
        assert results[0].duration_s == 0.0
        os.unlink(path)

    def test_results_to_records_returns_dicts(self):
        path = write_xml("""
        <testsuite name="S" tests="1">
          <testcase classname="com.x.A" name="t1" time="0.5"/>
        </testsuite>""")
        results = parse_junit_xml(path)
        records = results_to_records(results)
        assert isinstance(records, list)
        assert isinstance(records[0], dict)
        assert "test_name" in records[0]
        assert "status" in records[0]
        os.unlink(path)
