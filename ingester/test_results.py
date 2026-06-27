"""
Parses JUnit XML test reports (the most common format CI test runners emit)
into flat per-test records: {test_name, status, duration_s, run_id, commit_sha}.

This is the bridge between raw GitHub Actions artifacts and the tabular data
the detector module needs. Most JVM-based projects (Kafka, Elasticsearch) emit
JUnit XML by default from their test runners (Gradle/Maven surefire), which is
why those repos are good ingestion targets.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    test_name: str
    classname: str
    status: str  # "passed" | "failed" | "skipped" | "error"
    duration_s: float
    run_id: int = None
    commit_sha: str = None


def parse_junit_xml(xml_path: str, run_id: int = None, commit_sha: str = None) -> list[TestResult]:
    """Parse a single JUnit XML file into a list of TestResult records."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # JUnit XML sometimes wraps <testsuite> elements in a <testsuites> root,
    # sometimes <testsuite> is the root itself -- handle both.
    suites = root.findall(".//testsuite") if root.tag != "testsuite" else [root]

    results = []
    for suite in suites:
        for case in suite.findall("testcase"):
            name = case.get("name", "unknown")
            classname = case.get("classname", "unknown")
            duration = float(case.get("time", 0.0))

            if case.find("failure") is not None:
                status = "failed"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            else:
                status = "passed"

            results.append(TestResult(
                test_name=name,
                classname=classname,
                status=status,
                duration_s=duration,
                run_id=run_id,
                commit_sha=commit_sha,
            ))
    return results


def results_to_records(results: list[TestResult]) -> list[dict]:
    """Convert TestResult dataclasses to plain dicts (for pandas/JSON)."""
    return [asdict(r) for r in results]


if __name__ == "__main__":
    # Smoke test with a tiny inline JUnit XML sample (no network needed).
    import tempfile, os

    sample_xml = """<?xml version="1.0"?>
    <testsuite name="ExampleSuite" tests="3">
        <testcase classname="com.example.FooTest" name="testAdd" time="0.012"/>
        <testcase classname="com.example.FooTest" name="testSubtract" time="0.008">
            <failure message="assertion failed">stack trace here</failure>
        </testcase>
        <testcase classname="com.example.FooTest" name="testSkippedThing" time="0.0">
            <skipped/>
        </testcase>
    </testsuite>
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(sample_xml)
        path = f.name

    try:
        parsed = parse_junit_xml(path, run_id=123, commit_sha="abc123")
        for r in parsed:
            print(r)
    finally:
        os.remove(path)
