"""
FlakyGuard Dashboard — Flask backend.

Serves four pages:
  GET /           → main dashboard: ranked flaky tests + quarantine list
  GET /bisect     → bisection history + run a new bisection
  GET /ml         → ML vs heuristic side-by-side comparison
  POST /api/bisect → trigger a bisection and return results as JSON

Run with:
    python -m dashboard.app

Then open http://localhost:5000 in your browser.
"""
import os
import json
import sys
from flask import Flask, render_template, request, jsonify

# Add parent dir to path so we can import detector/bisector modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detector import features as feat
from detector import scorer as sc
from bisector import bisect_engine as be

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCORES_CSV = os.path.join(DATA_DIR, "test_results_flakiness_scores.csv")
RAW_CSV = os.path.join(DATA_DIR, "test_results.csv")
ML_SCORES_CSV = os.path.join(DATA_DIR, "ml_scores.csv")
COMPARISON_CSV = os.path.join(DATA_DIR, "comparison_report.csv")
BISECT_LOG = os.path.join(DATA_DIR, "bisect_history.json")

FLAKY_THRESHOLD = 0.3


def load_scores():
    """Load flakiness scores CSV, regenerating from raw CSV if needed."""
    if not os.path.exists(SCORES_CSV):
        if not os.path.exists(RAW_CSV):
            return []
        df = feat.load_results(RAW_CSV)
        features_df = feat.compute_features(df)
        scored = sc.score_tests(features_df)
        scored.to_csv(SCORES_CSV, index=False)

    import pandas as pd
    df = pd.read_csv(SCORES_CSV)
    return df.to_dict(orient="records")


def load_ml_comparison():
    """Load side-by-side comparison of heuristic vs ML scores."""
    import pandas as pd

    # Load heuristic scores
    heuristic_scores = load_scores()
    if not heuristic_scores:
        return [], False

    heuristic_df = pd.DataFrame(heuristic_scores)[
        ["classname", "test_name", "flakiness_score",
         "evidence_level", "fail_rate", "same_commit_inconsistency"]
    ]

    # Load ML scores if available
    ml_available = os.path.exists(ML_SCORES_CSV)
    if ml_available:
        ml_df = pd.read_csv(ML_SCORES_CSV)[
            ["classname", "test_name", "ml_flakiness_score"]
        ]
        merged = heuristic_df.merge(ml_df, on=["classname", "test_name"], how="left")
        merged["ml_flakiness_score"] = merged["ml_flakiness_score"].fillna(0)
    else:
        merged = heuristic_df.copy()
        merged["ml_flakiness_score"] = None

    merged["heuristic_flag"] = (
        (merged["flakiness_score"].astype(float) >= FLAKY_THRESHOLD) &
        (merged["evidence_level"] == "rerun_observed")
    )
    merged["ml_flag"] = merged["ml_flakiness_score"].apply(
        lambda x: float(x) >= FLAKY_THRESHOLD if x is not None else False
    )
    merged["agreement"] = merged.apply(
        lambda r: "agree" if r["heuristic_flag"] == r["ml_flag"] else "disagree",
        axis=1
    )
    merged = merged.sort_values("flakiness_score", ascending=False)
    return merged.to_dict(orient="records"), ml_available


def load_bisect_history():
    if not os.path.exists(BISECT_LOG):
        return []
    with open(BISECT_LOG) as f:
        return json.load(f)


def save_bisect_history(history):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BISECT_LOG, "w") as f:
        json.dump(history, f, indent=2)


@app.route("/")
def index():
    scores = load_scores()
    flagged = [r for r in scores
               if r.get("evidence_level") == "rerun_observed"
               and float(r.get("flakiness_score", 0)) >= FLAKY_THRESHOLD]
    flagged.sort(key=lambda r: float(r.get("flakiness_score", 0)), reverse=True)
    all_scores = sorted(scores,
                        key=lambda r: float(r.get("flakiness_score", 0)),
                        reverse=True)
    return render_template("index.html",
                           flagged=flagged,
                           all_scores=all_scores[:50],
                           threshold=FLAKY_THRESHOLD,
                           total_tests=len(scores))


@app.route("/ml")
def ml_page():
    comparison, ml_available = load_ml_comparison()
    agree_count = sum(1 for r in comparison if r.get("agreement") == "agree")
    disagree_count = sum(1 for r in comparison if r.get("agreement") == "disagree")
    heuristic_flagged = sum(1 for r in comparison if r.get("heuristic_flag"))
    ml_flagged = sum(1 for r in comparison if r.get("ml_flag"))
    feature_importances = [
        {"feature": "same_commit_inconsistency", "importance": 0.487, "heuristic_weight": 0.70},
        {"feature": "fail_rate",                 "importance": 0.361, "heuristic_weight": 0.30},
        {"feature": "duration_cv",               "importance": 0.152, "heuristic_weight": 0.10},
        {"feature": "total_runs",                "importance": 0.000, "heuristic_weight": 0.00},
        {"feature": "distinct_commits",          "importance": 0.000, "heuristic_weight": 0.00},
        {"feature": "reran_commits",             "importance": 0.000, "heuristic_weight": 0.00},
    ]
    return render_template("ml.html",
                           comparison=comparison,
                           ml_available=ml_available,
                           agree_count=agree_count,
                           disagree_count=disagree_count,
                           heuristic_flagged=heuristic_flagged,
                           ml_flagged=ml_flagged,
                           threshold=FLAKY_THRESHOLD,
                           feature_importances=feature_importances)


@app.route("/bisect")
def bisect_page():
    history = load_bisect_history()
    return render_template("bisect.html", history=history)


@app.route("/api/bisect", methods=["POST"])
def run_bisect():
    data = request.json
    repo_path = data.get("repo_path", "")
    good_sha = data.get("good_sha", "")
    bad_sha = data.get("bad_sha", "")
    test_command = data.get("test_command", "").split()

    if not all([repo_path, good_sha, bad_sha, test_command]):
        return jsonify({"error": "Missing required fields"}), 400

    steps = []
    def progress(commit, outcome, n, output):
        steps.append({"commit": commit[:8], "outcome": outcome, "output": output[:500]})

    try:
        result = be.bisect(repo_path, good_sha, bad_sha, test_command,
                           progress_callback=progress)
        entry = {
            "repo_path": repo_path,
            "good_sha": good_sha[:8],
            "bad_sha": bad_sha[:8],
            "culprit": result.culprit_commit[:8] if result.culprit_commit else None,
            "commits_checked": result.total_commits_checked,
            "steps": steps,
        }
        history = load_bisect_history()
        history.insert(0, entry)
        save_bisect_history(history[:20])
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
