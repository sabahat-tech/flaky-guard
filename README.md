# FlakyGuard — Self-Healing CI/CD Pipeline with Flaky Test Detection

Final year project: a GitHub Actions add-on that automatically detects flaky tests,
quarantines them, and auto-bisects genuine regressions to the breaking commit.

## Problem statement

Flaky tests (tests that pass/fail nondeterministically, with no relation to code
changes) waste engineering time and erode trust in CI signals. Most teams handle
this manually — someone notices a test failing intermittently, investigates, and
eventually marks it as "known flaky." FlakyGuard automates this loop, and goes one
step further: when a *real* regression occurs, it automatically bisects history to
find the breaking commit instead of leaving that to a human.

## Architecture

```
                  ┌─────────────────┐
   GitHub Actions │                 │
   webhook events ├──> ingester ───>│  data/ (run history, labeled ground truth)
                  │                 │
                  └─────────────────┘
                           │
                           v
                  ┌─────────────────┐
                  │    detector     │  statistical flakiness scoring per test
                  └─────────────────┘
                           │
                  ┌────────┴────────┐
                  v                 v
          ┌──────────────┐  ┌──────────────┐
          │  bisector     │  │  dashboard    │
          │ (real fails)  │  │ (flaky trends)│
          └──────────────┘  └──────────────┘
```

## Module breakdown (team split)

| Module | Owner | Description |
|---|---|---|
| `ingester/` | Person A | Pulls workflow run + per-test results from GitHub Actions API; mines GitHub issues for ground-truth flaky labels |
| `detector/` | Person B | Statistical/ML model scoring each test's flakiness probability |
| `bisector/` | Person C | Binary-search across commits to localize a genuine regression |
| `dashboard/` | Shared | Lightweight web UI showing flaky trends, quarantine list, bisection results |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set a GitHub token (classic PAT with repo:read scope is enough for public repos)
export GITHUB_TOKEN=ghp_xxx
```

## Phase plan

- **Phase 0 (week 1):** scope lock — pick target repos, write spec ✅ (this README)
- **Phase 1 (weeks 2-3):** ingester — pull run history + mine ground truth
- **Phase 2 (weeks 3-5):** detector — flakiness scoring model
- **Phase 3 (weeks 5-7):** bisector — auto-bisect on real failures
- **Phase 4 (weeks 7-9):** integration — GitHub bot + dashboard
- **Phase 5 (weeks 9-12):** evaluation — precision/recall vs naive baselines, writeup

## Target repos for ground-truth mining

Good candidates (large, active, with test-flakiness issue history):
- `apache/kafka`
- `elastic/elasticsearch`
- `pytorch/pytorch`

These get configured in `ingester/config.py`.

## Evaluation metrics

- Detector: precision / recall against labeled flaky tests (mined from issues)
- Bisector: bisection accuracy (does it find the actual breaking commit?) and number
  of CI runs needed vs naive linear search
- End-to-end: estimated engineer-hours saved vs manual triage (back-of-envelope,
  based on how often flaky failures currently block merges in sampled repos)
