"""
Reads models/metrics.json (written by train.py) and enforces a minimum
quality bar. Exits non-zero if the model doesn't meet it — this is what
turns "automated training" into an actual CI gate: a regression here should
block the build, not silently ship a worse model.

Thresholds are deliberately below what a well-tuned model achieves on this
dataset (typically F1 > 0.90) so normal variance across dataset refreshes or
minor code changes doesn't cause flaky failures. The gate exists to catch
real regressions (e.g. a broken preprocessing change, a bad hyperparameter),
not to enforce state-of-the-art performance.
"""

import argparse
import json
import sys
from pathlib import Path

METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "metrics.json"

# Minimum acceptable values. Recall is weighted more heavily than precision
# for spam detection — a missed spam message (false negative) is worse than
# an occasional false positive, since users can tolerate the rare legit
# message flagged, but a spam getting through defeats the point.
THRESHOLDS = {
    "f1": 0.80,
    "recall": 0.75,
    "precision": 0.75,
}


def evaluate(metrics_path: Path, thresholds: dict) -> bool:
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found at {metrics_path}", file=sys.stderr)
        return False

    with metrics_path.open() as f:
        metrics = json.load(f)

    passed = True
    for key, min_value in thresholds.items():
        actual = metrics.get(key)
        if actual is None:
            print(f"ERROR: metric '{key}' missing from metrics file", file=sys.stderr)
            passed = False
            continue
        status = "PASS" if actual >= min_value else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"{status}: {key} = {actual:.4f} (threshold: {min_value})")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    ok = evaluate(args.metrics, THRESHOLDS)
    if not ok:
        print("\nModel evaluation FAILED — blocking build.", file=sys.stderr)
        sys.exit(1)

    print("\nModel evaluation PASSED.")
