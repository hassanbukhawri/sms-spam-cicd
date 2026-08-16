"""
Smoke tests for the training pipeline and evaluation gate.

These deliberately train on a small synthetic dataset rather than the real
downloaded data — that keeps the test stage fast and independent of network
access / data fetch succeeding, while still exercising the real code path
(the same Pipeline class, the same fit/predict/predict_proba calls) that
train.py and serve.py depend on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from evaluate import evaluate  # noqa: E402
from train import build_pipeline  # noqa: E402

SYNTHETIC_TEXTS = [
    "hey are we still on for lunch",
    "call me when you get a chance",
    "win a free prize click here now",
    "urgent claim your cash reward today",
    "see you tomorrow at the office",
    "congratulations you won a free vacation call now",
    "can you send me the report",
    "limited time offer act now to win",
] * 5  # repeat so TF-IDF has enough vocabulary overlap to fit without error

SYNTHETIC_LABELS = [0, 0, 1, 1, 0, 1, 0, 1] * 5


def test_pipeline_fits_and_predicts():
    pipeline = build_pipeline()
    pipeline.fit(SYNTHETIC_TEXTS, SYNTHETIC_LABELS)

    predictions = pipeline.predict(SYNTHETIC_TEXTS[:2])
    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})


def test_pipeline_predict_proba_is_valid_distribution():
    pipeline = build_pipeline()
    pipeline.fit(SYNTHETIC_TEXTS, SYNTHETIC_LABELS)

    proba = pipeline.predict_proba(["free cash prize click now"])[0]
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_evaluate_passes_when_metrics_above_threshold(tmp_path):
    import json

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"f1": 0.9, "precision": 0.9, "recall": 0.9}))

    result = evaluate(metrics_path, {"f1": 0.8, "precision": 0.75, "recall": 0.75})
    assert result is True


def test_evaluate_fails_when_metrics_below_threshold(tmp_path):
    import json

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"f1": 0.5, "precision": 0.5, "recall": 0.5}))

    result = evaluate(metrics_path, {"f1": 0.8, "precision": 0.75, "recall": 0.75})
    assert result is False


def test_evaluate_fails_when_metrics_file_missing(tmp_path):
    result = evaluate(tmp_path / "does_not_exist.json", {"f1": 0.8})
    assert result is False
