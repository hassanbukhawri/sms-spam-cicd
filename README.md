# SMS Spam Classifier — CI/CD Pipeline

A small ML model with a fully automated Jenkins pipeline: on every push, it
trains a fresh model, gates the build on a minimum quality bar, and only
deploys if the gate passes. The model itself is intentionally simple —
the point of this project is the pipeline engineering around it, not the
model.

## Architecture

```
GitHub push
    │
    ▼ (webhook)
Jenkins pipeline
    │
    ├─ Install dependencies
    ├─ Run tests (pytest)
    ├─ Fetch data
    ├─ Train model (TF-IDF + Logistic Regression)
    ├─ Evaluate model  ──── FAILS BUILD if F1 < 0.80 / recall < 0.75 / precision < 0.75
    ├─ Build Docker image
    ├─ Push to Docker Hub
    └─ Deploy (restart container)
```

## Why these choices

**TF-IDF + Logistic Regression over Naive Bayes.** Naive Bayes is the
textbook baseline for spam detection, but its independence assumption
doesn't hold for text. Logistic Regression on TF-IDF features calibrates
better and gave F1 ≈ 0.91 on held-out data.

**The evaluation gate is the actual CI contribution.** Automating training
isn't CI — automated *rejection of regressions* is. `evaluate.py` exits
non-zero if metrics fall under threshold, which stops a bad model before it
ever reaches the image-build stage. Thresholds are set below what a
well-tuned model achieves (~0.91 F1) so normal variance doesn't cause flaky
failures — the gate exists to catch real regressions, not to enforce
state-of-the-art performance.

**Recall weighted over precision in the gate.** A missed spam message
(false negative) is worse than an occasional false positive — users
tolerate a rare legitimate message flagged, but spam getting through
defeats the point.

**Smoke tests use synthetic data, not the real dataset.** `test_model_smoke.py`
trains on a small synthetic dataset rather than the downloaded one. This
keeps the test stage fast and independent of network access — a test suite
that depends on an external download succeeding is a flaky test suite.

**Data fetched at pipeline-run time, not committed.** Keeps the repo small
and makes the data source explicit and reproducible.

## Jenkins Docker access

Jenkins runs in its own container, so building/pushing images and
controlling the app container requires giving the Jenkins container access
to the host's Docker daemon (mounting `/var/run/docker.sock` + installing
the `docker` CLI inside the Jenkins container). See setup notes in the repo
issues/wiki for the exact commands used on this deployment.

## Running locally

```bash
pip install -r requirements.txt
python app/fetch_data.py
python app/train.py
python app/evaluate.py
uvicorn app.serve:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WIN a free prize! Click now"}'
```

## Running tests

```bash
pytest tests/ -v
```

## Stack

Python, scikit-learn, FastAPI, pytest, Docker, Jenkins, Docker Hub, deployed
on an Oracle Cloud Always Free ARM/E2 VM.
