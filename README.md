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

Self-hosted end to end on an Oracle Cloud Always Free VM — Jenkins runs
in a Docker container on the VM, and the deployed inference API runs in
its own container alongside it.

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

## Infrastructure setup

This section documents how the hosting environment was actually built, so
it's reproducible from scratch rather than assuming Jenkins/Docker already
exist somewhere.

### 1. Oracle Cloud VM (Always Free tier)

- Shape: `VM.Standard.E2.1.Micro` (1 OCPU, 1GB RAM — the always-available
  free shape; `VM.Standard.A1.Flex` is also free and more powerful but
  subject to regional capacity limits)
- Image: Canonical Ubuntu 24.04
- Networking: a VCN created via Oracle's **"Create VCN with Internet
  Connectivity"** wizard (this auto-provisions a public subnet, internet
  gateway, and route table together — trying to configure these manually
  during instance creation is more error-prone)
- The instance's VNIC must be on the **public** subnet with "Automatically
  assign public IPv4 address" enabled, or it won't be reachable at all
- Security List: inbound TCP rules opened for port `8080` (Jenkins) and
  `8000` (the deployed inference API), source `0.0.0.0/0`

### 2. Swap space

With only 1GB RAM, Jenkins' JVM needs headroom:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Docker

```bash
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu   # log out and back in to apply
```

### 4. Jenkins, with Docker access ("Docker-outside-of-Docker")

Jenkins runs as a container, but pipeline stages need to run `docker build`
and `docker push` — which means the Jenkins container needs access to the
**host's** Docker daemon, not a nested Docker-in-Docker setup.

```bash
# Find the host's docker group GID
getent group docker
# → e.g. docker:x:112:ubuntu — the number you need is 112

docker run -d \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --group-add 112 \
  --name jenkins \
  --restart unless-stopped \
  jenkins/jenkins:lts

# The base image doesn't ship the docker CLI, only socket access —
# install it inside the container:
docker exec -u root jenkins apt-get update
docker exec -u root jenkins apt-get install -y docker.io python3-venv python3-pip

# Verify Jenkins can reach the host daemon:
docker exec jenkins docker ps
```

**Known limitation:** the installed `docker.io` / `python3-venv` packages
live in the container's writable layer, not the `jenkins_home` volume. If
this container is ever removed (not just stopped/restarted) and recreated,
these packages need reinstalling. A cleaner long-term fix is a custom
`FROM jenkins/jenkins:lts` image with these baked in — not done here since
the current container is long-running and this only bites on a full
teardown/rebuild.

**Common pitfall:** `--group-add` takes only the numeric GID (e.g. `112`),
not the full `getent` output string — passing the full string causes
Jenkins to fail to start with "Unable to find group ... no matching
entries in group file."

### 5. Jenkins pipeline job

- New Item → Pipeline
- Build Triggers → check **"GitHub hook trigger for GITScm polling"**
- Pipeline → Definition: **"Pipeline script from SCM"** → SCM: Git →
  Repository URL: this repo's HTTPS URL → Script Path: `Jenkinsfile`
  (default)

### 6. Docker Hub credentials in Jenkins

Manage Jenkins → Credentials → System → Global credentials → Add
Credentials:
- Kind: Username with password
- Username: Docker Hub username
- Password: a Docker Hub **access token** (Account Settings → Security →
  New Access Token), not the account password
- ID: `dockerhub-credentials` — must match exactly what the Jenkinsfile
  references

### 7. GitHub webhook

Repo → Settings → Webhooks → Add webhook:
- Payload URL: `http://<vm-public-ip>:8080/github-webhook/` (trailing
  slash required)
- Content type: `application/json`
- Events: Just the push event

Check "Recent Deliveries" for a green checkmark to confirm Jenkins is
reachable from GitHub before relying on it.

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
