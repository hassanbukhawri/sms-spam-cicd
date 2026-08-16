# Trained model is expected to already exist at models/model.joblib —
# training happens as its own CI pipeline stage (with an evaluation gate)
# before this image is built, so a bad model never gets baked into an image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]
