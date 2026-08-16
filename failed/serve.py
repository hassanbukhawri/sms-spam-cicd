"""
Minimal inference API for the spam classifier. Loads the trained pipeline
once at startup (not per-request) for latency.
"""

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_text  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

app = FastAPI(title="SMS Spam Classifier")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    spam_probability: float


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    model = get_model()
    cleaned = clean_text(request.text)
    proba = model.predict_proba([cleaned])[0]
    spam_idx = list(model.classes_).index(1)
    spam_probability = float(proba[spam_idx])
    label = "spam" if spam_probability >= 0.5 else "ham"

    return PredictResponse(label=label, spam_probability=spam_probability)
