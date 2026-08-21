"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
  GET  /health   -> health check
  POST /predict  -> accepts an image file, returns class label + probabilities
  GET  /metrics  -> Prometheus metrics (request count, latency, etc.)

Also implements:
  - request/response logging (excluding raw image bytes / any sensitive data)
  - basic in-app request counters as a fallback if Prometheus isn't scraping
"""
from __future__ import annotations

import io
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import HealthResponse, PredictionResponse
from src.data_preprocessing import get_transforms, CLASSES
from src.model import build_model

# ---------------------------------------------------------------------------
# Logging setup (structured, no sensitive/raw image data logged)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("catsdogs-inference")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pt")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")

# Simple in-app counters as a lightweight fallback/complement to Prometheus
_metrics = defaultdict(int)
_transform = get_transforms(train=False)

_model = None


def _load_model():
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at %s; /predict will return 503", MODEL_PATH)
        return None
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    logger.info("Model loaded from %s (version=%s)", MODEL_PATH, MODEL_VERSION)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    yield


app = FastAPI(title="Cats vs Dogs Inference Service", version=MODEL_VERSION, lifespan=lifespan)

# Prometheus metrics: request count, latency histogram, etc. exposed at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    _metrics["request_count"] += 1
    _metrics["total_latency_ms"] += duration_ms
    # Log method/path/status/latency only -- never request bodies / image bytes.
    logger.info(
        "%s %s -> %s (%.2fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness probe target for Docker/K8s and the CI/CD smoke tests."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
    )


@app.get("/stats")
def stats():
    """Basic monitoring counters (M5): request count + average latency."""
    count = _metrics["request_count"]
    avg_latency = (_metrics["total_latency_ms"] / count) if count else 0.0
    return {"request_count": count, "avg_latency_ms": round(avg_latency, 2)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image")

    start = time.time()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    pred_idx = int(torch.tensor(probs).argmax())
    latency_ms = (time.time() - start) * 1000

    return PredictionResponse(
        label=CLASSES[pred_idx][:-1],  # "cats" -> "cat"
        probabilities={CLASSES[i][:-1]: round(p, 4) for i, p in enumerate(probs)},
        latency_ms=round(latency_ms, 2),
    )
