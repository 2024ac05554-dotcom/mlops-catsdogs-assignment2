"""Unit tests for src/model.py and the FastAPI inference service in app/main.py"""
import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model


def test_model_output_shape():
    """Model utility test: forward pass should return logits for 2 classes."""
    model = build_model(num_classes=2)
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)  # batch of 4 RGB 224x224 images
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 2)


def test_model_is_deterministic_in_eval_mode():
    model = build_model(num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)
    assert torch.allclose(out1, out2)


def _make_test_image_bytes():
    img = Image.new("RGB", (100, 100), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    from app.main import app
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model_version" in body


def test_predict_endpoint_rejects_bad_content_type():
    from app.main import app
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("not_an_image.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400


def test_predict_endpoint_returns_valid_prediction():
    from app.main import app
    with TestClient(app) as client:
        img_bytes = _make_test_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        # 200 if a trained model.pt is present, 503 if it isn't (e.g. fresh CI checkout)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            body = response.json()
            assert body["label"] in ("cat", "dog")
            assert "cat" in body["probabilities"] and "dog" in body["probabilities"]
            assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_stats_endpoint_tracks_request_count():
    from app.main import app
    with TestClient(app) as client:
        before = client.get("/stats").json()["request_count"]
        client.get("/health")
        after = client.get("/stats").json()["request_count"]
        assert after > before
