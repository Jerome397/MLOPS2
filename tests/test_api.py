import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, x):
        assert x.shape == (1, 3, 128, 128)
        assert x.dtype == np.float32
        return [[4.5] + [0.0] * 10]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        yield client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_requires_model(client):
    assert client.post("/predict", files={"file": ("bad.jpg", b"bad")}).status_code == 503


def test_predict_returns_valid_json(client, monkeypatch):
    monkeypatch.setattr(serve, "model", FakeModel())
    with open("tests/fixtures/bread.jpg", "rb") as image:
        response = client.post("/predict", files={"file": ("bread.jpg", image, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Bread"
    assert body["confidence"] == pytest.approx(np.exp(4.5) / (np.exp(4.5) + 10))


def test_invalid_image(client, monkeypatch):
    monkeypatch.setattr(serve, "model", FakeModel())
    assert client.post("/predict", files={"file": ("bad.jpg", b"bad")}).status_code == 400
