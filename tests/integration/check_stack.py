"""Probe actual HTTP prediction and persistence in the CI Compose network."""

import json
import os
from pathlib import Path

import requests

import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()
champion = client.get_model_version_by_alias("food11", "champion")
assert int(champion.version) >= 1
assert requests.get("http://mlflow:5000/health", timeout=10).status_code == 200
assert requests.get("http://frontend:8501/_stcore/health", timeout=10).status_code == 200
assert requests.get("http://inference:8000/health", timeout=10).json() == {"status": "ok"}
with Path("/fixtures/bread.jpg").open("rb") as image:
    response = requests.post("http://inference:8000/predict", files={"file": image}, timeout=60)
response.raise_for_status()
body = response.json()
assert isinstance(body["category"], str) and 0 <= body["confidence"] <= 1
print(json.dumps({"champion_version": champion.version, "prediction": body}))
