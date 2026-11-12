"""Synthetic wiring smoke model only: no real training or quality claims."""

import os

import mlflow.pytorch
from torch import nn
from torchvision.models import resnet18

import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("compose-smoke")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 11)
model.eval()
with mlflow.start_run():
    info = mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle",
        pip_requirements=["torch", "torchvision"],
    )
    mlflow.set_tag("purpose", "synthetic compose wiring smoke, not a trained classifier")
version = mlflow.register_model(info.model_uri, "food11")
mlflow.MlflowClient().set_registered_model_alias("food11", "champion", version.version)
print("Synthetic smoke champion:", version.version)
