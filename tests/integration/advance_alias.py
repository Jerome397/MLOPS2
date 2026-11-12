"""Move the smoke alias to a new version before restarting inference."""

import os

import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()
old = client.get_model_version_by_alias("food11", "champion")
new = mlflow.register_model(old.source, "food11")
client.set_registered_model_alias("food11", "champion", new.version)
assert int(new.version) > int(old.version)
print(f"Champion moved from {old.version} to {new.version}")
