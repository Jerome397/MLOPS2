import mlflow
import mlflow.pytorch


# Load the model exported from your old MLflow server.
model = mlflow.pytorch.load_model(
    "lab4_model_export/model"
)


# Point MLflow to the NEW Dockerized tracking server.
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "food11"
)


# Log the existing trained model into the new server.
with mlflow.start_run() as run:

    mlflow.log_param(
        "source",
        "Lab 3 trained model migrated to Lab 4"
    )

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        serialization_format="pickle",
    )

    run_id = run.info.run_id


print(
    "New MLflow run ID:",
    run_id
)


# Register the model.
registered = mlflow.register_model(
    f"runs:/{run_id}/model",
    "food11",
)


print(
    "Registered model version:",
    registered.version
)


# Move that version to Staging.
client = mlflow.MlflowClient()

client.transition_model_version_stage(
    name="food11",
    version=registered.version,
    stage="Staging",
    archive_existing_versions=True,
)


print(
    f"food11 version "
    f"{registered.version} "
    f"moved to Staging."
)