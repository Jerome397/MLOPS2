"""Register an explicitly selected run and optionally initialize champion."""

import argparse
import os

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--champion", action="store_true")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    version = mlflow.register_model(f"runs:/{args.run_id}/model", "food11")
    if args.champion:
        mlflow.MlflowClient().set_registered_model_alias("food11", "champion", version.version)
    print(f"Registered version {version.version}")


if __name__ == "__main__":
    main()
