"""Copy a trained champion into a fresh Compose registry, preserving provenance."""

import argparse
import sys

import mlflow.pytorch

import mlflow


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--destination-uri", default="http://127.0.0.1:5001")
    args = parser.parse_args()
    if args.source_uri == args.destination_uri:
        raise ValueError("Source and destination must differ")
    mlflow.set_tracking_uri(args.source_uri)
    client = mlflow.MlflowClient()
    source = client.get_model_version_by_alias("food11", "champion")
    run = client.get_run(source.run_id)
    model = mlflow.pytorch.load_model(source.source)
    mlflow.set_tracking_uri(args.destination_uri)
    mlflow.set_experiment("food11")
    with mlflow.start_run():
        mlflow.log_params(dict(run.data.params))
        mlflow.log_metrics(dict(run.data.metrics))
        mlflow.set_tags(
            {
                "source_run_id": source.run_id,
                "source_model_version": source.version,
                "source_tracking_uri": args.source_uri,
                "purpose": "trained champion migration",
            }
        )
        info = mlflow.pytorch.log_model(
            model.cpu().eval(), name="model", serialization_format="pickle"
        )
    version = mlflow.register_model(info.model_uri, "food11")
    mlflow.MlflowClient().set_registered_model_alias("food11", "champion", version.version)
    print(f"Migrated trained champion to version {version.version} at {args.destination_uri}")


if __name__ == "__main__":
    main()
