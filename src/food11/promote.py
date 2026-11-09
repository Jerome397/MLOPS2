import argparse
import math
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "food11"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def count_test_images(
    test_dir: Path,
) -> int:

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    return sum(
        1
        for path
        in test_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        )
    )


def get_accuracy(
    client: MlflowClient,
    run_id: str,
) -> float:

    run = client.get_run(
        run_id
    )

    metrics = (
        run.data.metrics
    )

    if "test_accuracy" not in metrics:

        raise RuntimeError(
            "Run does not contain "
            "'test_accuracy'"
        )

    return float(
        metrics[
            "test_accuracy"
        ]
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help=(
            "Candidate registered "
            "model version"
        ),
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(
            "data/food11_processed/evaluation"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    candidate = (
        client.get_model_version(
            name=MODEL_NAME,
            version=str(
                args.version
            ),
        )
    )

    champion = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "champion",
        )
    )

    candidate_accuracy = (
        get_accuracy(
            client,
            candidate.run_id,
        )
    )

    champion_accuracy = (
        get_accuracy(
            client,
            champion.run_id,
        )
    )

    n = count_test_images(
        args.test_dir
    )

    if n <= 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    p = champion_accuracy

    standard_error = math.sqrt(
        p * (1 - p) / n
    )

    margin = (
        2 * standard_error
    )

    threshold = (
        champion_accuracy
        + margin
    )

    if (
        candidate_accuracy
        >= threshold
    ):

        decision = "promoted_to_champion"

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            str(args.version),
        )

        print(
            "Candidate promoted "
            "to champion."
        )

    else:

        decision = "set_as_challenger"

        client.set_registered_model_alias(
            MODEL_NAME,
            "challenger",
            str(args.version),
        )

        print(
            "Candidate did not beat "
            "the quality gate."
        )

        print(
            "Candidate set as challenger."
        )

    tags = {
        "promotion_decision":
            decision,

        "candidate_test_accuracy":
            f"{candidate_accuracy:.6f}",

        "champion_test_accuracy":
            f"{champion_accuracy:.6f}",

        "test_size":
            str(n),

        "two_standard_errors":
            f"{margin:.6f}",

        "required_accuracy":
            f"{threshold:.6f}",
    }

    for key, value in tags.items():

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(
                args.version
            ),
            key=key,
            value=value,
        )

    print()
    print(
        f"Candidate accuracy: "
        f"{candidate_accuracy:.4f}"
    )

    print(
        f"Champion accuracy: "
        f"{champion_accuracy:.4f}"
    )

    print(
        f"Test size: {n}"
    )

    print(
        f"2 * SE: {margin:.4f}"
    )

    print(
        f"Required accuracy: "
        f"{threshold:.4f}"
    )

    print(
        f"Decision: {decision}"
    )


if __name__ == "__main__":
    main()