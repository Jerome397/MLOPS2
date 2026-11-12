import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import mlflow.pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

import mlflow

# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on Food-11")

    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="processed",
        help="Use the full processed dataset or mini dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Train only the transfer-learning head"
    )
    return parser.parse_args()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 2:
        raise ValueError("epochs/lr must be positive; batch-size must be >= 2")
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # -----------------------------------------------------
    # MLflow configuration
    # -----------------------------------------------------

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    mlflow.set_experiment("food11")

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Dataset path
    # -----------------------------------------------------

    if args.dataset == "processed":
        data_dir = Path("data/food11_processed")
    else:
        data_dir = Path("data/food11_processed_mini")

    train_dir = data_dir / "training"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "evaluation"

    # -----------------------------------------------------
    # Check folders
    # -----------------------------------------------------

    for folder in [
        train_dir,
        val_dir,
        test_dir,
    ]:
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

    # -----------------------------------------------------
    # Image transforms
    # -----------------------------------------------------

    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    # -----------------------------------------------------
    # ImageFolder datasets
    # -----------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
    )

    print(f"Classes: {train_dataset.classes}")

    print(f"Training images: {len(train_dataset)}")

    print(f"Validation images: {len(val_dataset)}")

    print(f"Test images: {len(test_dataset)}")

    # -----------------------------------------------------
    # Verify Food-11 classes
    # -----------------------------------------------------

    if len(train_dataset.classes) != 11:
        raise RuntimeError(
            "Expected 11 Food-11 classes, "
            f"but found {len(train_dataset.classes)}: "
            f"{train_dataset.classes}"
        )

    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError("Training and validation classes do not match.")

    if train_dataset.classes != test_dataset.classes:
        raise RuntimeError("Training and evaluation classes do not match.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food11.validate_data import (
        check_distribution,
        check_duplicates,
        check_folders,
        check_images,
        check_minimum,
    )

    checks = [
        check_folders(data_dir),
        check_images(data_dir),
        check_minimum(data_dir, 50),
        check_duplicates(data_dir),
        check_distribution(data_dir, 0.05),
    ]
    if not all(checks):
        raise RuntimeError("Dataset contract failed; training aborted")
    fingerprint = hashlib.sha256()
    for path in sorted(data_dir.rglob("*.jpg")):
        fingerprint.update(str(path.relative_to(data_dir)).encode())
        fingerprint.update(path.read_bytes())
    dataset_hash = fingerprint.hexdigest()

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------
    # ResNet18
    # -----------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    number_of_features = model.fc.in_features

    # ResNet18 normally has 1000 outputs.
    # Food-11 requires 11 outputs.
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(
        number_of_features,
        11,
    )

    model = model.to(device)

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------------------------------------
    # MLflow run
    # -----------------------------------------------------

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # -------------------------------------------------
        # Log parameters
        # -------------------------------------------------

        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "classes": 11,
                "test_size": len(test_dataset),
                "dataset_hash": dataset_hash,
                "seed": 42,
                "freeze_backbone": args.freeze_backbone,
                "class_names": json.dumps(train_dataset.classes),
                "device": str(device),
            }
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        for epoch in range(args.epochs):
            model.train()
            if args.freeze_backbone:
                model.eval()
                model.fc.train()

            running_loss = 0.0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item() * images.size(0)

                total_samples += images.size(0)

            train_loss = running_loss / total_samples

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            # ---------------------------------------------
            # Log metrics
            # ---------------------------------------------

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{args.epochs} | "
                f"Train loss: "
                f"{train_loss:.4f} | "
                f"Val loss: "
                f"{val_loss:.4f} | "
                f"Val accuracy: "
                f"{val_accuracy:.4f}"
            )

        # -------------------------------------------------
        # Final test evaluation
        # -------------------------------------------------

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        mlflow.log_metric(
            "test_loss",
            test_loss,
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        print(f"Test loss: {test_loss:.4f}")

        print(f"Test accuracy: {test_accuracy:.4f}")

        # -------------------------------------------------
        # Log trained model
        # -------------------------------------------------

        model = model.cpu()
        model.eval()

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            serialization_format="pickle",
        )

        mlflow.log_dict({"classes": train_dataset.classes}, "class_names.json")

        print("Model logged to MLflow successfully.")

        print(f"Finished MLflow run: {run.info.run_id}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
