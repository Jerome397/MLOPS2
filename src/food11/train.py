import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Evaluate model
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
# Main training function
# ---------------------------------------------------------

def main():

    args = parse_args()

    # -----------------------------------------------------
    # MLflow setup
    # -----------------------------------------------------

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Dataset path
    # -----------------------------------------------------

    if args.dataset == "mini":
        data_dir = Path("data/food11_processed_mini")
    else:
        data_dir = Path("data/food11_processed")

    train_dir = data_dir / "training"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "evaluation"

    # -----------------------------------------------------
    # Image transforms
    # -----------------------------------------------------

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        11
    )

    model = model.to(device)

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.lr
    )

    # -----------------------------------------------------
    # MLflow run
    # -----------------------------------------------------

    with mlflow.start_run() as run:

        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
            "classes": 11,
            "device": str(device)
        })

        # -------------------------------------------------
        # Training loop
        # -------------------------------------------------

        for epoch in range(args.epochs):

            model.train()

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

                running_loss += (
                    loss.item() * images.size(0)
                )

                total_samples += images.size(0)

            train_loss = (
                running_loss / total_samples
            )

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            # ---------------------------------------------
            # Log metrics to MLflow
            # ---------------------------------------------

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val accuracy: {val_accuracy:.4f}"
            )

        # -------------------------------------------------
        # Test evaluation
        # -------------------------------------------------

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_loss",
            test_loss
        )

        print(
            f"Test accuracy: {test_accuracy:.4f}"
        )

        # -------------------------------------------------
        # Log trained model
        # -------------------------------------------------

        model = model.cpu()

        mlflow.pytorch.log_model(
            model,
            name="model"
        )

        print("Model logged to MLflow.")


if __name__ == "__main__":
    main()