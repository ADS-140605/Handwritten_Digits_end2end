import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate import evaluate
from model import MNISTCNN
from train import train

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_metrics(output_dir, history):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
        )

        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["test_loss"][i],
                    history["test_acc"][i],
                ]
            )

    if plt is None:
        print("matplotlib not installed, skipping curve image generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )
    axes[0].plot(
        history["epoch"],
        history["test_loss"],
        marker="o",
        label="Test Loss",
    )
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history["epoch"],
        history["train_acc"],
        marker="o",
        label="Train Accuracy",
    )
    axes[1].plot(
        history["epoch"],
        history["test_acc"],
        marker="o",
        label="Test Accuracy",
    )
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150)

    plt.close(fig)

    print(f"Saved training curves to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Digit Recognition Training"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("data", "mnist"),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )

    parser.add_argument("--download", action="store_true")

    parser.add_argument(
        "--model-path",
        type=str,
        default= "backend\\storage\\models\\original\\mnist_cnn.pt"
    )
    print(f"Default model path: {parser.get_default('model_path')}")

    parser.add_argument("--save-model", action="store_true")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    print(f"Default model path: {parser.get_default('model_path')}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=args.download,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        args.data_dir,
        train=False,
        download=args.download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = MNISTCNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0.0

    if os.path.exists(args.model_path):
        print(f"Loading existing checkpoint: {args.model_path}")

        model.load_state_dict(
            torch.load(
                args.model_path,
                map_location=device,
            )
        )

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        train_loss, train_acc = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch,
        )

        test_loss, test_acc = evaluate(
            model,
            device,
            test_loader,
            criterion,
        )

        scheduler.step(test_acc)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

            if args.save_model:
                os.makedirs(
                    os.path.dirname(args.model_path),
                    exist_ok=True,
                )

                torch.save(
                    model.state_dict(),
                    args.model_path,
                )

                print(
                    f"Best model saved "
                    f"(Accuracy: {best_acc:.2f}%)"
                )

    save_metrics(args.output_dir, history)

    print(f"Training completed.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
