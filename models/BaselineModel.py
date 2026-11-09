"""Train and evaluate matched four-class ECG heartbeat models.

Loss: plain cross-entropy by default. With ``--class-weights`` the *training*
loss is weighted by inverse class frequency, computed from the training labels
only (weight_c = N_train / (num_classes * count_c)). Validation loss and every
reported metric stay unweighted so runs remain comparable. Figures are named
with the run tag, so use a distinct ``--run-tag`` (e.g. ``prelim_weighted``) for
weighted runs.

Models
------
* ``cnn``    - Compact 1D CNN control.
* ``hybrid`` - Same CNN backbone followed by a Transformer encoder.

Protocol fixes:
* four classes (N/S/V/F) and 360-sample one-second windows,
* example-weighted epoch loss,
* fixed four-class macro-F1 (undefined classes count as zero and are flagged),
* best validation-macro-F1 checkpoint restored before any test evaluation,
* test evaluation is opt-in via ``--run-test``,
* configuration, split hash, history, checkpoint, predictions and metrics saved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    WINDOW_SIZE,
    class_counts,
    sha256_file,
    validate_feature_matrix,
    validate_labels,
    write_json,
)

class_list = list(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.sequential(x)


class CNNBackbone(nn.Module):
    """Shared convolutional backbone for both matched models."""

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            ConvBlock(1, 32, kernel_size=11),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3),
        )

    def forward(self, x):
        return self.sequential(x)


class CompactCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_prob=0.2):
        super().__init__()
        self.backbone = CNNBackbone()
        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_transformation = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.average_pooling(features)
        pooled = torch.flatten(pooled, start_dim=1)
        pooled = self.dropout(pooled)
        return self.linear_transformation(pooled)


class CompactCNNTransformer(nn.Module):
    """CNN backbone plus a Transformer encoder and the same linear head."""

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dropout_prob=0.2,
        d_model=128,
        nhead=4,
        num_layers=1,
    ):
        super().__init__()
        self.backbone = CNNBackbone()
        sequence_length = WINDOW_SIZE // 8  # three stride-2 pooling stages
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_transformation = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (batch, 128, 45)
        sequence = features.transpose(1, 2)  # (batch, 45, 128)
        sequence = sequence + self.positional_embedding
        sequence = self.encoder(sequence)
        pooled = sequence.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.linear_transformation(pooled)


def build_model(model_name: str, num_classes: int = NUM_CLASSES, dropout_prob: float = 0.2):
    if model_name == "cnn":
        return CompactCNN(num_classes=num_classes, dropout_prob=dropout_prob)
    if model_name == "hybrid":
        return CompactCNNTransformer(num_classes=num_classes, dropout_prob=dropout_prob)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class BeatDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        window = torch.tensor(self.windows[i], dtype=torch.float32).unsqueeze(0)
        target = int(self.labels[i])
        return window, target


def make_loader(
    features,
    labels,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
):
    dataset = BeatDataset(features, labels)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics and loss
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, num_classes: int = NUM_CLASSES) -> dict:
    """Compute fixed-class metrics; absent classes contribute zero F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise ValueError("y_true contains labels outside the declared class range.")
    if y_pred.min() < 0 or y_pred.max() >= num_classes:
        raise ValueError("y_pred contains labels outside the declared class range.")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)

    per_class = []
    f1_values = []
    sensitivity_values = []

    for class_id in range(num_classes):
        true_positives = int(confusion[class_id, class_id])
        support = int(confusion[class_id, :].sum())
        number_predicted = int(confusion[:, class_id].sum())

        sensitivity = true_positives / support if support > 0 else 0.0
        precision = true_positives / number_predicted if number_predicted > 0 else 0.0
        f1 = (
            2.0 * sensitivity * precision / (sensitivity + precision)
            if sensitivity + precision > 0
            else 0.0
        )

        per_class.append(
            {
                "class": CLASS_NAMES[class_id],
                "support": support,
                "predicted": number_predicted,
                "sensitivity": float(sensitivity),
                "precision": float(precision),
                "f1": float(f1),
                "absent": support == 0,
            }
        )
        f1_values.append(f1)
        sensitivity_values.append(sensitivity)

    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    return {
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(sensitivity_values)),
        "accuracy": float(correct / total),
        "support": total,
        "num_classes": num_classes,
        "macro_f1_convention": "fixed four classes; absent class F1 = 0",
        "balanced_accuracy_convention": "fixed four classes; absent class recall = 0",
        "absent_classes": [row["class"] for row in per_class if row["absent"]],
    }


def print_metrics(metrics: dict, split_name: str) -> None:
    print(f"\n{split_name} Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"{split_name} Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"{split_name} Accuracy: {100.0 * metrics['accuracy']:.2f}%")
    print(f"{split_name} Examples: {metrics['support']:,}")

    if metrics["absent_classes"]:
        print(f"WARNING absent classes (F1 treated as 0): {metrics['absent_classes']}")

    print(f"\n{'Class':6s}{'Sens':>8s}{'Prec':>8s}{'F1':>8s}{'Support':>10s}{'Pred':>9s}")
    for row in metrics["per_class"]:
        marker = "*" if row["absent"] else " "
        print(
            f"{row['class']:6s}{row['sensitivity']:8.3f}{row['precision']:8.3f}"
            f"{row['f1']:8.3f}{row['support']:10d}{row['predicted']:9d}{marker}"
        )

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(f"{'':6s}" + "".join(f"{name:>8s}" for name in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:6s}" + "".join(f"{value:8d}" for value in metrics["confusion_matrix"][i]))


def _run_epoch(model, dataloader, criterion, device, optimizer=None):
    """Run one epoch, weighting each batch loss by its number of examples."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in tqdm(
            dataloader, desc="Training" if training else "Evaluating", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int(outputs.argmax(dim=1).eq(labels).sum().item())

    if total == 0:
        raise ValueError("Cannot run an epoch over an empty dataloader.")

    return total_loss / total, 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Return example-weighted loss, metrics and predictions."""
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size

            y_true.append(labels.cpu().numpy())
            y_pred.append(probabilities.argmax(dim=1).cpu().numpy())
            y_prob.append(probabilities.cpu().numpy())

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total
    return metrics, y_true, y_pred, y_prob


def train_model(
    device,
    model,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    config: dict,
    num_epochs: int = 10,
    lr: float = 0.001,
    step_size: int = 5,
    gamma: float = 0.1,
    class_weights=None,
):
    """Train with a max-validation-macro-F1 checkpoint rule.

    ``class_weights`` (optional tensor) weights the *training* loss only.
    Validation loss always uses the unweighted criterion so it stays
    comparable across runs.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else criterion
    )
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_balanced_accuracy": [],
        "val_per_class": [],
        "epoch_time_s": [],
    }

    best_f1 = -float("inf")
    best_epoch = -1
    best_val_metrics = None
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        train_loss, train_acc = _run_epoch(
            model, train_loader, train_criterion, device, optimizer=optimizer
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        history["epoch_time_s"].append(time.perf_counter() - epoch_start)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(100.0 * val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["val_per_class"].append(val_metrics["per_class"])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {100.0 * val_metrics['accuracy']:.2f}%, "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )
        print_metrics(val_metrics, "VALIDATION")

        # Strict improvement keeps the earlier checkpoint on ties.
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_name": config["model"],
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "epoch_number": epoch + 1,
                    "val_macro_f1": best_f1,
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "config": config,
                },
                checkpoint_path,
            )
            print(
                f"Saved best checkpoint at epoch {epoch + 1} "
                f"(validation macro-F1 {best_f1:.4f})."
            )

        scheduler.step()

    if best_epoch < 0:
        raise RuntimeError("Training finished without saving a checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    efficiency = {
        "total_train_time_s": time.perf_counter() - train_start,
        "mean_epoch_time_s": float(np.mean(history["epoch_time_s"])),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "epochs": num_epochs,
        "note": "epoch time includes validation evaluation and checkpoint saving",
    }
    return history, best_epoch, best_f1, best_val_metrics, efficiency


def plot_training_history(history: dict, title: str, output_path: Path) -> None:
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Example-weighted loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["epoch"], history["train_acc"], label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history["epoch"], history["val_macro_f1"], label="Val Macro-F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro-F1")
    ax3.set_title(f"{title} - Validation Macro-F1")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_history(history: dict, run_dir: Path) -> None:
    write_json(run_dir / "history.json", history)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_balanced_accuracy",
        "epoch_time_s",
    ]
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(history["epoch"])):
            writer.writerow(
                {
                    field: history[field][row_index]
                    for field in fieldnames
                }
            )


def measure_latency(
    model,
    device,
    warmup: int = 10,
    repetitions: int = 100,
) -> dict:
    """Measure batch-one float32 latency with synchronization."""
    model.eval()
    example = torch.zeros(1, 1, WINDOW_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(example)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings_ms = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    timings = np.asarray(timings_ms, dtype=np.float64)
    return {
        "device": str(device),
        "precision": "float32",
        "batch_size": 1,
        "warmup": warmup,
        "repetitions": repetitions,
        "synchronization": "torch.cuda.synchronize()" if device.type == "cuda" else "not needed",
        "mean_ms": float(timings.mean()),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(timings.min()),
    }


def measure_complexity(model) -> dict:
    """Count forward-pass FLOPs for one beat on CPU.

    Runs with grad enabled on purpose: in eval mode with grad disabled,
    nn.TransformerEncoder takes a fused fast path that FlopCounterMode
    cannot see, which would undercount the hybrid model.
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    example = torch.zeros(1, 1, WINDOW_SIZE)
    with FlopCounterMode(display=False) as counter:
        cpu_model(example)
    flops = int(counter.get_total_flops())
    return {
        "flops_per_inference": flops,
        "macs_per_inference_approx": flops // 2,
        "input_shape": [1, 1, WINDOW_SIZE],
        "counted_ops": "matmul/convolution/attention only",
    }


# ---------------------------------------------------------------------------
# Data and artifact validation
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_artifacts(data_dir: Path) -> tuple[dict, dict]:
    """Verify normalized arrays against the frozen split and normalization metadata."""
    split_manifest_path = data_dir / "split_manifest.json"
    normalization_path = data_dir / "normalization_metadata.json"

    if not split_manifest_path.exists():
        raise FileNotFoundError(
            "split_manifest.json is missing. Run scripts/split_data.py first."
        )
    if not normalization_path.exists():
        raise FileNotFoundError(
            "normalization_metadata.json is missing. Run scripts/normalize_data.py first."
        )

    normalization = load_json(normalization_path)

    if normalization["split_manifest_sha256"] != sha256_file(split_manifest_path):
        raise ValueError(
            "Normalized arrays do not match the current split manifest. "
            "Rerun split_data.py and normalize_data.py."
        )

    for filename, record in normalization.get("arrays", {}).items():
        path = data_dir / filename
        if sha256_file(path) != record["sha256"]:
            raise ValueError(
                f"{filename} does not match normalization_metadata.json. "
                "Rerun normalize_data.py."
            )

    return normalization, load_json(split_manifest_path)


def evaluate_test_checkpoint(args, device, data_dir: Path) -> None:
    """Evaluate a saved checkpoint on test without retraining."""
    checkpoint_path = args.test_checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = checkpoint_path.parent
    if (run_dir / "test_metrics.json").exists():
        raise FileExistsError(
            f"Test metrics already exist in {run_dir}. "
            "Create a new run instead of overwriting reported test evidence."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_name = checkpoint.get("model_name") or config.get("model")

    if model_name not in {"cnn", "hybrid"}:
        raise ValueError(f"Checkpoint does not declare a supported model: {model_name}")

    split_manifest_path = data_dir / "split_manifest.json"
    normalization_path = data_dir / "normalization_metadata.json"

    if config.get("split_manifest_sha256") != sha256_file(split_manifest_path):
        raise ValueError(
            "Checkpoint was trained against a different split manifest. "
            "Use a checkpoint from the current data generation."
        )
    if config.get("normalization_metadata_sha256") != sha256_file(normalization_path):
        raise ValueError(
            "Checkpoint was trained against different normalization metadata. "
            "Use a checkpoint from the current data generation."
        )

    model = build_model(
        model_name,
        num_classes=NUM_CLASSES,
        dropout_prob=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    validate_dataset(X_test, y_test, "Test")

    test_loader = make_loader(
        X_test,
        y_test,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    test_metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "TEST")

    write_json(run_dir / "test_metrics.json", test_metrics)
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=np.asarray(CLASS_NAMES),
    )

    write_json(
        run_dir / "test_access.json",
        {
            "accessed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy": (
                "Test evaluation is an explicit final step using the restored "
                "best-validation-macro-F1 checkpoint."
            ),
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch_number": checkpoint.get("epoch_number"),
            "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
            "prior_access_note": (
                "Record any manual test access that occurred before this run; "
                "the code path cannot infer it."
            ),
        },
    )

    print(f"\nTest artifacts saved to: {run_dir}")


def validate_dataset(features, labels, name: str) -> None:
    validate_feature_matrix(features, name)
    labels = validate_labels(labels, name)
    counts = class_counts(labels)

    print(f"{name}: {len(labels):,} examples")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {counts[class_id]:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "hybrid", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help=(
            "Weight the training loss by inverse class frequency computed from "
            "the training labels only. Validation loss stays unweighted."
        ),
    )
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help=(
            "Evaluate the restored best-validation checkpoint on the test split. "
            "Use this as the final step after configurations are frozen."
        ),
    )
    parser.add_argument(
        "--test-checkpoint",
        type=Path,
        default=None,
        help=(
            "Evaluate an existing best_checkpoint.pt on the test split without "
            "retraining. Requires --run-test."
        ),
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_test:
        print(
            "WARNING: --run-test evaluates the held-out test split. "
            "Use it only after model selection is final."
        )

    set_seeds_to(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator {device} will be used.")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_dir = args.data_dir

    if args.test_checkpoint is not None:
        if not args.run_test:
            raise SystemExit("--test-checkpoint requires --run-test.")
        verify_artifacts(data_dir)
        evaluate_test_checkpoint(args, device, data_dir)
        return

    normalization_metadata, split_manifest = verify_artifacts(data_dir)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    print("\nLoaded ECG datasets:")
    validate_dataset(X_train, y_train, "Training")
    validate_dataset(X_val, y_val, "Validation")

    model_names = ["cnn", "hybrid"] if args.model == "both" else [args.model]
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_root = args.results_dir / "runs"
    figures_dir = args.results_dir / "figures"

    for model_name in model_names:
        print("\n" + "=" * 70)
        print(f"Model: {model_name}")
        print("=" * 70)

        set_seeds_to(args.seed)
        model = build_model(model_name, num_classes=NUM_CLASSES, dropout_prob=args.dropout)

        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Parameters: {total_parameters:,} (trainable: {trainable_parameters:,})")

        run_dir = runs_root / f"{run_tag}_{model_name}_seed{args.seed}"
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Choose a different --run-tag or remove the old run."
            )
        run_dir.mkdir(parents=True, exist_ok=False)

        class_weights = None
        if args.class_weights:
            train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
            class_weights = torch.tensor(
                train_counts.sum() / (NUM_CLASSES * train_counts), dtype=torch.float32
            )
            print(f"Class weights (N,S,V,F): {class_weights.tolist()}")

        config = {
            "model": model_name,
            "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES),
            "window_size": WINDOW_SIZE,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "scheduler": f"StepLR(step_size={args.step_size}, gamma={args.gamma})",
            "loss": (
                "CrossEntropyLoss (training: inverse-frequency class weights; "
                "validation: unweighted)"
                if args.class_weights
                else "CrossEntropyLoss"
            ),
            "class_weights": (
                class_weights.tolist() if class_weights is not None else None
            ),
            "loss_aggregation": "example-weighted epoch mean",
            "dropout": args.dropout,
            "device": str(device),
            "precision": "float32",
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "checkpoint_rule": "maximum validation macro-F1; earlier checkpoint wins ties",
            "macro_f1_convention": "fixed four classes; absent class F1 = 0",
            "split_manifest_sha256": sha256_file(data_dir / "split_manifest.json"),
            "normalization_metadata_sha256": sha256_file(
                data_dir / "normalization_metadata.json"
            ),
            "normalization_mean": normalization_metadata["mean"],
            "normalization_std": normalization_metadata["std"],
            "split_selection": split_manifest.get("selection"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        write_json(run_dir / "config.json", config)

        train_loader = make_loader(
            X_train,
            y_train,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = make_loader(
            X_val,
            y_val,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=False,
            num_workers=args.num_workers,
        )

        checkpoint_path = run_dir / "best_checkpoint.pt"
        history, best_epoch, best_f1, best_val_metrics, efficiency = train_model(
            device,
            model,
            train_loader,
            val_loader,
            checkpoint_path,
            config,
            num_epochs=args.epochs,
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
            class_weights=class_weights,
        )

        save_history(history, run_dir)
        write_json(run_dir / "val_metrics_best.json", best_val_metrics)
        plot_training_history(
            history,
            model_name,
            figures_dir / f"training_history_{run_tag}_{model_name}_seed{args.seed}.png",
        )

        latency = measure_latency(
            model,
            device,
            repetitions=args.latency_repetitions,
        )
        write_json(run_dir / "latency.json", latency)

        cpu_latency = measure_latency(
            copy.deepcopy(model).cpu(),
            torch.device("cpu"),
            repetitions=args.latency_repetitions,
        )
        cpu_latency["torch_threads"] = torch.get_num_threads()
        write_json(run_dir / "latency_cpu.json", cpu_latency)

        write_json(
            run_dir / "efficiency.json",
            {**efficiency, **measure_complexity(model), "parameters": total_parameters},
        )

        write_json(
            run_dir / "model_summary.json",
            {
                "model": model_name,
                "parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "best_epoch": best_epoch,
                "best_epoch_number": best_epoch + 1,
                "best_val_macro_f1": best_f1,
                "checkpoint": checkpoint_path.name,
            },
        )

        print(
            f"\nRestored best checkpoint from epoch {best_epoch + 1} "
            f"(validation macro-F1 {best_f1:.4f})."
        )

        test_access = {
            "accessed": bool(args.run_test),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy": (
                "Test evaluation is an explicit final step using the restored "
                "best-validation-macro-F1 checkpoint."
            ),
            "prior_access_note": (
                "Record any manual test access that occurred before this run; "
                "the code path cannot infer it."
            ),
        }

        if args.run_test:
            X_test = np.load(data_dir / "X_test.npy")
            y_test = np.load(data_dir / "y_test.npy")
            validate_dataset(X_test, y_test, "Test")

            test_loader = make_loader(
                X_test,
                y_test,
                batch_size=args.batch_size,
                seed=args.seed,
                shuffle=False,
                num_workers=args.num_workers,
            )
            criterion = nn.CrossEntropyLoss()
            test_metrics, y_true, y_pred, y_prob = evaluate(
                model, test_loader, criterion, device
            )
            print_metrics(test_metrics, "TEST")

            write_json(run_dir / "test_metrics.json", test_metrics)
            np.savez_compressed(
                run_dir / "predictions.npz",
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=np.asarray(CLASS_NAMES),
            )
        else:
            print("\nTest split was not loaded or evaluated in this run.")

        write_json(run_dir / "test_access.json", test_access)
        print(f"\nRun artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
