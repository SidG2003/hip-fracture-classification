"""Frozen-head training, checkpoint loading, and held-out evaluation."""

import csv
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.image_processing_utils import BaseImageProcessor

from hip_fracture.data import Sample, check_disjoint, class_counts, load_manifest, load_rgb, validate_classes
from hip_fracture.model import FrozenClassifier, load_pretrained, weight_digest


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    model: Path
    output: Path
    classes: list[str]
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 0.001
    seed: int = 2025
    device: str = "auto"
    patience: int | None = None
    min_delta: float = 0.0


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(name)


def save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def encoder_settings(model: FrozenClassifier) -> dict:
    config = model.encoder.config.to_dict()
    config.pop("_name_or_path", None)
    return config


class ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], classes: list[str], processor: BaseImageProcessor):
        self.samples = samples
        self.labels = {name: i for i, name in enumerate(classes)}
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        pixels = self.processor(images=load_rgb(sample.path), return_tensors="pt")["pixel_values"][0]
        return pixels, self.labels[sample.label]


def metrics_for(labels: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> dict:
    validate_classes(classes)
    indices = list(range(len(classes)))
    if probabilities.shape != (len(labels), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite probabilities for every image and configured class.")
    if set(labels.tolist()) != set(indices):
        raise ValueError("Metrics require examples of every configured class.")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Class probabilities must be between zero and one and sum to one.")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        **(
            {"auroc_class_1": float(roc_auc_score(labels, probabilities[:, 1]))}
            if len(classes) == 2 else
            {"auroc_macro_ovr": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", labels=indices))}
        ),
        "class_order": classes,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(labels, predictions, labels=indices).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(classes)
        },
    }


def evaluate_loader(
    model: FrozenClassifier, loader: DataLoader, device: torch.device, classes: list[str],
) -> tuple[dict, np.ndarray]:
    model.eval()
    labels = []
    probabilities = []
    loss_sum = 0.0
    with torch.no_grad():
        for pixels, target in loader:
            logits = model(pixels.to(device))
            loss = torch.nn.functional.cross_entropy(logits, target.to(device), reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation/evaluation loss.")
            loss_sum += loss.item()
            labels.extend(target.tolist())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    probs = np.concatenate(probabilities)
    metrics = metrics_for(np.asarray(labels), probs, classes)
    metrics["loss"] = loss_sum / len(labels)
    return metrics, probs


def train(config: TrainConfig) -> Path:
    validate_classes(config.classes)
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if config.patience is not None and config.patience < 1:
        raise ValueError("Early-stopping patience must be positive or omitted.")
    if not math.isfinite(config.min_delta) or config.min_delta < 0:
        raise ValueError("Early-stopping min_delta must be finite and non-negative.")
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {config.output}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    train_samples = load_manifest(config.train_manifest, config.classes)
    val_samples = load_manifest(config.val_manifest, config.classes)
    check_disjoint(train_samples, val_samples)
    digest = weight_digest(config.model)
    model, processor = load_pretrained(config.model, len(config.classes))
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        ImageDataset(train_samples, config.classes, processor),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    val_loader = DataLoader(ImageDataset(val_samples, config.classes, processor), batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.learning_rate)
    config.output.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    settings.update({
        "device_used": str(device), "model_sha256": digest,
        "train_counts": class_counts(train_samples), "val_counts": class_counts(val_samples),
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "Pillow", "scikit-learn")},
        "encoder_frozen": True, "mask_ratio": 0.0,
        "selection_metric": "minimum validation cross-entropy",
        "patient_independence": "Assumed: one image per unique patient; not verifiable from this manifest.",
    })
    save_json(config.output / "config.json", settings)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    stopping_reference = float("inf")
    stale_epochs = 0
    stopped_early = False
    checkpoint_path = config.output / "best.pt"
    print(f"Device: {device}; training only {sum(p.numel() for p in model.head.parameters())} head parameters.", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        for pixels, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels.to(device))
            loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
        metrics, _ = evaluate_loader(model, val_loader, device, config.classes)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_samples), "val": metrics})
        save_json(config.output / "history.json", history)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            torch.save({
                "format_version": 1,
                "classes": config.classes,
                "head_state": {key: value.detach().cpu() for key, value in model.head.state_dict().items()},
                "model_path": str(config.model.resolve()),
                "model_sha256": digest,
                "encoder_config": encoder_settings(model),
                "processor": processor.to_dict(),
                "epoch": epoch,
                "val_metrics": metrics,
                "used_image_hashes": [s.digest for s in train_samples + val_samples],
            }, checkpoint_path)
        print(f"Epoch {epoch}/{config.epochs}: train_loss={history[-1]['train_loss']:.4f} val_loss={metrics['loss']:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["loss"] < stopping_reference - config.min_delta:
            stopping_reference = metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.patience is not None and stale_epochs >= config.patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch}: no improvement exceeding {config.min_delta} for {config.patience} epochs.", flush=True)
            break
    save_json(config.output / "summary.json", {
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "checkpoint": str(checkpoint_path.resolve()),
    })
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path, device: torch.device, model_override: Path | None = None,
) -> tuple[FrozenClassifier, BaseImageProcessor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format.")
    validate_classes(checkpoint["classes"])
    directory = model_override if model_override is not None else Path(checkpoint["model_path"])
    if weight_digest(directory) != checkpoint["model_sha256"]:
        raise ValueError("Backbone weights do not match the training checkpoint SHA-256.")
    model, processor = load_pretrained(directory, len(checkpoint["classes"]))
    if encoder_settings(model) != checkpoint["encoder_config"]:
        raise ValueError("Encoder configuration differs from the training checkpoint.")
    if processor.to_dict() != checkpoint["processor"]:
        raise ValueError("Preprocessing configuration differs from the training checkpoint.")
    model.head.load_state_dict(checkpoint["head_state"], strict=True)
    return model.to(device).eval(), processor, checkpoint


def evaluate(
    checkpoint_path: Path, manifest: Path, output: Path,
    batch_size: int = 4, device_name: str = "auto", model_override: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output}")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = select_device(device_name)
    model, processor, checkpoint = load_checkpoint(checkpoint_path, device, model_override)
    classes = checkpoint["classes"]
    samples = load_manifest(manifest, classes)
    used = set(checkpoint["used_image_hashes"])
    if any(s.digest in used for s in samples):
        raise ValueError("Evaluation images overlap training/validation images; use an untouched test set.")
    loader = DataLoader(ImageDataset(samples, classes, processor), batch_size=batch_size)
    metrics, probabilities = evaluate_loader(model, loader, device, classes)
    output.mkdir(parents=True)
    save_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "true_label", "predicted_label", *[f"prob_class_{i}" for i in range(len(classes))]])
        for sample, probs in zip(samples, probabilities, strict=True):
            writer.writerow([str(sample.path), sample.label, classes[int(probs.argmax())], *probs.tolist()])
    return metrics


def predict(
    checkpoint_path: Path, image: Path, device_name: str = "auto", model_override: Path | None = None,
) -> dict:
    device = select_device(device_name)
    model, processor, checkpoint = load_checkpoint(checkpoint_path, device, model_override)
    pixels = processor(images=load_rgb(image), return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        probs = model(pixels).softmax(dim=1)[0].cpu().tolist()
    if not all(math.isfinite(value) for value in probs):
        raise RuntimeError("Non-finite prediction probabilities.")
    classes = checkpoint["classes"]
    return {
        "predicted_label": classes[int(np.argmax(probs))],
        "probabilities": dict(zip(classes, probs, strict=True)),
        "scope": "Research only: assumes the image belongs to one of the checkpoint's configured classes; no out-of-distribution rejection.",
    }
