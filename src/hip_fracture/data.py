"""Two-column manifests and reproducible image-level splits."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    digest: str


def validate_classes(classes: list[str]) -> None:
    if len(classes) != 2 or len(set(classes)) != 2 or any(not c.strip() for c in classes):
        raise ValueError("Supply exactly two distinct, non-empty class names.")


def load_manifest(path: Path, classes: list[str]) -> list[Sample]:
    validate_classes(classes)
    samples = []
    seen: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_path", "label"]:
            raise ValueError(f"{path}: expected exactly the columns image_path,label.")
        for line, row in enumerate(reader, start=2):
            if None in row or not row["image_path"] or row["label"] not in classes:
                raise ValueError(f"{path}:{line}: missing image path or invalid class/column count.")
            image_path = (path.parent / row["image_path"]).resolve(strict=True)
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                raise ValueError(f"{image_path}: unsupported format; export de-identified raster images first.")
            with Image.open(image_path) as image:
                check_image_format(image, image_path)
                image.verify()
            with image_path.open("rb") as image_file:
                digest = hashlib.file_digest(image_file, "sha256").hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image content: {seen[digest]} and {image_path}.")
            seen[digest] = image_path
            samples.append(Sample(image_path, row["label"], digest))
    if set(s.label for s in samples) != set(classes):
        raise ValueError(f"{path}: both pilot classes must be represented.")
    return samples


def check_disjoint(*splits: list[Sample]) -> None:
    seen: set[str] = set()
    for split in splits:
        for sample in split:
            if sample.digest in seen:
                raise ValueError(f"Image content occurs in multiple splits: {sample.path}")
            seen.add(sample.digest)


def class_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(Counter(s.label for s in samples))


def write_manifest(path: Path, samples: list[Sample]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "label"])
        writer.writerows((str(s.path), s.label) for s in samples)


def split_manifest(
    manifest: Path,
    output: Path,
    classes: list[str],
    seed: int = 2025,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[Sample]]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite split directory: {output}")
    samples = load_manifest(manifest, classes)
    try:
        train_val, test = train_test_split(
            samples, test_size=test_fraction, random_state=seed,
            stratify=[s.label for s in samples],
        )
        train, val = train_test_split(
            train_val, test_size=val_fraction / (1 - test_fraction), random_state=seed,
            stratify=[s.label for s in train_val],
        )
    except ValueError as error:
        raise ValueError(f"Cannot stratify these class counts {class_counts(samples)}: {error}") from error
    splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        if set(s.label for s in split) != set(classes):
            raise ValueError(f"{name} would lack a class; collect more cases or adjust fractions.")
    check_disjoint(*splits.values())
    output.mkdir(parents=True)
    for name, split in splits.items():
        write_manifest(output / f"{name}.csv", split)
    return splits


def check_image_format(image: Image.Image, path: Path) -> None:
    if image.mode not in {"1", "L", "P", "RGB", "RGBA"} or getattr(image, "n_frames", 1) != 1:
        raise ValueError(f"{path}: export a single-frame 8-bit grayscale/RGB image with expert-approved windowing.")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        check_image_format(image, path)
        return ImageOps.exif_transpose(image).convert("RGB")
