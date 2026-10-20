"""Prepare an exact-deduplicated Chest/Pelvis manifest without changing images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from hip_fracture.data import Sample, check_image_format, class_counts, load_manifest, validate_classes, write_manifest


def prepare(dataset: Path, output: Path, classes: list[str] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preparation directory: {output}")
    classes = ["Chest", "Pelvis"] if classes is None else classes
    validate_classes(classes)
    for label in classes:
        if Path(label).name != label or label in {".", ".."}:
            raise ValueError(f"Class must be an immediate folder name: {label}")
    retained: dict[str, Sample] = {}
    excluded = []
    original_counts = {}
    for label in classes:
        folder = dataset / label
        files = sorted(path for path in folder.iterdir() if path.is_file() and not path.name.startswith("."))
        if not files:
            raise ValueError(f"No images found in {folder}.")
        original_counts[label] = len(files)
        for path in files:
            path = path.resolve(strict=True)
            with Image.open(path) as image:
                check_image_format(image, path)
                image.verify()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            previous = retained.get(digest)
            if previous is None:
                retained[digest] = Sample(path, label, digest)
            elif previous.label != label:
                raise ValueError(f"Conflicting labels for identical images: {previous.path} and {path}")
            else:
                excluded.append([str(path), str(previous.path), label, digest])
    output.mkdir(parents=True)
    manifest = output / "manifest.csv"
    write_manifest(manifest, list(retained.values()))
    validated = load_manifest(manifest, classes)
    with (output / "excluded_duplicates.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["excluded_path", "retained_path", "label", "sha256"])
        writer.writerows(excluded)
    summary = {
        "dataset": str(dataset.resolve()),
        "original_counts": original_counts,
        "retained_counts": class_counts(validated),
        "excluded_copies": len(excluded),
        "original_images_modified": False,
        "method": "SHA-256 of file bytes; keep the lexicographically first path within each class.",
        "limitations": "Does not detect re-encoded/near-duplicate images or establish patient independence.",
    }
    with (output / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Anatomy Dataset"))
    parser.add_argument("--classes", nargs="+", default=["Chest", "Pelvis"])
    parser.add_argument("--output", type=Path, default=Path("data/anatomy"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output, args.classes), indent=2))
