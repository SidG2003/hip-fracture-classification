import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import numpy as np

from hip_fracture.data import check_disjoint, load_manifest, split_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.classes = ["type_a", "type_b"]
        self.rows = []
        for i in range(20):
            image = self.root / f"{i}.png"
            Image.new("RGB", (32, 32), (i, 50, 100)).save(image)
            self.rows.append([image.name, self.classes[i % 2]])
        self.write(self.rows)

    def write(self, rows):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)

    def test_validates_two_column_manifest(self):
        samples = load_manifest(self.manifest, self.classes)
        self.assertEqual(len(samples), 20)
        self.assertTrue(samples[0].path.is_absolute())

    def test_unknown_label_is_rejected(self):
        self.rows[0][1] = "unknown"
        self.write(self.rows)
        with self.assertRaisesRegex(ValueError, "invalid class"):
            load_manifest(self.manifest, self.classes)

    def test_duplicate_content_and_missing_classes_are_rejected(self):
        self.write(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_manifest(self.manifest, self.classes)
        self.write(self.rows[::2])
        with self.assertRaisesRegex(ValueError, "both pilot classes"):
            load_manifest(self.manifest, self.classes)

    def test_corrupt_and_missing_files_raise(self):
        (self.root / "0.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            load_manifest(self.manifest, self.classes)
        (self.root / "0.png").unlink()
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.manifest, self.classes)

    def test_high_bit_depth_is_not_silently_clipped(self):
        Image.fromarray(np.full((32, 32), 4096, dtype=np.uint16)).save(self.root / "0.png")
        with self.assertRaisesRegex(ValueError, "8-bit"):
            load_manifest(self.manifest, self.classes)

    def test_reproducible_disjoint_splits(self):
        first = split_manifest(self.manifest, self.root / "first", self.classes)
        second = split_manifest(self.manifest, self.root / "second", self.classes)
        self.assertEqual(first, second)
        self.assertEqual({k: len(v) for k, v in first.items()}, {"train": 12, "val": 4, "test": 4})
        for name in first:
            self.assertEqual(set(s.label for s in first[name]), set(self.classes))
            load_manifest(self.root / "first" / f"{name}.csv", self.classes)
        check_disjoint(*first.values())
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            check_disjoint(first["train"], first["train"])
        with self.assertRaises(FileExistsError):
            split_manifest(self.manifest, self.root / "first", self.classes)

    def test_invalid_fractions_and_insufficient_data_raise(self):
        with self.assertRaisesRegex(ValueError, "fractions"):
            split_manifest(self.manifest, self.root / "bad", self.classes, val_fraction=0.8)
        self.write(self.rows[:2])
        with self.assertRaisesRegex(ValueError, "Cannot stratify"):
            split_manifest(self.manifest, self.root / "small", self.classes)
