from pathlib import Path
import tempfile
import unittest

from PIL import Image

from hip_fracture.data import load_manifest
from scripts.prepare_anatomy import prepare


class AnatomyTests(unittest.TestCase):
    def test_duplicates_are_reported_without_modifying_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "images"
            for label, color in (("Chest", 20), ("Pelvis", 100)):
                folder = dataset / label
                folder.mkdir(parents=True)
                Image.new("L", (32, 32), color).save(folder / "first.jpg")
                (folder / "second.jpg").write_bytes((folder / "first.jpg").read_bytes())
            before = {p: p.read_bytes() for p in dataset.rglob("*.jpg")}
            output = root / "prepared"
            result = prepare(dataset, output)
            self.assertEqual(result["retained_counts"], {"Chest": 1, "Pelvis": 1})
            self.assertEqual(result["excluded_copies"], 2)
            self.assertEqual(len(load_manifest(output / "manifest.csv", ["Chest", "Pelvis"])), 2)
            self.assertEqual(before, {p: p.read_bytes() for p in dataset.rglob("*.jpg")})
            self.assertEqual(len((output / "excluded_duplicates.csv").read_text().splitlines()), 3)
            with self.assertRaises(FileExistsError):
                prepare(dataset, output)
            (dataset / "Pelvis/first.jpg").write_bytes((dataset / "Chest/first.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                prepare(dataset, root / "conflict")
