from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_kauh import download_file, target_path


class DownloadTests(unittest.TestCase):
    def test_rejects_unsafe_paths(self):
        root = Path("/tmp/dataset")
        for name in ("../secret", "/absolute", "Dataset/../../secret", "Dataset\\file", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                target_path(root, name)
        self.assertEqual(target_path(root, "Dataset/Normal/a.jpg"), root / "Dataset/Normal/a.jpg")

    def test_download_records_hash_and_reuses_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {"name": "Dataset/Normal/a.jpg", "totalBytes": 3}
            with patch("scripts.download_kauh.open_url", return_value=BytesIO(b"abc")) as request:
                first = download_file(root, entry)
                second = download_file(root, entry)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual((root / entry["name"]).read_bytes(), b"abc")

    def test_incomplete_file_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {"name": "Dataset/a.jpg", "totalBytes": 4}
            with patch("scripts.download_kauh.open_url", return_value=BytesIO(b"abc")):
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    download_file(root, entry)
            self.assertFalse((root / entry["name"]).exists())
