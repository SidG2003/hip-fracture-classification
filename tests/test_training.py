import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTMAEConfig, ViTMAEForPreTraining

from hip_fracture.data import load_manifest, split_manifest
from hip_fracture.training import TrainConfig, evaluate, load_checkpoint, metrics_for, predict, train
from tests.test_model import tiny_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model"
        config = ViTMAEConfig(
            image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, decoder_hidden_size=32,
            decoder_num_hidden_layers=1, decoder_num_attention_heads=4, decoder_intermediate_size=64,
        )
        ViTMAEForPreTraining(config).save_pretrained(self.model_path)
        ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(self.model_path)
        self.classes = ["type_a", "type_b"]
        manifest = self.root / "all.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_path", "label"])
            for i in range(20):
                image = self.root / f"image{i}.png"
                Image.new("RGB", (32, 32), (20 + i, 40 if i % 2 else 220, 100)).save(image)
                writer.writerow([image.name, self.classes[i % 2]])
        self.splits = self.root / "splits"
        split_manifest(manifest, self.splits, self.classes)
        self.config = TrainConfig(
            train_manifest=self.splits / "train.csv", val_manifest=self.splits / "val.csv",
            model=self.model_path, output=self.root / "run", classes=self.classes,
            epochs=3, batch_size=4, learning_rate=0.01, device="cpu",
        )

    def test_training_checkpoint_reload_evaluation_and_prediction(self):
        checkpoint = train(self.config)
        stored = torch.load(checkpoint, weights_only=True)
        history = json.loads((self.config.output / "history.json").read_text())
        self.assertEqual(stored["val_metrics"]["loss"], min(row["val"]["loss"] for row in history))
        self.assertEqual(set(stored["head_state"]), {"weight", "bias"})
        model, _, _ = load_checkpoint(checkpoint, torch.device("cpu"))
        self.assertEqual(model.head.out_features, len(self.classes))
        metrics = evaluate(checkpoint, self.splits / "test.csv", self.root / "evaluation", device_name="cpu")
        self.assertEqual(metrics["class_order"], self.classes)
        self.assertEqual(
            sum(row["support"] for row in metrics["per_class"].values()),
            len(load_manifest(self.splits / "test.csv", self.classes)),
        )
        with (self.root / "evaluation" / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(rows[0]), {
            "image_path", "true_label", "predicted_label",
            *[f"prob_class_{i}" for i in range(len(self.classes))],
        })
        image = load_manifest(self.splits / "test.csv", self.classes)[0].path
        first = predict(checkpoint, image, device_name="cpu")
        second = predict(checkpoint, image, device_name="cpu")
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first["probabilities"].values()), 1, places=6)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(checkpoint, self.config.train_manifest, self.root / "leaked", device_name="cpu")
        with self.assertRaises(FileExistsError):
            train(self.config)
        settings_path = self.model_path / "config.json"
        settings = json.loads(settings_path.read_text())
        settings["layer_norm_eps"] = 0.001
        settings_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "Encoder configuration"):
            load_checkpoint(checkpoint, torch.device("cpu"))
        settings["layer_norm_eps"] = model.encoder.config.layer_norm_eps
        settings_path.write_text(json.dumps(settings))
        with (self.model_path / "model.safetensors").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_checkpoint(checkpoint, torch.device("cpu"))

    def test_metrics_have_known_values(self):
        metrics = metrics_for(np.array([0, 0, 1, 1]), np.array([[.9, .1], [.8, .2], [.3, .7], [.2, .8]]), self.classes)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["auroc_class_1"], 1)
        self.assertEqual(metrics["confusion_matrix_rows_true_columns_predicted"], [[2, 0], [0, 2]])
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics_for(np.array([0]), np.array([[float("nan"), 1]]), self.classes)

    def test_early_stopping_preserves_true_best_checkpoint(self):
        config = replace(self.config, epochs=10, patience=2, min_delta=0.05)
        losses = [1.0, 0.8, 0.79, 0.78]
        results = [({"loss": loss, "macro_f1": 0.5}, np.empty((0, len(self.classes)))) for loss in losses]
        with patch("hip_fracture.training.evaluate_loader", side_effect=results) as evaluator:
            checkpoint = train(config)
        self.assertEqual(evaluator.call_count, 4)
        summary = json.loads((config.output / "summary.json").read_text())
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["epochs_completed"], 4)
        self.assertEqual(summary["best_epoch"], 4)
        self.assertEqual(torch.load(checkpoint, weights_only=True)["val_metrics"]["loss"], 0.78)

    def test_early_stopping_invalid_settings_raise(self):
        for settings in ({"patience": 0}, {"min_delta": -1}, {"min_delta": float("nan")}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "Early-stopping"):
                train(replace(self.config, **settings))

    def test_linear_head_can_overfit_separable_features(self):
        model = tiny_model()
        features = torch.cat([torch.ones(8, 32), -torch.ones(8, 32)])
        targets = torch.tensor([0] * 8 + [1] * 8)
        optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model.head(features), targets)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.01)
        self.assertTrue(torch.equal(model.head(features).argmax(dim=1), targets))


class MulticlassTrainingTests(TrainingTests):
    def setUp(self):
        super().setUp()
        self.classes = ["intertrochanteric", "femoral_neck", "normal"]
        manifest = self.root / "three.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_path", "label"])
            for i in range(30):
                image = self.root / f"three{i}.png"
                Image.new("RGB", (32, 32), (20 + i, 30 + 80 * (i % 3), 100)).save(image)
                writer.writerow([image.name, self.classes[i % 3]])
        self.splits = self.root / "three_splits"
        split_manifest(manifest, self.splits, self.classes)
        self.config = TrainConfig(
            train_manifest=self.splits / "train.csv", val_manifest=self.splits / "val.csv",
            model=self.model_path, output=self.root / "three_run", classes=self.classes,
            epochs=3, batch_size=4, learning_rate=0.01, device="cpu",
        )

    def test_metrics_have_known_values(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.full((6, 3), 0.1)
        probabilities[np.arange(6), labels] = 0.8
        metrics = metrics_for(labels, probabilities, self.classes)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["auroc_macro_ovr"], 1)
        self.assertNotIn("auroc_class_1", metrics)
        self.assertEqual(metrics["confusion_matrix_rows_true_columns_predicted"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        with self.assertRaisesRegex(ValueError, "every configured class"):
            metrics_for(labels[:2], probabilities[:2], self.classes)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            metrics_for(labels, probabilities / 2, self.classes)
