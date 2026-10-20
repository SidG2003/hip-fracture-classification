import unittest

import torch
from transformers import ViTMAEConfig, ViTMAEModel

from hip_fracture.model import FrozenClassifier


def tiny_model() -> FrozenClassifier:
    config = ViTMAEConfig(
        image_size=32, patch_size=16, hidden_size=32,
        num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    return FrozenClassifier(ViTMAEModel(config))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.model = tiny_model()
        self.images = torch.rand(2, 3, 32, 32)

    def test_only_head_updates_and_encoder_remains_in_eval(self):
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        self.model.train()
        self.assertTrue(self.model.head.training)
        self.assertFalse(self.model.encoder.training)
        optimizer = torch.optim.AdamW(self.model.head.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(self.model(self.images), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        for name, parameter in self.model.encoder.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(before[f"encoder.{name}"], parameter))
        self.assertFalse(torch.equal(before["head.weight"], self.model.head.weight))

    def test_all_patches_are_visible_and_predictions_are_deterministic(self):
        outputs = []
        hook = self.model.encoder.register_forward_hook(lambda module, args, output: outputs.append(output))
        self.addCleanup(hook.remove)
        self.model.eval()
        first = self.model(self.images)
        second = self.model(self.images)
        self.assertEqual(tuple(first.shape), (2, 2))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(outputs[0].last_hidden_state.shape[1], 5)
        self.assertEqual(outputs[0].mask.sum().item(), 0)
