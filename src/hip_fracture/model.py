"""Frozen MAE encoder with an independently initialized classification head."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from transformers import AutoImageProcessor, ViTMAEModel
from transformers.image_processing_utils import BaseImageProcessor


class FrozenClassifier(nn.Module):
    def __init__(self, encoder: ViTMAEModel):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.mask_ratio = 0.0
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Linear(encoder.config.hidden_size, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch = self.encoder.config.patch_size
        count = (pixel_values.shape[-2] // patch) * (pixel_values.shape[-1] // patch)
        # Ordered noise disables MAE's random patch shuffle as well as masking.
        noise = torch.arange(count, device=pixel_values.device, dtype=torch.float32)
        noise = noise.unsqueeze(0).expand(pixel_values.shape[0], -1)
        with torch.no_grad():
            features = self.encoder(pixel_values=pixel_values, noise=noise).last_hidden_state[:, 0]
        return self.head(features)


def weight_digest(directory: Path) -> str:
    with (directory / "model.safetensors").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_pretrained(directory: Path) -> tuple[FrozenClassifier, BaseImageProcessor]:
    directory = directory.resolve(strict=True)
    encoder, info = ViTMAEModel.from_pretrained(
        directory, local_files_only=True, output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Pretrained encoder did not load completely: {info}")
    unexpected = [key for key in info["unexpected_keys"] if not key.startswith("decoder.")]
    if unexpected:
        raise RuntimeError(f"Unexpected non-decoder weights in checkpoint: {unexpected}")
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
    return FrozenClassifier(encoder), processor
