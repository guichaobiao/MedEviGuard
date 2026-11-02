import json
from pathlib import Path

import torch
from torch import nn

from .layers import resize


class FrozenDINOv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        directory = Path(config["backbone_directory"]).expanduser()
        metadata = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        if metadata.model_type != "dinov3_vit":
            raise ValueError("A local DINOv3 ViT Transformers checkpoint is required")
        processor = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
        if not processor.get("do_normalize") or len(processor["image_mean"]) != 3:
            raise ValueError("The backbone must provide its pretrained RGB normalization metadata")
        self.register_buffer("mean", torch.tensor(processor["image_mean"])[None, :, None, None])
        self.register_buffer("std", torch.tensor(processor["image_std"])[None, :, None, None])
        if len(processor["image_std"]) != 3 or not bool((self.std > 0).all()):
            raise ValueError("Invalid backbone normalization metadata")
        self.network = AutoModel.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config["backbone_dtype"]),
        )
        self.network.requires_grad_(False).eval()
        self.channels = metadata.hidden_size
        self.layers = tuple(config["backbone_layers"])
        self.input_size = config["input_size"]
        self.patch_size = metadata.patch_size
        self.prefix_tokens = 1 + metadata.num_register_tokens
        if self.input_size % self.patch_size or max(self.layers) >= len(self.network.layer):
            raise ValueError("Input size or layer indices are incompatible with this backbone")

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()
        return self

    @torch.no_grad()
    def forward(self, image):
        captured = {}

        def capture(index):
            def hook(module, inputs, output):
                tokens = self.network.norm(output)[:, self.prefix_tokens:].float()
                side = self.input_size // self.patch_size
                captured[index] = tokens.transpose(1, 2).reshape(image.shape[0], self.channels, side, side)
            return hook

        handles = [self.network.layer[index].register_forward_hook(capture(index)) for index in self.layers]
        try:
            with torch.autocast(device_type=image.device.type, enabled=False):
                pixels = (resize(image.float(), (self.input_size, self.input_size)) - self.mean) / self.std
                self.network(pixel_values=pixels, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        return [captured[index] for index in self.layers]
