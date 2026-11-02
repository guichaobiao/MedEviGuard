import math

import torch
from torch import nn
import torch.nn.functional as F


def resize(value, size):
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def block(inputs, outputs, config, stride=1):
    kernel = config["convolution_kernel"]
    padding = kernel // 2
    groups = math.gcd(outputs, config["normalization_groups"])
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(groups, outputs, eps=config["normalization_epsilon"]),
        nn.SiLU(),
        nn.Conv2d(outputs, outputs, kernel, padding=padding, bias=False),
        nn.GroupNorm(groups, outputs, eps=config["normalization_epsilon"]),
        nn.SiLU(),
    )


def local_average(value, kernel):
    pad = kernel // 2
    return F.avg_pool2d(F.pad(value, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)


class SpectralEvidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config["kernel_size"]
        coordinates = torch.arange(width, dtype=torch.float32) + 1 / 2
        filters = []
        for u, v in config["frequencies"]:
            row = torch.cos(math.pi * u * coordinates / width) * math.sqrt((1 if u == 0 else 2) / width)
            col = torch.cos(math.pi * v * coordinates / width) * math.sqrt((1 if v == 0 else 2) / width)
            filters.append(row[:, None] * col[None, :])
        self.register_buffer("filters", torch.stack(filters)[:, None].repeat(3, 1, 1, 1))
        self.channels = 2 * 3 * len(filters)

    def forward(self, image):
        cfg = self.config
        with torch.autocast(device_type=image.device.type, enabled=False):
            inputs = (image.float(), F.avg_pool2d(image.float(), cfg["downsample_factor"]))
            outputs = []
            for value in inputs:
                value = F.pad(value, (cfg["padding"],) * 4, mode="reflect")
                response = F.conv2d(value, self.filters.float(), stride=cfg["stride"], groups=3)
                outputs.append(response.sign() * torch.log1p(cfg["log_scale"] * response.abs()))
            return torch.cat((outputs[0], resize(outputs[1], outputs[0].shape[-2:])), dim=1)


class ContentMatchedRelation(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.config = config["relation"]
        self.epsilon = config["normalization_epsilon"]
        self.projection = block(2 * channels, channels, config)

    def forward(self, forensic, content):
        grid = self.config["grid_size"]
        height, width = min(grid, forensic.shape[-2]), min(grid, forensic.shape[-1])
        trace = F.adaptive_avg_pool2d(forensic, (height, width))
        guide = F.adaptive_avg_pool2d(content.detach(), (height, width))
        with torch.autocast(device_type=forensic.device.type, enabled=False):
            guide = F.normalize(guide.float().flatten(2).transpose(1, 2), dim=-1, eps=self.epsilon)
            affinity = guide @ guide.transpose(1, 2)
            y, x = torch.meshgrid(torch.arange(height, device=trace.device),
                                  torch.arange(width, device=trace.device), indexing="ij")
            coordinates = torch.stack((y.flatten(), x.flatten()), dim=-1)
            allowed = (coordinates[:, None] - coordinates[None]).abs().amax(-1) > 1
            available = allowed.any(-1)
            count = min(self.config["neighbors"], affinity.shape[-1])
            values, indices = affinity.masked_fill(~allowed, -torch.inf).topk(count, dim=-1)
            finite = values.isfinite()
            values = torch.where(available[None, :, None], values, torch.zeros_like(values))
            weights = (values / self.config["temperature"]).softmax(-1) * finite
            attention = torch.zeros_like(affinity).scatter(-1, indices, weights)
            tokens = trace.float().flatten(2).transpose(1, 2)
            context = attention @ tokens
            discrepancy = (tokens - context) * available[None, :, None]
            discrepancy = discrepancy.transpose(1, 2).reshape_as(trace)
        enhanced = self.projection(torch.cat((trace, discrepancy.to(trace.dtype)), dim=1))
        return resize(enhanced, forensic.shape[-2:])


class MultiRatioPooling(nn.Module):
    def __init__(self, ratios):
        super().__init__()
        self.ratios = tuple(ratios)
        self.weights = nn.Parameter(torch.zeros(len(ratios)))

    def forward(self, logits):
        flattened = logits.float().flatten(1)
        scores = [flattened.topk(max(1, int(flattened.shape[1] * ratio)), dim=1).values.mean(1)
                  for ratio in self.ratios]
        return (torch.stack(scores, dim=1) * self.weights.softmax(0)).sum(1)
