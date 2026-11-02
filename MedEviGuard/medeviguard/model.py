import torch
from torch import nn
import torch.nn.functional as F

from .backbone import FrozenDINOv3
from .layers import ContentMatchedRelation, MultiRatioPooling, SpectralEvidence, block, local_average, resize


MODALITIES = ("CT", "CXR", "Bone_Xray", "Dermoscopy", "Trichoscopy", "Fundus", "Colon_histopathology")


class PairEvidenceTeacher(nn.Module):
    def __init__(self, spectral_channels, config):
        super().__init__()
        channels = [3 * config["evidence_channels"] + spectral_channels, *config["teacher_channels"]]
        modules = [block(a, b, config) for a, b in zip(channels, channels[1:])]
        self.decoder = nn.Sequential(*modules, nn.Conv2d(channels[-1], 1, 1))

    def forward(self, edited, original):
        a = edited["evidence"].detach()
        b = original["evidence"].detach()
        delta = (edited["spectral"].detach() - original["spectral"].detach()).abs()
        inputs = torch.cat((a, b, (a - b).abs(), resize(delta, a.shape[-2:])), dim=1)
        return self.decoder(inputs).float()


class MedEviGuard(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.foundation = FrozenDINOv3(config) if backbone is None else backbone
        self.foundation.requires_grad_(False).eval()
        channels = config["channels"]
        self.spectral = SpectralEvidence(config["spectral"])
        self.rgb_stem = block(2 * 3, channels[0], config, config["stem_stride"])
        self.gray_stem = block(2, channels[0], config, config["stem_stride"])
        self.ct_stem = block(2, channels[0], config, config["stem_stride"])
        self.stages = nn.ModuleList([block(a, b, config, config["stage_stride"])
                                    for a, b in zip(channels, channels[1:])])
        self.content_projections = nn.ModuleList([nn.Conv2d(self.foundation.channels, c, 1) for c in channels])
        self.frequency_projections = nn.ModuleList([block(self.spectral.channels, c, config) for c in channels])
        self.fusions = nn.ModuleList([block(3 * c, c, config) for c in channels])
        self.modulation = nn.ModuleList([nn.Embedding(len(MODALITIES), 2 * c) for c in channels])
        for embedding in self.modulation:
            nn.init.zeros_(embedding.weight)
        self.relation = ContentMatchedRelation(channels[-1], config)
        self.decoders = nn.ModuleList([block(channels[i + 1] + channels[i], channels[i], config)
                                      for i in reversed(range(len(channels) - 1))])
        self.auxiliary = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in channels])
        self.local_head = nn.Sequential(block(channels[0] + 3, channels[0], config), nn.Conv2d(channels[0], 1, 1))
        self.evidence_projection = nn.Conv2d(channels[1], config["evidence_channels"], 1)
        self.global_head = nn.Sequential(nn.Linear(channels[-1], config["global_hidden_channels"]),
                                         nn.GELU(), nn.Linear(config["global_hidden_channels"], 1))
        refinement = config["refinement"]
        width = refinement["channels"]
        self.detail_stem = block(3 * 3, width, config, refinement["stride"])
        context_channels = width + self.spectral.channels + config["evidence_channels"] + 1
        self.detail_context = block(context_channels, width, config)
        self.detail_embedding = nn.Conv2d(width, refinement["embedding_channels"], 1)
        self.correction = nn.Sequential(block(width + 3 * 3, width, config), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.pooling = MultiRatioPooling(config["pool_ratios"])
        self.evidence_fusion = nn.Linear(2, 1)
        with torch.no_grad():
            self.evidence_fusion.weight.copy_(torch.tensor([[1., 0.]]))
            self.evidence_fusion.bias.zero_()

    def train(self, mode=True):
        super().train(mode)
        self.foundation.eval()
        return self

    def forward(self, image, modality_id):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an RGB BCHW tensor")
        if image.shape[-2:] != (self.config["input_size"],) * 2:
            raise ValueError("Image resolution must match model.input_size")
        if modality_id.shape != (image.shape[0],) or modality_id.dtype != torch.long:
            raise ValueError("Expected one integer modality identifier per image")
        if bool(((modality_id < 0) | (modality_id >= len(MODALITIES))).any()):
            raise ValueError("Unknown modality identifier")
        content = self.foundation(image)
        spectral = self.spectral(image)
        with torch.autocast(device_type=image.device.type, enabled=False):
            pixels = image.float()
            highpass = pixels - local_average(pixels, 5)
            energy = local_average(highpass.square(), 17)
            settings = self.config["refinement"]
            normalized = (highpass / (energy + settings["residual_epsilon"]).sqrt()).clamp(
                -settings["clip_value"], settings["clip_value"])
            detail = torch.cat((pixels, highpass, normalized), dim=1)
        gray = torch.cat((pixels.mean(1, keepdim=True), highpass.mean(1, keepdim=True)), dim=1)
        ct = (modality_id == MODALITIES.index("CT"))[:, None, None, None]
        color = (modality_id >= MODALITIES.index("Dermoscopy"))[:, None, None, None]
        value = (self.ct_stem(gray) * ct + self.gray_stem(gray) * (~ct & ~color)
                 + self.rgb_stem(torch.cat((pixels, highpass), dim=1)) * color)
        features = []
        for index in range(len(self.config["channels"])):
            if index:
                value = self.stages[index - 1](value)
            semantic = resize(self.content_projections[index](content[index]), value.shape[-2:])
            frequency = self.frequency_projections[index](resize(spectral, value.shape[-2:]))
            value = self.fusions[index](torch.cat((value, semantic, frequency), dim=1))
            scale, shift = self.modulation[index](modality_id).chunk(2, dim=1)
            value = value * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            features.append(value)
        value = self.relation(features[-1], content[-1])
        evidence = None
        for index, (skip, decoder) in enumerate(zip(reversed(features[:-1]), self.decoders)):
            value = decoder(torch.cat((resize(value, skip.shape[-2:]), skip), dim=1))
            if index == len(self.decoders) - 2:
                evidence = self.evidence_projection(value)
        coarse = self.local_head(torch.cat((resize(value, image.shape[-2:]), pixels), dim=1)).float()
        trace = self.detail_stem(detail)
        size = trace.shape[-2:]
        trace = self.detail_context(torch.cat((trace, resize(spectral, size), resize(evidence, size),
                                               resize(coarse.detach(), size)), dim=1))
        embedding = F.normalize(self.detail_embedding(trace).float(), dim=1,
                                eps=self.config["normalization_epsilon"])
        correction = self.correction(torch.cat((resize(trace, image.shape[-2:]), detail), dim=1)).float()
        logits = coarse + correction
        local_logit = self.pooling(logits)
        global_logit = self.global_head(features[-1].mean((-2, -1))).flatten().float()
        image_logit = self.evidence_fusion(torch.stack((global_logit, local_logit), dim=1)).flatten().float()
        return {"mask_logits": logits, "mask_probability": logits.sigmoid(), "coarse_logits": coarse,
                "image_logit": image_logit, "image_score": image_logit.sigmoid(),
                "local_logit": local_logit, "local_score": local_logit.sigmoid(), "global_logit": global_logit,
                "auxiliary_logits": [head(feature).float() for head, feature in zip(self.auxiliary, features)],
                "embedding": embedding, "evidence": evidence, "spectral": spectral}
