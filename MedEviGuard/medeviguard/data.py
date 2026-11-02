from bisect import bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F

from .augment import augment
from .model import MODALITIES


def resolve(root, path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root).expanduser() / path).resolve()


def read_records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()] if path.suffix == ".jsonl" else json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty JSON array or JSONL manifest is required")
    return records


def load_image(path, modality):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float32)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] not in (1, 3):
            raise ValueError("NumPy images must be preprocessed HW or CHW arrays with one or three channels")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("NumPy images must contain finite intensities in [0, 1]")
        value = torch.from_numpy(array.copy())
        if modality in MODALITIES[:3]:
            value = value.mean(0, keepdim=True)
        return value.repeat(3, 1, 1) if value.shape[0] == 1 else value
    with Image.open(path) as image:
        array = np.array(image.convert("L" if modality in MODALITIES[:3] else "RGB"), dtype=np.float32) / 255
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_mask(path, shape):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float32).squeeze()
    else:
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32)
    if array.ndim != 2 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Masks must be finite and aligned with their input image")
    if array.max() > 1:
        array = array / 255
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Mask values must be in [0, 1] or [0, 255]")
    return torch.from_numpy((array > 0.5).astype(np.float32))[None]


def validate_records(records, split, config):
    identifiers = set()
    required = {"sample_id", "group_id", "image_path", "modality", "task", "label",
                "generator_id", "generator_family", "split"}
    for row in records:
        if required - row.keys():
            raise ValueError(f"Missing manifest fields: {sorted(required - row.keys())}")
        if row["sample_id"] in identifiers:
            raise ValueError("Duplicate sample identifiers in manifest")
        identifiers.add(row["sample_id"])
        if row["split"] != split or row["modality"] not in MODALITIES:
            raise ValueError("Manifest split or modality does not match the requested dataset")
        if row["task"] not in ("real", "local_edit", "full_image"):
            raise ValueError("Unsupported task")
        if row["label"] != int(row["task"] != "real"):
            raise ValueError("Image labels must agree with task metadata")
        if row["task"] == "full_image" and row.get("mask_valid", False):
            raise ValueError("Full-image synthesis must not receive localization targets")
        if row["task"] == "local_edit":
            area = row.get("mask_area")
            if not isinstance(area, (int, float)) or not math.isfinite(area) or not 0 < area <= 1:
                raise ValueError("Local edits require their native mask_area fraction")
            if row.get("mask_valid", bool(row.get("mask_path"))) and not row.get("mask_path"):
                raise ValueError("A valid local-edit target requires a mask file")
        if split in ("train", "val"):
            excluded = config["heldout_generators"].get(row["task"], [])
            if row["generator_id"] in excluded or row["generator_family"] in excluded:
                raise ValueError("A held-out manipulation method entered development data")
            if row.get("external_source", False):
                raise ValueError("External evaluation sources cannot enter development data")


class MedicalImageDataset(Dataset):
    def __init__(self, manifest, config, split):
        self.config = config
        self.root = config["data"]["root"]
        self.split = split
        self.records = read_records(resolve(self.root, manifest))
        validate_records(self.records, split, config["data"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_image(resolve(self.root, row["image_path"]), row["modality"])
        shape = image.shape[-2:]
        mask_valid = row["task"] == "real" or (row["task"] == "local_edit"
                     and row.get("mask_valid", bool(row.get("mask_path"))))
        mask = (load_mask(resolve(self.root, row["mask_path"]), shape)
                if mask_valid and row["task"] == "local_edit" else torch.zeros(1, *shape))
        pair_valid = bool(self.split == "train" and row["task"] == "local_edit" and mask_valid
                          and row.get("original_image_path") and row.get("pair_valid", True))
        original = (load_image(resolve(self.root, row["original_image_path"]), row["modality"])
                    if pair_valid else torch.zeros_like(image))
        if original.shape != image.shape:
            raise ValueError("Original and edited image geometry differs")
        lesion = (load_mask(resolve(self.root, row["lesion_mask_path"]), shape)
                  if row["task"] == "real" and row.get("lesion_mask_path") else torch.zeros_like(mask))
        size = (self.config["model"]["input_size"],) * 2
        image, original = [F.interpolate(value[None], size=size, mode="bilinear", align_corners=False)[0]
                           for value in (image, original)]
        mask, lesion = [F.interpolate(value[None], size=size, mode="nearest")[0] for value in (mask, lesion)]
        if mask_valid and row["task"] == "local_edit" and not bool(mask.any()):
            raise ValueError("The editing mask is empty at the configured input resolution")
        sample = {"image": image, "original": original, "mask": mask, "lesion_mask": lesion,
                  "pair_valid": pair_valid, "mask_valid": mask_valid, "label": float(row["label"]),
                  "local_label": float(row["task"] == "local_edit"), "local_label_valid": mask_valid,
                  "modality_id": MODALITIES.index(row["modality"])}
        sample.update({key: row[key] for key in ("sample_id", "group_id", "modality", "task", "generator_id")})
        return augment(sample, self.config["data"]["augmentation"]) if self.split == "train" else sample


def sampling_weights(records, config):
    weights = [0.] * len(records)
    keys = ("modality", "generator_family", "generator_id", "area", "group_id")

    def distribute(indices, mass, level):
        if level == len(keys):
            for index in indices:
                weights[index] = mass / len(indices)
            return
        groups = defaultdict(list)
        for index in indices:
            row = records[index]
            key = keys[level]
            value = (bisect_right(config["area_bins"], row["mask_area"]) if row["task"] == "local_edit" else None) if key == "area" else row[key]
            groups[value].append(index)
        for group in groups.values():
            distribute(group, mass / len(groups), level + 1)

    for task, probability in config["task_probabilities"].items():
        indices = [index for index, row in enumerate(records) if row["task"] == task]
        if not indices:
            raise ValueError(f"Task-balanced sampling requires training examples for {task}")
        distribute(indices, probability, 0)
    return weights


def check_development_boundary(train_records, validation_records, root):
    for key in ("sample_id", "group_id"):
        if {row[key] for row in train_records} & {row[key] for row in validation_records}:
            raise ValueError(f"Training and validation share {key}")
    sources = []
    for records in (train_records, validation_records):
        paths = {resolve(root, row[key]) for row in records
                 for key in ("image_path", "source_image_path", "original_image_path") if row.get(key)}
        sources.append(paths)
    if sources[0] & sources[1]:
        raise ValueError("Training and validation share source image paths")


def build_loader(config, split, manifest):
    dataset = MedicalImageDataset(manifest, config, split)
    sampler = (WeightedRandomSampler(sampling_weights(dataset.records, config["data"]), len(dataset), replacement=True)
               if split == "train" else None)
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], sampler=sampler,
                      shuffle=False, num_workers=config["training"]["num_workers"], drop_last=False)
