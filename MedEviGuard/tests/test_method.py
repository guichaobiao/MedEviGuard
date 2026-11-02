import ast
import copy
import inspect
import json
import os
from pathlib import Path
import tokenize
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from medeviguard.augment import shrink_edit
from medeviguard.config import ConfigurationError, load_config
from medeviguard.data import MedicalImageDataset, check_development_boundary, sampling_weights
from medeviguard.losses import MedEviGuardLoss
from medeviguard.model import MODALITIES, MedEviGuard, PairEvidenceTeacher


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    path = os.environ.get("MEDEVIGUARD_TEST_CONFIG")
    if path is None:
        pytest.skip("Set MEDEVIGUARD_TEST_CONFIG to a complete private test configuration")
    return copy.deepcopy(load_config(path))


class SyntheticFrozenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config["channels"][0]
        self.network = nn.Conv2d(3, self.channels, 1, stride=config["stage_stride"])

    def forward(self, image):
        with torch.no_grad():
            output = self.network(image)
        return [output] * 4


def model_and_batch(config):
    torch.manual_seed(config["training"]["seed"])
    model = MedEviGuard(config["model"], backbone=SyntheticFrozenEncoder(config["model"]))
    size = config["model"]["input_size"]
    count = len(MODALITIES)
    image = torch.rand(count, 3, size, size)
    mask = torch.zeros(count, 1, size, size)
    mask[1, :, size // 2, size // 2] = 1
    labels = torch.zeros(count)
    labels[1:3] = 1
    valid = torch.ones(count, dtype=torch.bool)
    valid[2] = False
    paired = torch.zeros_like(valid)
    paired[1] = True
    return model, {"image": image, "original": image.clone(), "mask": mask,
                   "lesion_mask": torch.zeros_like(mask), "label": labels,
                   "mask_valid": valid, "local_label_valid": valid,
                   "local_label": paired.float(), "pair_valid": paired,
                   "modality_id": torch.arange(count)}


def test_public_configuration_is_deliberately_incomplete():
    with pytest.raises(ConfigurationError, match="Required configuration values are unset"):
        load_config(ROOT / "configs" / "paper.yaml")
    config = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text())
    assert config["model"]["input_size"] == 512
    assert config["training"]["epochs"] == 60
    assert config["training"]["learning_rate"] == 0.0001
    assert all(value is None for value in config["loss"]["weights"].values())
    assert config["validation"]["mask_threshold"] is None


def test_python_sources_have_no_comments_or_docstrings():
    for path in [*ROOT.joinpath("medeviguard").glob("*.py"), *ROOT.joinpath("tests").glob("*.py")]:
        with tokenize.open(path) as stream:
            assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(stream.readline))
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node) is None


def test_single_image_interface():
    assert list(inspect.signature(MedEviGuard.forward).parameters) == ["self", "image", "modality_id"]


def test_refinement_initialization_and_backward(config):
    model, batch = model_and_batch(config)
    model.train()
    teacher = PairEvidenceTeacher(model.spectral.channels, config["model"])
    criterion = MedEviGuardLoss(config["loss"])
    output = model(batch["image"], batch["modality_id"])
    assert output["mask_logits"].shape == batch["mask"].shape
    torch.testing.assert_close(output["mask_logits"], output["coarse_logits"])
    assert not model.foundation.training
    original = model(batch["original"], batch["modality_id"])
    losses = criterion(output, batch, original, teacher(output, original), teacher(original, original),
                       output, config["loss"]["teacher_warmup_epochs"])
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in model.foundation.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.grad is not None)
    assert model.correction[-1].weight.grad is not None
    assert any(parameter.grad is not None for parameter in teacher.parameters())


def test_teacher_cannot_backpropagate_into_student_features(config):
    model, batch = model_and_batch(config)
    output = model(batch["image"], batch["modality_id"])
    output["evidence"].retain_grad()
    teacher = PairEvidenceTeacher(model.spectral.channels, config["model"])
    teacher(output, output).sum().backward()
    assert output["evidence"].grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_generated_images_have_no_localization_or_pair_targets(config):
    model, batch = model_and_batch(config)
    batch = {key: value[2:3] for key, value in batch.items()}
    output = model(batch["image"], batch["modality_id"])
    criterion = MedEviGuardLoss(config["loss"])
    losses = criterion(output, batch, view=output)
    for key in ("dense", "auxiliary", "local", "small_region", "agreement", "paired_features",
                "teacher", "distillation", "original_negative", "regional_margin", "real_pathology"):
        assert float(losses[key].detach()) == 0
    expected = config["loss"]["appearance_image_weight"] * F.binary_cross_entropy_with_logits(
        output["image_logit"], batch["label"])
    torch.testing.assert_close(losses["appearance"], expected)


def test_validation_never_opens_authentic_reference(config, tmp_path):
    size = config["model"]["input_size"]
    image = np.zeros((size, size), dtype=np.uint8)
    mask = image.copy()
    mask[size // 2, size // 2] = 255
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    row = {"sample_id": "edited", "group_id": "validation_source", "image_path": "image.png",
           "modality": "CXR", "task": "local_edit", "label": 1, "generator_id": "seen_test_fixture",
           "generator_family": "seen_test_fixture", "split": "val", "mask_path": "mask.png",
           "mask_area": float((mask > 0).mean()), "original_image_path": "does_not_exist.png"}
    (tmp_path / "manifest.json").write_text(json.dumps([row]))
    config["data"]["root"] = str(tmp_path)
    sample = MedicalImageDataset("manifest.json", config, "val")[0]
    assert sample["mask_valid"] and not sample["pair_valid"]
    assert not bool(sample["original"].any())


def test_sampling_preserves_task_probability(config):
    rows = [{"task": task, "modality": modality, "generator_family": task, "generator_id": task,
             "group_id": f"{task}:{modality}", "mask_area": config["loss"]["small_region_area"]}
            for task in config["data"]["task_probabilities"] for modality in MODALITIES]
    rows.append(dict(rows[0]))
    weights = sampling_weights(rows, config["data"])
    for task, mass in config["data"]["task_probabilities"].items():
        assert sum(weight for row, weight in zip(rows, weights) if row["task"] == task) == pytest.approx(mass)


def test_source_group_overlap_is_rejected(tmp_path):
    train = [{"sample_id": "train_image", "group_id": "shared_source", "image_path": "train.png"}]
    validation = [{"sample_id": "val_image", "group_id": "shared_source", "image_path": "val.png"}]
    with pytest.raises(ValueError, match="group_id"):
        check_development_boundary(train, validation, tmp_path)


def test_small_edit_preserves_reference_outside_mask(config):
    model, batch = model_and_batch(config)
    sample = {key: value[1] for key, value in batch.items()}
    sample["pair_valid"] = True
    sample["mask"] = torch.ones_like(sample["mask"])
    sample["original"] = torch.zeros_like(sample["image"])
    sample["image"] = torch.ones_like(sample["image"])
    with patch("medeviguard.augment.torch.rand", return_value=torch.zeros(())):
        result = shrink_edit(sample, config["data"]["augmentation"])
    outside = (result["mask"] == 0).expand_as(result["image"])
    assert bool(outside.any())
    torch.testing.assert_close(result["image"][outside], result["original"][outside])
