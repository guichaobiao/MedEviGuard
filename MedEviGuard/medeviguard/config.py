import math
from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    pass


def unresolved(value, prefix=""):
    if isinstance(value, dict):
        return [path for key, item in value.items()
                for path in unresolved(item, f"{prefix}.{key}" if prefix else key)]
    if isinstance(value, list):
        return [path for index, item in enumerate(value)
                for path in unresolved(item, f"{prefix}[{index}]")]
    return [prefix] if value is None else []


def check_number(value, path, minimum=None, maximum=None, integer=False, strict=False):
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid = valid and math.isfinite(value)
    valid = valid and (not integer or isinstance(value, int))
    if minimum is not None:
        valid = valid and (value > minimum if strict else value >= minimum)
    if maximum is not None:
        valid = valid and value <= maximum
    if not valid:
        raise ConfigurationError(f"Invalid value for {path}: {value!r}")


def validate_config(config):
    missing = unresolved(config)
    if missing:
        raise ConfigurationError("Required configuration values are unset:\n" + "\n".join(missing))
    m, loss, train, data, val = (config[key] for key in
                               ("model", "loss", "training", "data", "validation"))
    for key in ("input_size", "convolution_kernel", "normalization_groups", "stem_stride",
                "stage_stride", "evidence_channels", "global_hidden_channels"):
        check_number(m[key], f"model.{key}", minimum=0, integer=True, strict=True)
    if m["convolution_kernel"] % 2 != 1:
        raise ConfigurationError("model.convolution_kernel must be odd")
    for key in ("channels", "backbone_layers"):
        if len(m[key]) != 4:
            raise ConfigurationError(f"model.{key} must contain four entries")
        for value in m[key]:
            check_number(value, f"model.{key}", minimum=0, integer=True,
                         strict=key == "channels")
    if sorted(set(m["backbone_layers"])) != m["backbone_layers"]:
        raise ConfigurationError("Backbone layer indices must be unique and increasing")
    if not m["teacher_channels"] or not m["pool_ratios"]:
        raise ConfigurationError("Teacher channels and pooling proportions must be nonempty")
    for value in m["teacher_channels"]:
        check_number(value, "model.teacher_channels", minimum=0, integer=True, strict=True)
    for value in m["pool_ratios"]:
        check_number(value, "model.pool_ratios", minimum=0, maximum=1, strict=True)
    check_number(m["normalization_epsilon"], "model.normalization_epsilon", minimum=0, strict=True)
    spectral = m["spectral"]
    for key in ("kernel_size", "stride", "downsample_factor"):
        check_number(spectral[key], f"model.spectral.{key}", minimum=0, integer=True, strict=True)
    check_number(spectral["padding"], "model.spectral.padding", minimum=0, integer=True)
    check_number(spectral["log_scale"], "model.spectral.log_scale", minimum=0, strict=True)
    if spectral["downsample_factor"] <= 1 or not spectral["frequencies"]:
        raise ConfigurationError("The spectral branch requires two resolutions and non-DC filters")
    for pair in spectral["frequencies"]:
        if len(pair) != 2 or pair == [0, 0]:
            raise ConfigurationError("DCT frequencies must be non-DC coordinate pairs")
        for value in pair:
            check_number(value, "model.spectral.frequencies", minimum=0,
                         maximum=spectral["kernel_size"] - 1, integer=True)
    for key in ("grid_size", "neighbors"):
        check_number(m["relation"][key], f"model.relation.{key}", minimum=0, integer=True, strict=True)
    check_number(m["relation"]["temperature"], "model.relation.temperature", minimum=0, strict=True)
    for key in ("channels", "embedding_channels", "stride"):
        check_number(m["refinement"][key], f"model.refinement.{key}", minimum=0, integer=True, strict=True)
    for key in ("residual_epsilon", "clip_value"):
        check_number(m["refinement"][key], f"model.refinement.{key}", minimum=0, strict=True)
    for key, value in loss["weights"].items():
        check_number(value, f"loss.weights.{key}", minimum=0)
    if not any(loss["weights"].values()):
        raise ConfigurationError("At least one loss weight must be positive")
    for key in ("tversky_false_positive", "tversky_false_negative", "regional_margin",
                "feature_margin", "outside_weight", "appearance_image_weight"):
        check_number(loss[key], f"loss.{key}", minimum=0)
    check_number(loss["smoothing"], "loss.smoothing", minimum=0, strict=True)
    check_number(loss["small_region_area"], "loss.small_region_area", minimum=0, maximum=1, strict=True)
    check_number(loss["teacher_warmup_epochs"], "loss.teacher_warmup_epochs", minimum=0, integer=True)
    if loss["weights"]["distillation"] and not loss["weights"]["teacher"]:
        raise ConfigurationError("Distillation requires a supervised teacher")
    for key in ("epochs", "batch_size", "accumulation_steps", "appearance_interval", "log_interval"):
        check_number(train[key], f"training.{key}", minimum=0, integer=True, strict=True)
    for key in ("seed", "num_workers", "warmup_epochs"):
        check_number(train[key], f"training.{key}", minimum=0, integer=True)
    if train["warmup_epochs"] >= train["epochs"]:
        raise ConfigurationError("Warm-up must be shorter than total training")
    for key in ("learning_rate", "adam_epsilon", "gradient_clip"):
        check_number(train[key], f"training.{key}", minimum=0, strict=True)
    check_number(train["weight_decay"], "training.weight_decay", minimum=0)
    check_number(train["minimum_lr_ratio"], "training.minimum_lr_ratio", minimum=0, maximum=1)
    if len(train["adam_betas"]) != 2:
        raise ConfigurationError("AdamW requires two beta values")
    for value in train["adam_betas"]:
        check_number(value, "training.adam_betas", minimum=0, maximum=1)
        if value == 1:
            raise ConfigurationError("AdamW beta values must be less than one")
    if m["backbone_dtype"] not in ("float32", "bfloat16", "float16"):
        raise ConfigurationError("Unsupported backbone dtype")
    if train["amp_dtype"] not in ("bfloat16", "float16"):
        raise ConfigurationError("Unsupported mixed-precision dtype")
    if not isinstance(train["mixed_precision"], bool):
        raise ConfigurationError("training.mixed_precision must be boolean")
    if set(data["task_probabilities"]) != {"real", "local_edit", "full_image"}:
        raise ConfigurationError("Task probabilities must describe all three tasks")
    for value in data["task_probabilities"].values():
        check_number(value, "data.task_probabilities", minimum=0, maximum=1, strict=True)
    if not math.isclose(sum(data["task_probabilities"].values()), 1):
        raise ConfigurationError("Task probabilities must sum to one")
    if sorted(set(data["area_bins"])) != data["area_bins"]:
        raise ConfigurationError("Area boundaries must be unique and increasing")
    for value in data["area_bins"]:
        check_number(value, "data.area_bins", minimum=0, maximum=1, strict=True)
    aug = data["augmentation"]
    for key in ("horizontal_flip_probability", "processing_probability", "jpeg_probability",
                "small_edit_probability", "maximum_shrink_scale"):
        check_number(aug[key], f"data.augmentation.{key}", minimum=0, maximum=1)
    if aug["maximum_shrink_scale"] == 0:
        raise ConfigurationError("Maximum shrink scale must be positive")
    check_number(aug["minimum_residual"], "data.augmentation.minimum_residual", minimum=0)
    for key in ("intensity_range", "resampling_range", "jpeg_quality_range", "small_edit_area_range"):
        values = aug[key]
        if len(values) != 2 or values[0] > values[1]:
            raise ConfigurationError(f"Invalid interval: data.augmentation.{key}")
        maximum = 100 if key == "jpeg_quality_range" else 1 if key == "small_edit_area_range" else None
        for value in values:
            check_number(value, f"data.augmentation.{key}", minimum=0, maximum=maximum,
                         integer=key == "jpeg_quality_range", strict=key != "jpeg_quality_range")
    for key in ("image_threshold", "mask_threshold"):
        check_number(val[key], f"validation.{key}", minimum=0, maximum=1)
    required_metrics = {"full_image_auc", "local_edit_auc", "macro_dice", "worst_group_dice", "small_region_dice"}
    if set(val["selection_weights"]) != required_metrics:
        raise ConfigurationError("All five model-selection terms must be configured")
    for value in val["selection_weights"].values():
        check_number(value, "validation.selection_weights", minimum=0)
    if not math.isclose(sum(val["selection_weights"].values()), 1):
        raise ConfigurationError("Model-selection weights must sum to one")
    return config


def load_config(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration must be a mapping")
    return validate_config(config)
