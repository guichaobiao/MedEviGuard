import math
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def uniform(interval):
    return float(torch.empty(()).uniform_(*interval))


def jpeg(image, quality):
    array = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="JPEG", quality=quality, subsampling=0)
    stream.seek(0)
    with Image.open(stream) as decoded:
        array = np.array(decoded.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def resample(image, factor):
    size = image.shape[-2:]
    smaller = tuple(max(1, round(side * factor)) for side in size)
    image = F.interpolate(image, size=smaller, mode="bilinear", align_corners=False)
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def processing(sample, config):
    if float(torch.rand(())) >= config["processing_probability"]:
        return sample
    result = dict(sample)
    intensity = uniform(config["intensity_range"])
    factor = uniform(config["resampling_range"])
    low, high = config["jpeg_quality_range"]
    quality = int(torch.randint(low, high + 1, ()))
    compress = float(torch.rand(())) < config["jpeg_probability"]
    keys = ("image", "original") if sample["pair_valid"] else ("image",)
    for key in keys:
        value = (resample(sample[key][None], factor)[0] * intensity).clamp(0, 1)
        result[key] = jpeg(value, quality) if compress else value
    return result


def shrink_edit(sample, config):
    if not sample["pair_valid"] or float(torch.rand(())) >= config["small_edit_probability"]:
        return sample
    mask = sample["mask"]
    coordinates = torch.nonzero(mask[0] > 0)
    if not len(coordinates):
        return sample
    y0, x0 = coordinates.min(0).values.tolist()
    y1, x1 = (coordinates.max(0).values + 1).tolist()
    height, width = mask.shape[-2:]
    desired = uniform(config["small_edit_area_range"])
    scale = min(config["maximum_shrink_scale"], math.sqrt(desired * height * width / float(mask.sum())))
    shape = (max(1, int((y1 - y0) * scale)), max(1, int((x1 - x0) * scale)))
    residual = (sample["image"] - sample["original"]) * mask
    delta = F.interpolate(residual[None, :, y0:y1, x0:x1], size=shape, mode="bilinear", align_corners=False)[0]
    support = F.interpolate(mask[None, :, y0:y1, x0:x1], size=shape, mode="nearest")[0]
    y, x = (y0 + y1 - shape[0]) // 2, (x0 + x1 - shape[1]) // 2
    support = support * mask[:, y:y + shape[0], x:x + shape[1]]
    delta = delta * support
    if not bool(support.any()) or float(delta.abs().max()) <= config["minimum_residual"]:
        return sample
    new_mask = torch.zeros_like(mask)
    new_mask[:, y:y + shape[0], x:x + shape[1]] = support
    edited = sample["original"].clone()
    edited[:, y:y + shape[0], x:x + shape[1]] = (edited[:, y:y + shape[0], x:x + shape[1]] + delta).clamp(0, 1)
    result = dict(sample)
    result.update(image=edited, mask=new_mask)
    return result


def augment(sample, config):
    sample = shrink_edit(sample, config)
    if float(torch.rand(())) < config["horizontal_flip_probability"]:
        sample = dict(sample)
        for key in ("image", "original", "mask", "lesion_mask"):
            sample[key] = sample[key].flip(-1)
    return processing(sample, config)


def second_view(image, config):
    return (resample(image, uniform(config["resampling_range"])) * uniform(config["intensity_range"])).clamp(0, 1)
