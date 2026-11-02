from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def classification_metrics(rows, threshold):
    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["image_score"] for row in rows])
    positive, negative = labels == 1, labels == 0
    both = bool(positive.any() and negative.any())
    predictions = scores >= threshold
    return {"count": len(rows), "auroc": float(roc_auc_score(labels, scores)) if both else None,
            "auprc": float(average_precision_score(labels, scores)) if both else None,
            "sensitivity": float(predictions[positive].mean()) if positive.any() else None,
            "specificity": float((~predictions[negative]).mean()) if negative.any() else None,
            "accuracy": float((predictions == labels).mean()) if len(rows) else None}


def source_macro(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            groups[row["group_id"]].append(row[key])
    return float(np.mean([np.mean(values) for values in groups.values()])) if groups else None


def summarize(rows, config):
    threshold = config["validation"]["image_threshold"]
    result = {"detection": classification_metrics(rows, threshold), "task_detection": {}}
    for task in ("full_image", "local_edit"):
        groups = {}
        identities = {(row["modality"], row["generator_id"]) for row in rows if row["task"] == task}
        for modality, generator in sorted(identities):
            subset = [row for row in rows if row["modality"] == modality
                      and (row["task"] == "real" or row["task"] == task and row["generator_id"] == generator)]
            groups[f"{modality}:{generator}"] = classification_metrics(subset, threshold)
        aucs = [values["auroc"] for values in groups.values() if values["auroc"] is not None]
        result["task_detection"][task] = groups
        result[f"{task}_auc"] = float(np.mean(aucs)) if aucs else None
    localized = [row for row in rows if row["dice"] is not None]
    groups = defaultdict(list)
    for row in localized:
        groups[f"{row['modality']}:{row['generator_id']}"].append(row)
    dice = {key: source_macro(values, "dice") for key, values in groups.items()}
    result.update(localization_by_group=dice,
                  source_macro_dice=source_macro(localized, "dice"),
                  source_macro_iou=source_macro(localized, "iou"),
                  macro_dice=float(np.mean(list(dice.values()))) if dice else None,
                  worst_group_dice=min(dice.values()) if dice else None,
                  small_region_dice=source_macro([row for row in localized
                                                 if row["mask_area"] < config["loss"]["small_region_area"]], "dice"),
                  image_threshold=threshold, mask_threshold=config["validation"]["mask_threshold"])
    return result


def selection_score(metrics, config):
    weights = config["validation"]["selection_weights"]
    missing = [key for key, weight in weights.items() if weight > 0 and metrics[key] is None]
    if missing:
        raise ValueError(f"Validation is missing required selection strata: {', '.join(missing)}")
    return sum(weight * metrics[key] for key, weight in weights.items() if weight > 0)


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["image"], batch["modality_id"])
        if not all(bool(torch.isfinite(output[key]).all()) for key in ("image_logit", "mask_logits")):
            raise FloatingPointError("Non-finite evaluation output")
        prediction = output["mask_probability"] >= config["validation"]["mask_threshold"]
        target = batch["mask"] > 0
        intersection = (prediction & target).sum((1, 2, 3))
        predicted_area = prediction.sum((1, 2, 3))
        target_area = target.sum((1, 2, 3))
        for index in range(len(batch["label"])):
            row = {key: batch[key][index] for key in
                   ("sample_id", "group_id", "modality", "task", "generator_id")}
            valid = bool(batch["mask_valid"][index]) and row["task"] == "local_edit" and int(target_area[index]) > 0
            overlap = int(intersection[index])
            total = int(predicted_area[index] + target_area[index])
            row.update(label=int(batch["label"][index]), image_score=float(output["image_score"][index]),
                       local_score=float(output["local_score"][index]),
                       mask_area=float(target[index].float().mean()),
                       predicted_area=float(prediction[index].float().mean()),
                       dice=2 * overlap / total if valid else None,
                       iou=overlap / (total - overlap) if valid else None)
            rows.append(row)
    return summarize(rows, config), rows
