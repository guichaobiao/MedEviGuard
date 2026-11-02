import torch
from torch import nn
import torch.nn.functional as F


LOSS_NAMES = ("dense", "auxiliary", "image", "local", "original_negative", "regional_margin",
              "teacher", "distillation", "real_pathology", "small_region", "paired_features",
              "agreement", "appearance")


def gated_mean(values, valid):
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class MedEviGuardLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weights = config["weights"]
        if set(self.weights) != set(LOSS_NAMES):
            raise ValueError("Every objective must have an explicit loss weight")

    def dense(self, logits, target):
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="area")
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        axes = (1, 2, 3)
        positive = target.sum(axes)
        negative = (1 - target).sum(axes)
        balanced = ((bce * target).sum(axes) / positive.clamp_min(1)
                    + (bce * (1 - target)).sum(axes) / negative.clamp_min(1)) / 2
        balanced = torch.where((positive > 0) & (negative > 0), balanced, bce.mean(axes))
        probability = logits.sigmoid()
        true_positive = (probability * target).sum(axes)
        false_positive = (probability * (1 - target)).sum(axes)
        false_negative = ((1 - probability) * target).sum(axes)
        smooth = self.config["smoothing"]
        denominator = (true_positive + self.config["tversky_false_positive"] * false_positive
                       + self.config["tversky_false_negative"] * false_negative + smooth)
        tversky = 1 - (true_positive + smooth) / denominator
        return balanced + tversky * (positive > 0)

    def forward(self, output, batch, original=None, teacher=None, identity_teacher=None, view=None, epoch=0):
        masks = batch["mask"].float()
        labels = batch["label"].float()
        valid = batch["mask_valid"].bool()
        local_valid = batch["local_label_valid"].bool()
        zero = output["mask_logits"].sum() * 0
        losses = {key: zero for key in LOSS_NAMES}
        dense = self.dense(output["mask_logits"], masks)
        losses["dense"] = gated_mean(dense, valid)
        losses["auxiliary"] = torch.stack([gated_mean(self.dense(value, masks), valid)
                                           for value in output["auxiliary_logits"]]).mean()
        losses["image"] = F.binary_cross_entropy_with_logits(output["image_logit"], labels)
        losses["local"] = gated_mean(F.binary_cross_entropy_with_logits(
            output["local_logit"], batch["local_label"].float(), reduction="none"), local_valid)
        area = masks.mean((1, 2, 3))
        local = valid & (labels == 1) & (area > 0)
        paired = batch["pair_valid"].bool() & local
        losses["small_region"] = gated_mean(dense, local & (area < self.config["small_region_area"]))
        losses["agreement"] = gated_mean((output["image_score"] - output["local_score"]).square(), local_valid)
        lesion = batch["lesion_mask"].float() * (labels == 0)[:, None, None, None]
        losses["real_pathology"] = (output["mask_probability"] * lesion).sum() / lesion.sum().clamp_min(1)
        if original is not None:
            negative = F.softplus(original["mask_logits"]).mean((1, 2, 3)) + F.softplus(original["image_logit"])
            losses["original_negative"] = gated_mean(negative, paired)
            gap = output["mask_probability"] - original["mask_probability"]
            margin = (F.relu(self.config["regional_margin"] - gap) * masks).sum((1, 2, 3))
            losses["regional_margin"] = gated_mean(margin / masks.sum((1, 2, 3)).clamp_min(1), paired)
            distance = 1 - (output["embedding"] * original["embedding"].detach()).sum(1, keepdim=True)
            support = F.interpolate(masks, size=distance.shape[-2:], mode="area")
            inside = (F.relu(self.config["feature_margin"] - distance) * support).sum((1, 2, 3))
            inside = inside / support.sum((1, 2, 3)).clamp_min(1)
            outside = (distance * (1 - support)).sum((1, 2, 3))
            outside = outside / (1 - support).sum((1, 2, 3)).clamp_min(1)
            losses["paired_features"] = gated_mean(inside + self.config["outside_weight"] * outside, paired)
        if teacher is not None:
            if identity_teacher is None:
                raise ValueError("Teacher supervision requires an authentic identity pair")
            losses["teacher"] = gated_mean(self.dense(teacher, masks)
                                             + F.softplus(identity_teacher).mean((1, 2, 3)), paired)
            if epoch >= self.config["teacher_warmup_epochs"]:
                probability = teacher.detach().sigmoid()
                target = F.interpolate(masks, size=probability.shape[-2:], mode="area")
                compatible = (probability - 0.5) * (target - 0.5) > 0
                confidence = (2 * probability - 1).abs() * compatible * paired[:, None, None, None]
                logits = F.interpolate(output["mask_logits"], size=probability.shape[-2:],
                                       mode="bilinear", align_corners=False)
                error = F.binary_cross_entropy_with_logits(logits, probability, reduction="none")
                losses["distillation"] = (error * confidence).sum() / confidence.sum().clamp_min(1)
        if view is not None:
            losses["appearance"] = (gated_mean(self.dense(view["mask_logits"], masks), valid)
                                     + self.config["appearance_image_weight"]
                                     * F.binary_cross_entropy_with_logits(view["image_logit"], labels))
        losses["total"] = sum(self.weights[name] * losses[name] for name in LOSS_NAMES)
        return losses
