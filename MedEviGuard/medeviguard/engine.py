from collections import defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .augment import second_view
from .checkpoint import atomic_save, backbone_identity, file_hash, restore_student, student_state
from .data import build_loader, check_development_boundary, resolve
from .losses import MedEviGuardLoss
from .metrics import evaluate, move_batch, selection_score
from .model import MedEviGuard, PairEvidenceTeacher


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate_factor(update, total, warmup, minimum):
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1, (update - warmup) / max(1, total - warmup - 1))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def rng_state(device):
    numpy = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config, output_directory, device, resume=None):
    settings = config["training"]
    seed_all(settings["seed"])
    directory = Path(output_directory).expanduser()
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError("Choose an empty output directory or resume explicitly")
    training = build_loader(config, "train", config["data"]["train_manifest"])
    validation = build_loader(config, "val", config["data"]["validation_manifest"])
    check_development_boundary(training.dataset.records, validation.dataset.records, config["data"]["root"])
    manifests = {key: file_hash(resolve(config["data"]["root"], config["data"][key]))
                 for key in ("train_manifest", "validation_manifest")}
    identity = backbone_identity(config["model"]["backbone_directory"])
    model = MedEviGuard(config["model"]).to(device)
    teacher = PairEvidenceTeacher(model.spectral.channels, config["model"]).to(device)
    criterion = MedEviGuardLoss(config["loss"]).to(device)
    parameters = [parameter for module in (model, teacher) for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                                  betas=tuple(settings["adam_betas"]), eps=settings["adam_epsilon"])
    amp = settings["mixed_precision"] and device.type == "cuda"
    amp_dtype = getattr(torch, settings["amp_dtype"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)
    start, update, best = 0, 0, -math.inf
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("format") != "medeviguard" or state["config"] != config:
            raise ValueError("Resume requires the same architecture and experiment configuration")
        if state["backbone_identity"] != identity or state["manifest_hashes"] != manifests:
            raise ValueError("Resume data or backbone identity differs")
        restore_student(model, state["student"])
        teacher.load_state_dict(state["teacher"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, update, best = state["epoch"] + 1, state["update"], state["best"]
        restore_rng(state["rng"], device)
        del state
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    accumulate = settings["accumulation_steps"]
    updates_per_epoch = math.ceil(len(training) / accumulate)
    total_updates = settings["epochs"] * updates_per_epoch
    warmup_updates = settings["warmup_epochs"] * updates_per_epoch
    for epoch in range(start, settings["epochs"]):
        model.train()
        teacher.train()
        sums = defaultdict(float)
        for step, batch in enumerate(training):
            batch = move_batch(batch, device)
            if step % accumulate == 0:
                optimizer.zero_grad(set_to_none=True)
                factor = learning_rate_factor(update, total_updates, warmup_updates, settings["minimum_lr_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = settings["learning_rate"] * factor
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(batch["image"], batch["modality_id"])
                original = teacher_output = identity_output = view = None
                if bool(batch["pair_valid"].any()):
                    original = model(batch["original"], batch["modality_id"])
                    if criterion.weights["teacher"] > 0:
                        teacher_output = teacher(output, original)
                        identity_output = teacher(original, original)
                if criterion.weights["appearance"] > 0 and step % settings["appearance_interval"] == 0:
                    view = model(second_view(batch["image"], config["data"]["augmentation"]), batch["modality_id"])
                losses = criterion(output, batch, original, teacher_output, identity_output, view, epoch)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Non-finite training loss")
            group_size = min(accumulate, len(training) - (step // accumulate) * accumulate)
            scaler.scale(losses["total"] / group_size).backward()
            if (step + 1) % accumulate == 0 or step + 1 == len(training):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                update += 1
            for name, value in losses.items():
                sums[name] += float(value.detach())
            if step == 0 or (step + 1) % settings["log_interval"] == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step + 1, "loss": float(losses["total"].detach()),
                                  "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        metrics, _ = evaluate(model, validation, device, config)
        score = selection_score(metrics, config)
        improved = score > best
        best = max(best, score)
        state = {"format": "medeviguard", "config": config, "student": student_state(model),
                 "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "update": update, "best": best, "rng": rng_state(device),
                 "backbone_identity": identity, "manifest_hashes": manifests}
        atomic_save(state, directory / "last.pt")
        if improved:
            atomic_save(state, directory / "best.pt")
        del state
        record = {"epoch": epoch + 1, "training": {key: value / len(training) for key, value in sums.items()},
                  "validation": metrics, "selection": score}
        with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({"epoch": epoch + 1, "selection": score, "best": best}), flush=True)
    return directory
