import copy
import hashlib
from pathlib import Path

import torch

from .config import validate_config
from .model import MedEviGuard


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backbone_identity(directory):
    directory = Path(directory).expanduser()
    files = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix in (".json", ".bin", ".safetensors", ".pt", ".pth"))
    if not any(path.suffix in (".bin", ".safetensors", ".pt", ".pth") for path in files):
        raise ValueError("The backbone directory contains no pretrained weights")
    return {path.name: file_hash(path) for path in files}


def student_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith("foundation.network.")}


def restore_student(model, state):
    expected = {key for key in model.state_dict() if not key.startswith("foundation.network.")}
    if set(state) != expected:
        raise ValueError("Student checkpoint keys do not match the configured architecture")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("foundation.network.") for key in result.missing_keys):
        raise ValueError("Invalid student checkpoint")


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_student(checkpoint, device, backbone_directory=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "medeviguard":
        raise ValueError("Expected a MedEviGuard release checkpoint")
    config = copy.deepcopy(state["config"])
    if backbone_directory is not None:
        config["model"]["backbone_directory"] = str(Path(backbone_directory).expanduser().resolve())
    validate_config(config)
    if backbone_identity(config["model"]["backbone_directory"]) != state["backbone_identity"]:
        raise ValueError("Pretrained backbone files differ from the training checkpoint")
    model = MedEviGuard(config["model"]).to(device)
    restore_student(model, state["student"])
    return model.eval(), config
