import argparse
import json
from pathlib import Path

from .config import ConfigurationError, load_config


def build_parser():
    parser = argparse.ArgumentParser(prog="medeviguard")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-config")
    check.add_argument("--config", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", required=True)
    training.add_argument("--resume")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--data-root", required=True)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", required=True)
    evaluation.add_argument("--backbone-directory")
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--image", required=True)
    inference.add_argument("--modality", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", required=True)
    inference.add_argument("--backbone-directory")
    return parser


def infer(args):
    import numpy as np
    from PIL import Image
    import torch

    from .checkpoint import load_student
    from .data import load_image
    from .layers import resize
    from .model import MODALITIES

    if args.modality not in MODALITIES:
        raise ValueError(f"Modality must be one of {', '.join(MODALITIES)}")
    directory = Path(args.output).expanduser()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Choose an empty prediction directory")
    device = torch.device(args.device)
    model, config = load_student(args.checkpoint, device, args.backbone_directory)
    image = load_image(Path(args.image).expanduser(), args.modality)
    native_size = image.shape[-2:]
    pixels = resize(image[None], (config["model"]["input_size"],) * 2).to(device)
    modality = torch.tensor([MODALITIES.index(args.modality)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(pixels, modality)
        if not all(bool(torch.isfinite(output[key]).all()) for key in ("image_logit", "mask_logits")):
            raise FloatingPointError("Non-finite inference output")
        model_probability = output["mask_probability"][0, 0].cpu().numpy()
        native_probability = resize(output["mask_probability"], native_size)[0, 0].cpu().numpy()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "probabilities.npz", model_probability=model_probability,
                        original_size_probability=native_probability)
    mask = (native_probability >= config["validation"]["mask_threshold"]).astype(np.uint8) * 255
    Image.fromarray(mask).save(directory / "mask.png")
    result = {"image_score": float(output["image_score"][0]), "local_score": float(output["local_score"][0]),
              "flagged": bool(output["image_score"][0] >= config["validation"]["image_threshold"]),
              "modality": args.modality, "input_height": native_size[0], "input_width": native_size[1],
              "image_threshold": config["validation"]["image_threshold"],
              "mask_threshold": config["validation"]["mask_threshold"]}
    (directory / "prediction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "check-config":
            load_config(args.config)
            print("Configuration is complete and valid")
            return
        if args.command == "train":
            config = load_config(args.config)
            import torch
            from .engine import train
            train(config, args.output, torch.device(args.device), args.resume)
        elif args.command == "evaluate":
            import torch
            from .checkpoint import load_student
            from .data import build_loader
            from .metrics import evaluate
            device = torch.device(args.device)
            model, config = load_student(args.checkpoint, device, args.backbone_directory)
            config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
            loader = build_loader(config, args.split, args.manifest)
            metrics, rows = evaluate(model, loader, device, config)
            destination = Path(args.output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"metrics": metrics, "predictions": rows}, indent=2,
                                               allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps(metrics, allow_nan=False))
        elif args.command == "infer":
            infer(args)
    except (ConfigurationError, ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
