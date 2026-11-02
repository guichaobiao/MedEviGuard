# MedEviGuard

Method implementation for **Evidence-guided open-set medical image authenticity verification across generative and acquisition shifts**.

MedEviGuard predicts a manipulation probability map and an image-level manipulation score from one medical image and its modality..

## Installation

```bash
python -m pip install -e .
```

Use a PyTorch installation providing `torch.amp.GradScaler` and a Transformers installation with DINOv3 ViT support. Prepare a local Transformers DINOv3 directory containing its pretrained weights, `config.json` and `preprocessor_config.json`. The loader works offline and checks that the model type is DINOv3 ViT. Native upstream `.pth` files must first be exported to that format with their matching architecture metadata.

```bash
cp configs/paper.yaml configs/experiment.local.yaml
python -m medeviguard check-config --config configs/experiment.local.yaml
```

## Training and evaluation

```bash
python -m medeviguard train --config configs/experiment.local.yaml --output outputs/experiment --device cuda
python -m medeviguard train --config configs/experiment.local.yaml --output outputs/experiment --device cuda --resume outputs/experiment/last.pt
python -m medeviguard evaluate --checkpoint outputs/experiment/best.pt --manifest test.jsonl --data-root /path/to/data --split test --output outputs/test.json --device cuda
```

## Verification

```bash
python -m pip install -e '.[test]'
python -m pytest
MEDEVIGUARD_TEST_CONFIG=/path/to/private/test.yaml python -m pytest
```
