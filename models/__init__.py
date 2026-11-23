"""Original ViTTT architectures; checkpoint handling lives outside the model."""
from collections.abc import Mapping
from pathlib import Path

import torch

from .vittt import vittt_tiny, vittt_small, vittt_base

MODELS = {"tiny": vittt_tiny, "small": vittt_small, "base": vittt_base}
WEIGHT_URLS = {
    "tiny": "https://cloud.tsinghua.edu.cn/f/e12d1bd1c20c4e06a2da/?dl=1",
    "small": "https://cloud.tsinghua.edu.cn/f/4ab59ca54b474dff9b70/?dl=1",
    "base": "https://cloud.tsinghua.edu.cn/f/c9e9bb09f85d449ab8cf/?dl=1",
}


def build_model(name):
    # Keep the upstream constructors and inference architecture unchanged.
    return MODELS[name](pretrained=False)


def load_weights(model, path, key="model"):
    # Upstream training checkpoints contain a pickled YACS config. Only load
    # trusted checkpoints (the official releases or your own experiment files).
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state_dict or a mapping containing one")
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state = checkpoint
    else:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint key {key!r} not found; available: {list(checkpoint)}")
        state = checkpoint[key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty or invalid model state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    # Never silently accept missing, unexpected, or shape-mismatched weights.
    model.load_state_dict(state, strict=True)


def download_weights(name, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"vittt_{name}.pth"
    if destination.exists():
        load_weights(build_model(name), destination)
        return destination
    temporary = destination.with_suffix(".download")
    try:
        torch.hub.download_url_to_file(WEIGHT_URLS[name], str(temporary))
        load_weights(build_model(name), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
