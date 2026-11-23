"""Configuration and experiment files, kept in one place."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import uuid

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def merge_config(base, update, prefix=""):
    for key, value in update.items():
        if key not in base:
            raise KeyError(f"Unknown config key: {prefix}{key}")

        if isinstance(base[key], dict):
            merge_config(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = deepcopy(value)


def load_config(paths=(), overrides=()):
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    for path in paths:
        merge_config(config, yaml.safe_load(Path(path).read_text()))

    for override in overrides:
        key, value = override.split("=", 1)
        nested = yaml.safe_load(value)
        for component in reversed(key.split(".")):
            nested = {component: nested}
        merge_config(config, nested)

    path_fields = [("model", "checkpoint"), ("data", "root"),
                   ("data", "manifest"), ("run", "output")]
    for section, key in path_fields:
        if config[section][key]:
            path = Path(os.path.expandvars(config[section][key])).expanduser()
            config[section][key] = str(path if path.is_absolute() else ROOT / path)

    validate_config(config)
    return config


def validate_config(config):
    """Check experiment choices here; PyTorch validates optimizer arguments."""
    from data import get_domains

    data = config["data"]
    get_domains(data)
    severities = data["severities"]

    if data["name"] == "imagenet":
        if severities != []:
            raise ValueError("Clean ImageNet requires data.severities=[]")
    elif (not severities
            or any(type(x) is not int or x not in range(1, 6) for x in severities)
            or len(set(severities)) != len(severities)):
        raise ValueError("data.severities must contain unique integers in 1..5")

    if data["samples"] <= 0:
        raise ValueError("data.samples must be positive")
    if data["sampling"] == "balanced" and data["samples"] % 1000:
        raise ValueError("Balanced ImageNet sampling requires data.samples divisible by 1000")

    seed = config["run"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("run.seed must be an integer in [0, 2**32)")
    if config["method"]["steps"] < 1 or config["run"]["log_every"] < 1:
        raise ValueError("method.steps and run.log_every must be positive")

    choices = [
        ("protocol", "reset", ("task", "domain", "batch", "never")),
        ("method", "prediction", ("before_update", "after_update")),
        ("run", "save_weights", ("none", "last", "task")),
    ]
    for section, key, allowed in choices:
        if config[section][key] not in allowed:
            raise ValueError(f"{section}.{key} must be one of {allowed}")


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_run_dir(config):
    name = str(config["run"]["name"])
    if Path(name).name != name:
        raise ValueError("run.name must be a single directory name")

    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    directory = Path(config["run"]["output"]) / f"{timestamp}_{name}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)

    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


def make_logger(directory):
    logger = logging.getLogger(str(directory))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "run.log")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def snapshot_code(directory):
    files = list(ROOT.glob("*.py")) + list((ROOT / "models").glob("*.py"))
    files += list((ROOT / "configs").glob("*.yaml"))
    files += [ROOT / "models/UPSTREAM.json", ROOT / "requirements.txt"]

    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        destination = directory / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha256_file(path)

    return hashes


def environment_info(device):
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "PyYAML"):
        packages[name] = importlib.metadata.version(name)

    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_config(config):
    """Execution location and logging may change; scientific settings may not."""
    result = deepcopy(config)

    if result["method"]["name"] != "lambda":
        for key in ("lambda_max", "branches", "layers"):
            result["method"].pop(key, None)

    # Checkpoints saved before clean ImageNet support were ImageNet-C runs.
    result["data"].setdefault("name", "imagenet_c")
    # Compare old three-seed checkpoints by their actual data RNG settings.
    # A new single-seed run matches only when both historical data seeds match.
    result["data"].setdefault("sample_seed", result["run"]["seed"])
    result["data"].setdefault("order_seed", result["run"]["seed"])

    for key in ("output", "name", "device", "log_every", "save_weights"):
        result["run"].pop(key)
    result["data"].pop("workers")
    # Sample identities and order are checked independently by their digest.
    result["data"].pop("manifest")

    return result
