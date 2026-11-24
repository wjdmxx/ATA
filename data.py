"""ImageNet / ImageNet-C: fixed sample identities and order across methods."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode

STANDARD = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")


def get_domains(config):
    name = config.get("name", "imagenet_c")
    group = config["group"]

    if name == "imagenet":
        if group != "clean" or config["domains"] not in (None, ["clean"]):
            raise ValueError("Clean ImageNet requires data.group=clean and domains=null or [clean]")
        return ["clean"]

    if name != "imagenet_c":
        raise ValueError("data.name must be imagenet or imagenet_c")
    groups = {"standard": STANDARD, "extra": EXTRA}
    if group not in groups:
        raise ValueError("data.group must be standard or extra")

    domains = config["domains"] if config["domains"] is not None else list(groups[group])
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("data.domains must be nonempty and unique")

    invalid = set(domains) - set(groups[group])
    if invalid:
        raise ValueError(f"Domains {sorted(invalid)} do not belong to group {group!r}")

    return list(domains)


def get_tasks(config):
    if config.get("name", "imagenet_c") == "imagenet":
        get_domains(config)
        return [("clean", None)]

    return [
        (domain, severity)
        for domain in get_domains(config)
        for severity in config["severities"]
    ]


def task_folder(config, domain, severity):
    root = Path(config["root"])
    if config.get("name", "imagenet_c") == "imagenet":
        return root
    return root / domain / str(severity)


def build_transform(config):
    operations = []
    if config["resize"] is not None:
        interpolation = {
            "bicubic": InterpolationMode.BICUBIC,
            "bilinear": InterpolationMode.BILINEAR,
        }[config["interpolation"]]
        operations.append(transforms.Resize(config["resize"], interpolation=interpolation))

    operations.extend([
        transforms.CenterCrop(config["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    return transforms.Compose(operations)


def manifest_digest(manifest):
    content = {k: manifest[k] for k in ("classes", "samples")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def validate_manifest(manifest, num_classes=1000):
    classes, samples = manifest["classes"], manifest["samples"]
    if len(classes) != num_classes or classes != sorted(set(classes)):
        raise ValueError(f"Expected {num_classes} sorted, unique ImageNet class folders")
    if not samples:
        raise ValueError("Sample manifest is empty")

    paths = [sample["path"] for sample in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate images in sample manifest")

    labels = {name: index for index, name in enumerate(classes)}
    for item in samples:
        parts = Path(item["path"]).parts
        if len(parts) != 2 or parts[0] not in labels:
            raise ValueError(f"Expected class/image in sample manifest: {item['path']}")
        if item["label"] != labels[parts[0]]:
            raise ValueError(f"Class label mismatch: {item}")

    if manifest.get("sha256") != manifest_digest(manifest):
        raise ValueError("Sample manifest hash mismatch")


def make_manifest(config, *, seed, num_classes=1000):
    """Budget is per domain/severity, or over the clean validation set.

    Build once from the first task; reuse relative image paths everywhere.
    An explicitly provided manifest preserves its sample membership and order.
    """
    requested = config["samples"]

    if config["manifest"]:
        manifest = json.loads(Path(config["manifest"]).read_text())
        validate_manifest(manifest, num_classes)
        if len(manifest["samples"]) != requested:
            raise ValueError("Manifest length differs from data.samples")
        return manifest

    domain, severity = get_tasks(config)[0]
    folder = task_folder(config, domain, severity)
    dataset = datasets.ImageFolder(folder)
    if len(dataset.classes) != num_classes:
        raise ValueError(f"{folder}: expected {num_classes} classes, got {len(dataset.classes)}")
    if requested > len(dataset):
        raise ValueError(f"Requested {requested} images, but {folder} only has {len(dataset)}")

    rng = np.random.default_rng(seed)
    if config["sampling"] == "balanced":
        if requested % num_classes:
            raise ValueError("Balanced sampling requires data.samples divisible by 1000")
        quota = requested // num_classes
        selected = []
        targets = np.asarray(dataset.targets)

        for label in range(num_classes):
            candidates = np.flatnonzero(targets == label)
            if len(candidates) < quota:
                raise ValueError(f"Class {dataset.classes[label]} has fewer than {quota} images")
            selected.extend(rng.choice(candidates, quota, replace=False).tolist())
    elif config["sampling"] == "random":
        selected = rng.choice(len(dataset), requested, replace=False).tolist()
    else:
        raise ValueError("data.sampling must be balanced or random")

    selected.sort()
    if config["shuffle"]:
        np.random.default_rng(seed).shuffle(selected)

    samples = [
        {
            "path": Path(dataset.samples[i][0]).relative_to(folder).as_posix(),
            "label": int(dataset.samples[i][1]),
        }
        for i in selected
    ]
    manifest = {
        "classes": dataset.classes,
        "samples": samples,
        "selection": {
            "seed": seed,
            "samples": requested,
            "sampling": config["sampling"],
            "shuffle": config["shuffle"],
        },
    }
    manifest["sha256"] = manifest_digest(manifest)

    return manifest


class ImageNetImages(Dataset):
    def __init__(self, folder, manifest, transform):
        self.folder = Path(folder)
        self.samples = manifest["samples"]
        self.transform = transform

        classes = sorted(p.name for p in self.folder.iterdir() if p.is_dir())
        if classes != manifest["classes"]:
            raise ValueError(f"Class folders differ from fixed label mapping: {self.folder}")

        for sample in self.samples:
            path = self.folder / sample["path"]
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image = default_loader(self.folder / item["path"])

        return self.transform(image), item["label"]


def build_fisher_loader(data_config, options, seed):
    """Official EATA: random clean images, followed by a shuffled batch order."""
    folder = Path(os.path.expandvars(options["fisher_root"])).expanduser()
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    source = datasets.ImageFolder(folder)
    size = options["fisher_size"]
    if not 0 < size <= len(source):
        raise ValueError(f"Fisher sample budget {size} exceeds clean dataset size {len(source)}")

    indices = list(range(len(source)))
    random.Random(seed).shuffle(indices)
    indices = indices[:size]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    samples = [{"path": Path(source.samples[indices[i]][0]).relative_to(folder).as_posix(),
                "label": source.samples[indices[i]][1]} for i in order]
    manifest = {"classes": source.classes, "samples": samples,
                "selection": {"seed": seed, "samples": size, "sampling": "random", "shuffle": True}}
    manifest["sha256"] = manifest_digest(manifest)

    transform_config = dict(data_config, resize=256, center_crop=224, interpolation="bilinear")
    dataset = ImageNetImages(folder, manifest, build_transform(transform_config))
    loader = DataLoader(dataset, batch_size=options["fisher_batch_size"], shuffle=False,
                        num_workers=data_config["workers"], generator=generator)
    return loader, manifest
