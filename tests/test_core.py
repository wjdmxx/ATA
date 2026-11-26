"""Fast scientific-contract checks; no real dataset or network required."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import (EXTRA, STANDARD, ImageNetImages, build_transform, get_domains,
                  get_tasks, make_manifest, task_folder, validate_manifest)
from methods import InfoMax, Source, Tent
from models import build_model, load_weights
from run import report_top1, run, should_reset
from utils import (atomic_save, capture_rng, load_config, restore_rng, resume_config,
                   seed_everything)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.config = load_config()["data"]
        self.config.update(root=str(self.root), domains=["gaussian_noise"],
                           severities=[1], samples=4, sampling="balanced", workers=0)
        for domain in ("gaussian_noise", "shot_noise"):
            for label, name in enumerate(("n00000001", "n00000002")):
                folder = self.root / domain / "1" / name
                folder.mkdir(parents=True)
                for i in range(5):
                    Image.new("RGB", (20, 20), color=(label * 50, i * 20, 0)).save(folder / f"{i}.JPEG")

    def tearDown(self):
        self.scratch.cleanup()

    def test_group_separation_and_counts(self):
        self.assertEqual(len(STANDARD), 15)
        self.assertEqual(len(EXTRA), 4)
        self.assertFalse(set(STANDARD) & set(EXTRA))
        self.assertEqual(set(EXTRA), {"speckle_noise", "gaussian_blur", "spatter", "saturate"})
        config = load_config()["data"]
        self.assertEqual(len(get_tasks(config)), 15)
        config.update(group="extra", severities=[1, 2, 3, 4, 5])
        self.assertEqual(len(get_tasks(config)), 20)
        config["domains"] = ["gaussian_noise"]
        with self.assertRaises(ValueError):
            get_domains(config)

    def test_imagenet_c_keeps_224_image_pixels(self):
        pixels = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        expected = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(image)

        for preset in ["test_5k.yaml", "test_50k.yaml", "val_5k.yaml", "val_50k.yaml"]:
            config = load_config([ROOT / "configs" / preset])["data"]
            self.assertIsNone(config["resize"])
            self.assertEqual(config["batch_size"], 64)
            actual = build_transform(config)(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_clean_imagenet_preprocessing_is_unchanged(self):
        pixels = np.random.default_rng(0).integers(0, 256, (320, 400, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        expected = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(image)

        for preset in ["imagenet_5k.yaml", "imagenet_50k.yaml"]:
            config = load_config([ROOT / "configs" / preset])["data"]
            self.assertEqual(config["resize"], 256)
            self.assertEqual(config["batch_size"], 64)
            torch.testing.assert_close(build_transform(config)(image), expected, rtol=0, atol=0)

    def test_balanced_reproducible_samples_and_cross_domain_alignment(self):
        manifest = make_manifest(self.config, seed=0, num_classes=2)
        self.assertEqual(manifest, make_manifest(self.config, seed=0, num_classes=2))
        self.assertEqual(np.bincount([s["label"] for s in manifest["samples"]]).tolist(), [2, 2])
        other = deepcopy(self.config)
        other["domains"] = ["shot_noise"]
        self.assertEqual(manifest, make_manifest(other, seed=0, num_classes=2))
        datasets = [ImageNetImages(self.root / domain / "1", manifest, build_transform(self.config))
                    for domain in ("gaussian_noise", "shot_noise")]
        for first, second in zip(datasets[0], datasets[1]):
            self.assertEqual(first[1], second[1])
            torch.testing.assert_close(first[0], second[0])

    def test_manifest_reuse_and_missing_images_fail(self):
        manifest = make_manifest(self.config, seed=0, num_classes=2)
        path = self.root / "samples.json"
        path.write_text(json.dumps(manifest))
        self.config["manifest"] = str(path)
        self.assertEqual(manifest, make_manifest(self.config, seed=0, num_classes=2))
        self.config["samples"] = 2
        with self.assertRaises(ValueError):
            make_manifest(self.config, seed=0, num_classes=2)
        (self.root / "shot_noise/1" / manifest["samples"][0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            ImageNetImages(self.root / "shot_noise/1", manifest, build_transform(self.config))
        corrupted = deepcopy(manifest)
        corrupted["samples"][0]["label"] = 1 - corrupted["samples"][0]["label"]
        with self.assertRaises(ValueError):
            validate_manifest(corrupted, num_classes=2)

    def test_full_budget_and_no_silent_truncation(self):
        self.config["samples"] = 10
        self.assertEqual(len(make_manifest(self.config, seed=0, num_classes=2)["samples"]), 10)
        self.config["samples"] = 12
        with self.assertRaises(ValueError):
            make_manifest(self.config, seed=0, num_classes=2)

    def test_one_seed_controls_membership_and_order(self):
        first = make_manifest(self.config, seed=0, num_classes=2)
        repeated = make_manifest(self.config, seed=0, num_classes=2)
        changed = make_manifest(self.config, seed=42, num_classes=2)
        self.assertEqual(first, repeated)
        self.assertEqual(changed["selection"]["seed"], 42)
        self.assertNotEqual({s["path"] for s in first["samples"]},
                            {s["path"] for s in changed["samples"]})
        # All images stay selected for a full-budget run; the seed changes order.
        self.config["samples"] = 10
        first = make_manifest(self.config, seed=0, num_classes=2)
        changed = make_manifest(self.config, seed=42, num_classes=2)
        self.assertEqual({s["path"] for s in first["samples"]},
                         {s["path"] for s in changed["samples"]})
        self.assertNotEqual(first["samples"], changed["samples"])

    def test_clean_imagenet_uses_one_task_and_shared_manifest(self):
        clean_root = self.root / "val"
        shutil.copytree(self.root / "gaussian_noise/1", clean_root)
        corrupted = make_manifest(self.config, seed=0, num_classes=2)
        self.config.update(name="imagenet", root=str(clean_root), group="clean",
                           domains=None, severities=[])
        self.assertEqual(get_tasks(self.config), [("clean", None)])
        folder = task_folder(self.config, "clean", None)
        self.assertEqual(folder, clean_root)
        manifest = make_manifest(self.config, seed=0, num_classes=2)
        self.assertEqual(manifest, corrupted)
        dataset = ImageNetImages(folder, manifest, build_transform(self.config))
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(label, manifest["samples"][0]["label"])
        path = self.root / "clean_samples.json"
        path.write_text(json.dumps(manifest))
        self.config["manifest"] = str(path)
        self.assertEqual(manifest, make_manifest(self.config, seed=0, num_classes=2))


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.config = load_config()["method"]
        self.model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 5))
        self.images = torch.randn(6, 4)

    def test_source_is_immutable(self):
        before = deepcopy(self.model.state_dict())
        method = Source(self.model, self.config)
        logits, _ = method.step(self.images)
        self.assertFalse(logits.requires_grad)
        for key, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[key], rtol=0, atol=0)

    def test_only_selected_norm_parameters_update_and_reset(self):
        before = deepcopy(self.model.state_dict())
        expected = self.model(self.images).detach()
        method = Tent(self.model, self.config)
        logits, _ = method.step(self.images)
        torch.testing.assert_close(expected, logits, rtol=0, atol=0)
        self.assertEqual(method.parameter_names, ["1.weight", "1.bias"])
        self.assertFalse(torch.equal(before["1.weight"], self.model[1].weight))
        for key in ("0.weight", "0.bias", "2.weight", "2.bias"):
            torch.testing.assert_close(before[key], self.model.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(method.optimizer.state)
        method.reset()
        self.assertEqual(method.updates, 0)
        self.assertFalse(method.optimizer.state)
        for key, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[key], rtol=0, atol=0)

    def test_infomax_and_after_update_predictions(self):
        self.config.update(prediction="after_update", diversity_weight=0.7)
        method = InfoMax(self.model, self.config)
        uniform = torch.zeros(4, 5, requires_grad=True)
        loss, _ = method.loss(uniform)
        self.assertAlmostEqual(float(loss), 0.3 * np.log(5), places=6)
        logits, _ = method.step(self.images)
        with torch.no_grad():
            torch.testing.assert_close(logits, self.model(self.images), rtol=0, atol=0)

    def test_checkpoint_continues_optimizer_and_reset_restores_source(self):
        second_model = deepcopy(self.model)
        first = Tent(self.model, self.config)
        second = Tent(second_model, self.config)
        first.step(self.images)
        second.load_state_dict(deepcopy(first.state_dict()))
        first_logits, _ = first.step(self.images * 2)
        second_logits, _ = second.step(self.images * 2)
        torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
        second.reset()
        for key, value in second.initial_model.items():
            torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)


class ExperimentTests(unittest.TestCase):
    def test_copyable_top1_report(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            log = directory / "run.log"
            log.write_text("existing experiment log\n")
            summary = {"by_domain": {"snow": 43.144, "fog": 28.726}, "mean_top1": 35.935}

            with redirect_stdout(io.StringIO()) as output:
                report_top1(directory, summary)

            expected = "43.14 28.73 35.94"
            self.assertEqual((directory / "top1.txt").read_text(), expected + "\n")
            self.assertEqual(output.getvalue().splitlines()[-1], expected)
            self.assertIn("Top-1 order: snow fog mean", log.read_text())
            self.assertTrue(log.read_text().startswith("existing experiment log\n"))
            self.assertEqual(log.read_text().splitlines()[-1], expected)

    def test_config_overlays_and_typos(self):
        config = load_config([ROOT / "configs/val_50k.yaml", ROOT / "configs/tent.yaml"],
                             ["model.name=small", "data.severities=[1,2,3,4,5]"])
        self.assertEqual(config["data"]["samples"], 50000)
        self.assertEqual(config["data"]["group"], "extra")
        self.assertEqual(config["method"]["name"], "tent")
        with self.assertRaises(KeyError):
            load_config(overrides=["method.optmizer.lr=0.1"])
        with self.assertRaises(ValueError):
            load_config(overrides=["data.severities=[5,5]"])

    def test_reset_boundaries(self):
        tasks = [("snow", 1), ("snow", 2), ("fog", 1)]
        self.assertEqual([should_reset("domain", i, tasks) for i in range(3)], [True, False, True])
        self.assertTrue(all(should_reset("task", i, tasks) for i in range(3)))
        self.assertFalse(any(should_reset("never", i, tasks) for i in range(3)))

    def test_clean_config_rejects_corruption_settings(self):
        for preset, count in [("imagenet_5k.yaml", 5000), ("imagenet_50k.yaml", 50000)]:
            config = load_config([ROOT / "configs" / preset])
            self.assertEqual(config["data"]["samples"], count)
            self.assertEqual(get_tasks(config["data"]), [("clean", None)])
        for override in ("data.severities=[5]", "data.group=standard",
                         "data.domains=[gaussian_noise]", "data.name=typo"):
            with self.assertRaises(ValueError):
                load_config([ROOT / "configs/imagenet_5k.yaml"], [override])

    def test_legacy_imagenet_c_resume_config_is_compatible(self):
        config = load_config()
        legacy = deepcopy(config)
        legacy["data"].pop("name")
        legacy["data"].update(sample_seed=0, order_seed=0)
        self.assertEqual(resume_config(config), resume_config(legacy))
        legacy["data"]["order_seed"] = 1
        self.assertNotEqual(resume_config(config), resume_config(legacy))

    def test_single_seed_configuration(self):
        config = load_config(overrides=["run.seed=42"])
        self.assertEqual(config["run"]["seed"], 42)
        self.assertNotIn("sample_seed", config["data"])
        self.assertNotIn("order_seed", config["data"])
        for key in ("sample_seed", "order_seed"):
            with self.assertRaises(KeyError):
                load_config(overrides=[f"data.{key}=42"])
        for value in ("-1", "1.5", "true", "4294967296"):
            with self.assertRaises(ValueError):
                load_config(overrides=[f"run.seed={value}"])

    def test_rng_round_trip(self):
        seed_everything(0, True)
        state = capture_rng(torch.device("cpu"))
        expected = (np.random.rand(), torch.rand(3))
        restore_rng(state, torch.device("cpu"))
        self.assertEqual(expected[0], np.random.rand())
        torch.testing.assert_close(expected[1], torch.rand(3), rtol=0, atol=0)

    def test_full_runner_and_task_resume(self):
        # A real 1000-class folder layout and original ViTTT model, just two
        # images per task for speed. Include a short final batch (drop_last=False).
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for domain in ("gaussian_noise", "shot_noise"):
                for i in range(1000):
                    folder = root / "data" / domain / "1" / f"n{i:08d}"
                    folder.mkdir(parents=True)
                    Image.new("RGB", (224, 224), (i % 255, 10, 20)).save(folder / "sample.JPEG")
            model = build_model("tiny")
            checkpoint = root / "source.pth"
            torch.save({"model": model.state_dict()}, checkpoint)
            config = load_config(overrides=[
                "run.device=cpu", f"run.output={root / 'outputs'}",
                f"model.checkpoint={checkpoint}", f"data.root={root / 'data'}",
                "data.domains=[gaussian_noise,shot_noise]", "data.severities=[1]",
                "data.sampling=random", "data.samples=3", "data.batch_size=2", "data.workers=0",
                "method.name=tent", "run.save_weights=task", "protocol.reset=never", "run.seed=17"])
            directory = run(config)
            manifest = json.loads((directory / "samples.json").read_text())
            self.assertEqual(manifest, make_manifest(config["data"], seed=17))
            self.assertNotEqual(manifest["sha256"], make_manifest(config["data"], seed=0)["sha256"])
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["seed"], 17)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["evaluated_images"], 6)
            self.assertEqual(summary["completed_tasks"], 2)
            resumed = run(config, directory / "checkpoints/gaussian_noise_s1.pt")
            first = torch.load(directory / "checkpoints/last.pt", weights_only=False)
            second = torch.load(resumed / "checkpoints/last.pt", weights_only=False)
            for key, value in first["model"].items():
                torch.testing.assert_close(value, second["model"][key], rtol=0, atol=0)
            for a, b in zip(first["results"], second["results"]):
                for key in ("domain", "severity", "samples", "correct1", "correct5", "updates"):
                    self.assertEqual(a[key], b[key])
            self.assertEqual(second["next_task"], 2)
            load_weights(build_model("tiny"), resumed / "checkpoints/last.pt")

            # The same runner handles clean classification with no fake severity.
            config["data"].update(name="imagenet", root=str(root / "data/gaussian_noise/1"),
                                  group="clean", domains=None, severities=[])
            config["method"]["name"] = "source"
            config["run"]["save_weights"] = load_config()["run"]["save_weights"]
            self.assertEqual(config["run"]["save_weights"], "none")

            with patch("torch.save", side_effect=AssertionError("Default run must not save weights")):
                clean_run = run(config)

            clean_summary = json.loads((clean_run / "summary.json").read_text())
            self.assertEqual(clean_summary["dataset"], "imagenet")
            self.assertEqual(clean_summary["completed_tasks"], 1)
            self.assertEqual(clean_summary["evaluated_images"], 3)
            top1_line = (clean_run / "top1.txt").read_text().strip()
            self.assertEqual(len(top1_line.split()), 1)
            self.assertEqual(float(top1_line), round(clean_summary["mean_top1"], 2))
            self.assertEqual((clean_run / "run.log").read_text().splitlines()[-1], top1_line)
            self.assertFalse((clean_run / "checkpoints").exists())
            self.assertFalse(list(clean_run.rglob("*.pt")))
            metrics = [json.loads(line) for line in (clean_run / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["domain"] == "clean" and row["severity"] is None for row in metrics))


class ModelTests(unittest.TestCase):
    def test_copied_files_are_byte_identical(self):
        upstream = json.loads((ROOT / "models/UPSTREAM.json").read_text())
        for name, expected in upstream["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "models" / name).read_bytes()).hexdigest(), expected)

    def test_upstream_forward_and_backward_equivalence(self):
        # Optional local reference check; standalone copies still run all other tests.
        original = ROOT.parent / "ViTTT/vittt/models"
        if not original.is_dir():
            self.skipTest("Original checkout unavailable")
        package = types.ModuleType("ata_reference")
        package.__path__ = [str(original)]
        sys.modules["ata_reference"] = package
        reference_module = importlib.import_module("ata_reference.vittt")
        current = build_model("tiny")
        reference = reference_module.vittt_tiny(pretrained=False)
        reference.load_state_dict(current.state_dict(), strict=True)
        current.eval()
        reference.eval()
        x = torch.randn(2, 3, 224, 224)
        current_logits = current(x)
        reference_logits = reference(x)
        torch.testing.assert_close(current_logits, reference_logits, rtol=0, atol=0)
        current_logits.square().mean().backward()
        reference_logits.square().mean().backward()
        for (name, p), (other_name, q) in zip(current.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_strict_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as scratch:
            model = build_model("tiny")
            path = Path(scratch) / "weights.pt"
            wrapped = {"module." + k: v for k, v in model.state_dict().items()}
            atomic_save(path, {"state_dict_ema": wrapped})
            load_weights(build_model("tiny"), path, key="state_dict_ema")
            wrapped.pop(next(iter(wrapped)))
            atomic_save(path, {"model": wrapped})
            with self.assertRaises(RuntimeError):
                load_weights(build_model("tiny"), path)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
