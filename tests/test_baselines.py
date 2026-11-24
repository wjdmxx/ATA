"""Small checks for the new parameter scopes, Fisher, filtering and W0 updates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import EATA, SAR, DeYO, LambdaW0
from models.vittt import VisionTransformer
from utils import load_config


def model():
    return VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                             depth=3, num_heads=2, num_classes=5).eval()


class NewMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = model()
        self.images = torch.randn(4, 3, 12, 12)
        self.config = load_config()["method"]

    def test_lambda_w0_two_learning_rates_update_and_reset(self):
        config = load_config([ROOT / "configs/lambda_w0.yaml"])["method"]
        config.update(layers=[0, 2], branches={"swiglu": True, "conv": False}, w0_lr=0.01)
        # Make this randomly initialized tiny fixture's W0 step visible in FP32.
        with torch.no_grad():
            self.model.head.weight.mul_(50)
        before = deepcopy(self.model.state_dict())
        expected = self.model(self.images).detach()
        method = LambdaW0(self.model, config)
        self.assertEqual([g["lr"] for g in method.optimizer.param_groups], [0.3, 0.01])
        self.assertEqual(set(method.parameter_names), {
            f"blocks.{i}.attn.{key}" for i in (0, 2) for key in ("theta_swiglu", "w1", "w2")})
        logits, _ = method.step(self.images)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        for name, parameter in self.model.named_parameters():
            if name in method.parameter_names:
                reference = before.get(name, torch.zeros_like(parameter))
                self.assertFalse(torch.equal(parameter, reference), name)
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                self.assertFalse(parameter.requires_grad)
        method.reset()
        self.assertFalse(method.optimizer.state)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.images), expected, rtol=0, atol=0)

    def test_official_layer_scopes_and_eval(self):
        # Same block/norm names as ViTTT-S/B, using a small channel count.
        source = VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                                   depth=12, num_heads=2, num_classes=5)
        for cls, count in ((EATA, 50), (SAR, 36), (DeYO, 36)):
            method = cls(deepcopy(source), self.config)
            self.assertFalse(method.model.training)
            self.assertEqual(len(method.parameter_names), count)
            self.assertEqual(set(method.parameter_names), {
                name for name, parameter in method.model.named_parameters() if parameter.requires_grad})
            self.assertEqual("norm.weight" in method.parameter_names, cls is EATA)
            if cls is not EATA:
                self.assertFalse(any(name.startswith("blocks.9.") for name in method.parameter_names))

    def test_fisher_uses_pseudo_labels_and_does_not_change_source(self):
        method = EATA(self.model, self.config)
        before = deepcopy(self.model.state_dict())
        loader = DataLoader(TensorDataset(self.images, torch.zeros(4, dtype=torch.long)), batch_size=2)
        method.estimate_fisher(loader)
        fishers = {name: value[0].clone() for name, value in method.fishers.items()}
        other_labels = DataLoader(TensorDataset(self.images, torch.full((4,), 4)), batch_size=2)
        method.estimate_fisher(other_labels)
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)
        for name, value in fishers.items():
            torch.testing.assert_close(value, method.fishers[name][0], rtol=0, atol=0)
        self.assertTrue(any(value.count_nonzero() for value in fishers.values()))
        self.assertFalse(method.optimizer.state)

    def test_empty_filters_and_sar_cross_domain_momentum_reset(self):
        config = deepcopy(self.config)
        config["eata"]["e_margin"] = 0
        config["deyo"]["deyo_margin"] = 0
        config["sar"]["margin_e0"] = 0
        for cls in (EATA, DeYO):
            method = cls(deepcopy(self.model), config)
            if cls is EATA:
                method.estimate_fisher(DataLoader(TensorDataset(self.images, torch.zeros(4)), batch_size=2))
            before = deepcopy(method.model.state_dict())
            _, metrics = method.step(self.images)
            self.assertEqual(metrics["updates"], 0)
            self.assertIsNone(metrics["loss"])
            for name, value in before.items():
                torch.testing.assert_close(value, method.model.state_dict()[name], rtol=0, atol=0)

        method = SAR(deepcopy(self.model), config)
        parameters = [p for p in method.model.parameters() if p.requires_grad]
        for parameter in parameters:
            method.optimizer.base_optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
        before = parameters[0].detach().clone()
        _, metrics = method.step(self.images)
        self.assertIsNone(metrics["loss"])
        torch.testing.assert_close(parameters[0], before - 0.9 * config["optimizer"]["lr"], rtol=0, atol=1e-7)
        method.reset()
        self.assertFalse(method.optimizer.base_optimizer.state)
        self.assertIsNone(method.ema)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
