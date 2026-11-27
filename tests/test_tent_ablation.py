"""Check Tent's LN ablations preserve frozen weights and experiment state."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import build_method
from models.vittt import VisionTransformer
from utils import load_config, resume_config


class TentAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                                       depth=12, num_heads=2, num_classes=5).eval()
        with torch.no_grad():
            self.model.head.weight.mul_(50)
        self.images = torch.randn(4, 3, 12, 12)

    def config(self, name):
        return load_config([ROOT / f"configs/{name}.yaml"])

    def assert_model_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_scopes_updates_and_reset_including_original_tent(self):
        source = deepcopy(self.model.state_dict())
        expected_logits = self.model(self.images).detach()
        for name, norms in (("tent", ("norm1", "norm2")),
                            ("tent_no_norm1", ("norm2",)),
                            ("tent_no_norm2", ("norm1",))):
            with self.subTest(method=name):
                method = build_method(deepcopy(self.model), self.config(name)["method"])
                expected = {f"blocks.{i}.{norm}.{key}" for i in range(12)
                            for norm in norms for key in ("weight", "bias")}
                expected.update(("norm.weight", "norm.bias"))
                self.assertEqual(set(method.parameter_names), expected)
                self.assertEqual({n for n, p in method.model.named_parameters()
                                  if p.requires_grad}, expected)
                self.assertEqual({id(p) for group in method.optimizer.param_groups
                                  for p in group["params"]},
                                 {id(p) for p in method.model.parameters() if p.requires_grad})
                self.assertEqual(sum(p.numel() for p in method.model.parameters()
                                     if p.requires_grad), 400 if name == "tent" else 208)
                self.assertTrue(all(not module.training for module in method.model.modules()))

                logits, metrics = method.step(self.images)
                torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                self.assertEqual(metrics["updates"], 1)
                changed = {n for n, p in method.model.named_parameters()
                           if not torch.equal(p, source[n])}
                self.assertTrue(changed)
                self.assertLessEqual(changed, expected)
                for norm in norms:
                    self.assertTrue(any(f".{norm}." in n for n in changed))
                self.assertTrue(any(n.startswith("norm.") for n in changed))
                self.assertTrue(method.optimizer.state)
                method.reset()
                self.assertFalse(method.optimizer.state)
                self.assertEqual(method.updates, 0)
                self.assert_model_equal(method.model.state_dict(), source)

    def test_checkpoint_continuation_and_reset(self):
        for name in ("tent_no_norm1", "tent_no_norm2"):
            with self.subTest(method=name):
                config = self.config(name)["method"]
                first = build_method(deepcopy(self.model), config)
                resumed = build_method(deepcopy(self.model), config)
                first.step(self.images)
                resumed.load_state_dict(deepcopy(first.state_dict()))
                expected, _ = first.step(self.images * 2)
                actual, _ = resumed.step(self.images * 2)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assert_model_equal(first.model.state_dict(), resumed.model.state_dict())
                self.assertEqual(resumed.updates, 2)
                resumed.reset()
                self.assertFalse(resumed.optimizer.state)
                self.assert_model_equal(resumed.model.state_dict(), self.model.state_dict())

    def test_resume_config_distinguishes_ablation_scopes(self):
        configs = [resume_config(self.config(name)) for name in
                   ("tent", "tent_no_norm1", "tent_no_norm2")]
        for i in range(len(configs)):
            for j in range(i):
                self.assertNotEqual(configs[i], configs[j])

    def test_ablation_rejects_all_parameter_adaptation(self):
        for name in ("tent_no_norm1", "tent_no_norm2"):
            with self.subTest(method=name):
                config = self.config(name)["method"]
                config["parameters"] = "all"
                with self.assertRaisesRegex(ValueError, "require method.parameters=norm"):
                    build_method(deepcopy(self.model), config)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
