"""Check independent full-depth theta and first-six-block W0 adaptation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import LambdaW0, LambdaW0First6, build_method
from models.vittt import VisionTransformer
from utils import load_config


class LambdaW0First6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                                       depth=12, num_heads=2, num_classes=5).eval()
        with torch.no_grad():
            self.model.head.weight.mul_(50)
        self.images = torch.randn(4, 3, 12, 12)
        self.config = load_config([ROOT / "configs/lambda_w0_first6.yaml"])["method"]

    def assert_model_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_scopes_update_and_reset(self):
        expected = self.model(self.images).detach()
        source = deepcopy(self.model.state_dict())
        method = build_method(self.model, self.config)
        self.assertIs(type(method), LambdaW0First6)
        names = {f"blocks.{i}.attn.{key}" for i in range(12)
                 for key in ("theta_swiglu", "theta_conv")}
        names.update(f"blocks.{i}.attn.{key}" for i in range(6) for key in ("w1", "w2", "w3"))
        self.assertEqual(set(method.parameter_names), names)
        self.assertEqual({name for name, p in self.model.named_parameters() if p.requires_grad}, names)
        self.assertEqual({id(p) for p in self.model.parameters() if p.requires_grad},
                         {id(p) for group in method.optimizer.param_groups for p in group["params"]})
        self.assertEqual([group["lr"] for group in method.optimizer.param_groups], [0.3, 0.001])
        self.assertFalse(any(module.training for module in self.model.modules()))
        logits, metrics = method.step(self.images)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertAlmostEqual(metrics["loss"], metrics["entropy"] - metrics["marginal_entropy"], places=6)
        for name, parameter in self.model.named_parameters():
            if name in names:
                self.assertFalse(torch.equal(parameter, source.get(name, torch.zeros_like(parameter))), name)
            else:
                torch.testing.assert_close(parameter, source[name], rtol=0, atol=0)
        self.assertTrue(method.optimizer.state)
        method.reset()
        self.assertFalse(method.optimizer.state)
        self.assertEqual(method.updates, 0)
        self.assertTrue(all(p.count_nonzero() == 0 for name, p in self.model.named_parameters() if "theta_" in name))
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.images), expected, rtol=0, atol=0)

    def test_old_lambda_w0_still_updates_all_selected_blocks(self):
        method = LambdaW0(deepcopy(self.model), self.config)
        self.assertEqual(len(method.parameter_names), 12 * 5)
        self.assertTrue(all(getattr(block.attn, key).requires_grad
                            for block in method.model.blocks for key in ("w1", "w2", "w3")))

    def test_custom_theta_layers_and_branch_selection(self):
        self.config.update(layers=[0, 7, 11], branches={"swiglu": False, "conv": True})
        method = build_method(self.model, self.config)
        self.assertEqual(set(method.parameter_names), {
            "blocks.0.attn.theta_conv", "blocks.7.attn.theta_conv", "blocks.11.attn.theta_conv",
            "blocks.0.attn.w3"})

    def test_checkpoint_continuation_and_lambda_settings(self):
        first = build_method(deepcopy(self.model), self.config)
        second = build_method(deepcopy(self.model), deepcopy(self.config))
        first.step(self.images)
        state = deepcopy(first.state_dict())
        second.load_state_dict(state)
        logits, _ = first.step(self.images * 2)
        restored_logits, _ = second.step(self.images * 2)
        torch.testing.assert_close(logits, restored_logits, rtol=0, atol=0)
        self.assert_model_equal(first.model.state_dict(), second.model.state_dict())
        changed = deepcopy(self.config)
        changed["lambda_max"] = 0.5
        with self.assertRaisesRegex(ValueError, "Lambda settings differ"):
            build_method(deepcopy(self.model), changed).load_state_dict(state)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
