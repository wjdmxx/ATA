"""Check the derived inner gradients and the theta-only adaptation protocol."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import Lambda, build_method
from models import build_model
from models.lambda_ttt import LambdaTTT
from models.ttt_block import TTT
from models.vittt import VisionTransformer
from utils import load_config, resume_config


def small_model():
    return VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                             depth=3, num_heads=2, num_classes=5)


class LambdaGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.block = LambdaTTT(TTT(8, 2).double(), lambda_max=0.7)
        with torch.no_grad():
            self.block.theta_swiglu.uniform_(-0.6, 0.6)
            self.block.theta_conv.uniform_(-0.6, 0.6)

    def test_swiglu_closed_form_and_theta_gradient_match_explicit_loss(self):
        block = self.block
        k, v = torch.randn(2, 2, 9, 4, dtype=torch.double), torch.randn(2, 2, 9, 4, dtype=torch.double)
        w1 = block.w1.detach().repeat(2, 1, 1, 1).requires_grad_()
        w2 = block.w2.detach().repeat(2, 1, 1, 1).requires_grad_()
        prediction = (k @ w1) * F.silu(k @ w2)
        lam = block.lambda_max * block.theta_swiglu.tanh()
        loss = (-prediction * v + 0.5 * lam * prediction.square()).sum() / 9 * block.scale
        gradients = torch.autograd.grad(loss, (w1, w2), create_graph=True)
        expected = [w - 0.4 * g / (g.norm(dim=-2, keepdim=True) + 1)
                    for w, g in zip((w1, w2), gradients)]
        actual = block.inner_train_simplified_swiglu(k, v, block.w1, block.w2, lr=0.4)

        for first, second in zip(actual, expected):
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-14)
        actual_grad = torch.autograd.grad(sum(w.square().sum() for w in actual), block.theta_swiglu)[0]
        expected_grad = torch.autograd.grad(sum(w.square().sum() for w in expected), block.theta_swiglu)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-14)
        self.assertGreater(actual_grad.abs().max().item(), 0)

    def test_convolution_closed_form_and_theta_gradient_match_explicit_loss(self):
        block = self.block
        k, v = torch.randn(2, 4, 3, 3, dtype=torch.double), torch.randn(2, 4, 3, 3, dtype=torch.double)
        weights = block.w3.detach().repeat(2, 1, 1, 1).requires_grad_()
        prediction = F.conv2d(k.reshape(1, 8, 3, 3), weights, padding=1, groups=8).reshape_as(v)
        lam = block.lambda_max * block.theta_conv.tanh()
        loss = (-prediction * v + 0.5 * lam * prediction.square()).sum() / 9 * block.scale
        gradient = torch.autograd.grad(loss, weights, create_graph=True)[0]
        expected = weights - 0.4 * gradient / (gradient.norm(dim=(-2, -1), keepdim=True) + 1)
        expected_grad = torch.autograd.grad(expected.square().sum(), block.theta_conv)[0]

        for implementation in ("prod", "conv"):
            actual = block.inner_train_3x3dwc(k, v, block.w3, lr=0.4, implementation=implementation)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
            actual_grad = torch.autograd.grad(actual.square().sum(), block.theta_conv)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-14)
            self.assertGreater(actual_grad.abs().max().item(), 0)


class LambdaMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.config = load_config([ROOT / "configs/lambda.yaml"])["method"]
        self.model = small_model().eval()
        self.images = torch.randn(4, 3, 12, 12)

    def test_zero_theta_exactly_matches_source_and_branch_switches(self):
        with torch.no_grad():
            expected = self.model(self.images)
        for swiglu, conv in ((True, True), (True, False), (False, True)):
            config = deepcopy(self.config)
            config["branches"] = {"swiglu": swiglu, "conv": conv}
            method = build_method(deepcopy(self.model), config)
            with torch.no_grad():
                torch.testing.assert_close(method.model(self.images), expected, rtol=0, atol=0)
            self.assertEqual(len(method.parameter_names), 3 * (swiglu + conv))
            for name in method.parameter_names:
                self.assertEqual(torch.count_nonzero(dict(method.model.named_parameters())[name]).item(), 0)

    def test_layer_selection_theta_only_update_and_reset(self):
        self.config["layers"] = [0, 2]
        before = deepcopy(self.model.state_dict())
        expected = self.model(self.images).detach()
        method = Lambda(self.model, self.config)
        self.assertEqual(method.parameter_names, [
            "blocks.0.attn.theta_swiglu", "blocks.0.attn.theta_conv",
            "blocks.2.attn.theta_swiglu", "blocks.2.attn.theta_conv"])
        self.assertIs(type(self.model.blocks[1].attn), TTT)
        selected = {id(p) for p in self.model.parameters() if p.requires_grad}
        optimized = {id(p) for group in method.optimizer.param_groups for p in group["params"]}
        self.assertEqual(selected, optimized)

        logits, metrics = method.step(self.images)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertAlmostEqual(metrics["loss"], metrics["entropy"] - metrics["marginal_entropy"], places=6)
        self.assertTrue(all(torch.count_nonzero(dict(self.model.named_parameters())[name]) for name in method.parameter_names))
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(dict(self.model.named_parameters())[name].requires_grad)

        self.assertTrue(method.optimizer.state)
        method.reset()
        self.assertFalse(method.optimizer.state)
        self.assertEqual(method.updates, 0)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.images), expected, rtol=0, atol=0)

    def test_checkpoint_continuation_and_reset(self):
        first = Lambda(deepcopy(self.model), self.config)
        second = Lambda(deepcopy(self.model), self.config)
        first.step(self.images)
        second.load_state_dict(deepcopy(first.state_dict()))
        first_logits, _ = first.step(self.images * 2)
        second_logits, _ = second.step(self.images * 2)
        torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
        second.reset()
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in second.model.parameters() if p.requires_grad))

    def test_small_model_uses_all_actual_layers_and_5376_parameters(self):
        model = build_model("small")
        method = Lambda(model, self.config)
        self.assertEqual(len(method.parameter_names), len(model.blocks) * 2)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 5376)
        self.assertTrue(all(name.endswith(("theta_swiglu", "theta_conv")) for name in method.parameter_names))

    def test_invalid_layer_branch_and_bound_settings(self):
        for layers in ([], [3], [-1], [0, 0], [True], "last"):
            config = deepcopy(self.config)
            config["layers"] = layers
            with self.assertRaises(ValueError):
                Lambda(deepcopy(self.model), config)
        for maximum in (0, -1, float("inf"), float("nan")):
            config = deepcopy(self.config)
            config["lambda_max"] = maximum
            with self.assertRaises(ValueError):
                Lambda(deepcopy(self.model), config)
        self.config["branches"] = {"swiglu": False, "conv": False}
        with self.assertRaises(ValueError):
            Lambda(self.model, self.config)

    def test_new_lambda_options_do_not_break_old_baseline_resume(self):
        config = load_config()
        old = deepcopy(config)
        for key in ("lambda_max", "branches", "layers"):
            old["method"].pop(key)
        self.assertEqual(resume_config(config), resume_config(old))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
