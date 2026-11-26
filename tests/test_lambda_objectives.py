"""Check alternative outer objectives, theta-only updates and state handling."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import DeYO, LambdaDeYO, LambdaTent, Tent, build_method, build_optimizer
from models.lambda_ttt import LambdaTTT
from models.vittt import VisionTransformer
from utils import load_config


class LambdaObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = VisionTransformer(img_size=12, patch_size=4, embed_dim=8,
                                       depth=3, num_heads=2, num_classes=5).eval()
        # Make theta updates visible on the small randomly initialized fixture.
        with torch.no_grad():
            self.model.head.weight.mul_(50)
        self.images = torch.randn(4, 3, 12, 12)

    def config(self, name):
        return load_config([ROOT / f"configs/{name}.yaml"])["method"]

    def adapting_deyo_config(self):
        config = self.config("lambda_deyo")
        # Let the random fixture exercise the full update instead of filtering all.
        config["deyo"].update(deyo_margin=2.0, plpd_threshold=-2.0)
        return config

    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_factory_zero_theta_and_layer_branch_scopes(self):
        expected = self.model(self.images).detach()
        for name, cls in (("lambda_tent", LambdaTent), ("lambda_deyo", LambdaDeYO)):
            for layers, branches in (("all", {"swiglu": True, "conv": True}),
                                     ([0, 2], {"swiglu": False, "conv": True})):
                with self.subTest(name=name, layers=layers):
                    config = self.config(name)
                    config.update(layers=layers, branches=branches)
                    method = build_method(deepcopy(self.model), config)
                    self.assertIs(type(method), cls)
                    self.assertFalse(any(module.training for module in method.model.modules()))
                    indices = range(3) if layers == "all" else layers
                    names = {f"blocks.{i}.attn.theta_{branch}"
                             for i in indices for branch, enabled in branches.items() if enabled}
                    self.assertEqual(set(method.parameter_names), names)
                    self.assertEqual({n for n, p in method.model.named_parameters() if p.requires_grad}, names)
                    self.assertEqual({id(p) for p in method.model.parameters() if p.requires_grad},
                                     {id(p) for group in method.optimizer.param_groups for p in group["params"]})
                    with torch.no_grad():
                        torch.testing.assert_close(method.model(self.images), expected, rtol=0, atol=0)
                    for i, block in enumerate(method.model.blocks):
                        self.assertEqual(isinstance(block.attn, LambdaTTT), i in indices)

    def test_tent_reuses_entropy_and_updates_only_theta(self):
        self.assertIs(LambdaTent.loss, Tent.loss)
        self.assertIs(LambdaTent.step, Tent.step)
        config = self.config("lambda_tent")
        method = build_method(deepcopy(self.model), config)
        expected = self.model(self.images).detach()
        entropy = -(expected.softmax(1) * expected.log_softmax(1)).sum(1).mean()
        logits, metrics = method.step(self.images)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertAlmostEqual(metrics["loss"], entropy.item(), places=6)
        self.assertEqual(metrics["loss"], metrics["entropy"])
        self.assertNotIn("marginal_entropy", metrics)
        self.assertEqual(metrics["updates"], 1)
        self.assertTrue(any(p.count_nonzero() for p in method.model.parameters() if p.requires_grad))
        self.assert_state_equal(self.model.state_dict(),
                                {n: p for n, p in method.model.state_dict().items()
                                 if n not in method.parameter_names})

    def test_deyo_matches_existing_update_with_theta_parameters(self):
        self.assertIs(LambdaDeYO.step, DeYO.step)
        self.assertIs(LambdaDeYO.perturb, DeYO.perturb)
        for prediction in ("before_update", "after_update"):
            with self.subTest(prediction=prediction):
                config = self.adapting_deyo_config()
                config.update(steps=2, prediction=prediction)
                method = build_method(deepcopy(self.model), config)
                # Independently wire the existing DeYO to an identical Lambda model.
                reference = DeYO(deepcopy(method.model), config)
                reference.model.requires_grad_(False)
                parameters = dict(reference.model.named_parameters())
                for name in method.parameter_names:
                    parameters[name].requires_grad_(True)
                reference.optimizer = build_optimizer(
                    [parameters[name] for name in method.parameter_names], config["optimizer"])
                torch.manual_seed(71)
                logits, metrics = method.step(self.images)
                torch.manual_seed(71)
                reference_logits, reference_metrics = reference.step(self.images)
                torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
                self.assertEqual(metrics, reference_metrics)
                self.assertEqual(metrics["updates"], 2)
                self.assertEqual(metrics["n_selected"], len(self.images))
                self.assert_state_equal(method.model.state_dict(), reference.model.state_dict())
                self.assertTrue(any(p.count_nonzero() for p in method.model.parameters() if p.requires_grad))
                self.assert_state_equal(self.model.state_dict(),
                                        {n: p for n, p in method.model.state_dict().items()
                                         if n not in method.parameter_names})

    def test_deyo_empty_filters_preserve_theta_and_existing_momentum(self):
        for stage in ("entropy", "plpd"):
            with self.subTest(stage=stage):
                method = build_method(deepcopy(self.model), self.adapting_deyo_config())
                method.step(self.images)
                before = deepcopy(method.model.state_dict())
                momentum = [state["momentum_buffer"].clone() for state in method.optimizer.state.values()]
                self.assertTrue(any(value.count_nonzero() for value in momentum))
                if stage == "entropy":
                    method.entropy_threshold = 0
                else:
                    method.options["plpd_threshold"] = 2
                _, metrics = method.step(self.images)
                self.assertEqual(metrics["updates"], 0)
                self.assertEqual(method.updates, 1)
                self.assertIsNone(metrics["loss"])
                self.assert_state_equal(before, method.model.state_dict())
                for expected, state in zip(momentum, method.optimizer.state.values()):
                    torch.testing.assert_close(expected, state["momentum_buffer"], rtol=0, atol=0)

    def test_checkpoint_continuation_and_reset(self):
        for name in ("lambda_tent", "lambda_deyo"):
            with self.subTest(name=name):
                config = self.config(name) if name == "lambda_tent" else self.adapting_deyo_config()
                first = build_method(deepcopy(self.model), config)
                second = build_method(deepcopy(self.model), deepcopy(config))
                first.step(self.images)
                second.load_state_dict(deepcopy(first.state_dict()))
                torch.manual_seed(72)
                logits, metrics = first.step(self.images * 2)
                torch.manual_seed(72)
                restored_logits, restored_metrics = second.step(self.images * 2)
                torch.testing.assert_close(logits, restored_logits, rtol=0, atol=0)
                self.assertEqual(metrics, restored_metrics)
                self.assert_state_equal(first.model.state_dict(), second.model.state_dict())
                second.reset()
                self.assertFalse(second.optimizer.state)
                self.assertEqual(second.updates, 0)
                self.assertTrue(all(p.count_nonzero() == 0 for p in second.model.parameters() if p.requires_grad))
                with torch.no_grad():
                    torch.testing.assert_close(second.model(self.images), self.model(self.images), rtol=0, atol=0)

    def test_checkpoint_rejects_changed_lambda_settings(self):
        for name in ("lambda_tent", "lambda_deyo"):
            config = self.config(name)
            state = deepcopy(build_method(deepcopy(self.model), config).state_dict())
            for key, value in (("lambda_max", 0.5), ("layers", [0]),
                               ("branches", {"swiglu": True, "conv": False})):
                with self.subTest(name=name, key=key):
                    changed = deepcopy(config)
                    changed[key] = value
                    method = build_method(deepcopy(self.model), changed)
                    with self.assertRaisesRegex(ValueError, "Lambda settings differ"):
                        method.load_state_dict(deepcopy(state))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
