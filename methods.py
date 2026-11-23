"""Small TTA implementations. Methods receive images, never target labels.

Tent here means entropy minimization on ViTTT's LayerNorm affine parameters.
InfoMax is a batch information-maximization baseline, not full SHOT.
Lambda shapes the inner loss while keeping the original weights frozen.
"""
from copy import deepcopy

import torch
from torch import nn

from models.lambda_ttt import LambdaTTT


def build_optimizer(parameters, config):
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    options = {"lr": config["lr"], "weight_decay": config["weight_decay"]}

    if config["name"] == "sgd":
        options["momentum"] = config["momentum"]
    else:
        options["betas"] = tuple(config["betas"])

    return optimizers[config["name"]](parameters, **options)


class Source:
    def __init__(self, model, config):
        self.model = model.eval().requires_grad_(False)
        self.config = config
        self.optimizer = None
        self.parameter_names = []

        self.initial_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.initial_optimizer = None
        self.updates = 0

    @torch.no_grad()
    def step(self, images):
        return self.model(images), {"updates": 0}

    def reset(self):
        self.model.load_state_dict(self.initial_model, strict=True)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(deepcopy(self.initial_optimizer))
            self.optimizer.zero_grad(set_to_none=True)

        self.updates = 0

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"], strict=True)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])

        self.updates = state["updates"]


class Tent(Source):
    def __init__(self, model, config):
        super().__init__(model, config)

        # Keep dropout/stochastic depth off. LayerNorm behaves the same in eval.
        if config["parameters"] == "norm":
            for module in model.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        elif config["parameters"] == "all":
            model.requires_grad_(True)
        else:
            raise ValueError("method.parameters must be norm or all")

        self.parameter_names = [name for name, p in model.named_parameters() if p.requires_grad]
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No parameters selected for adaptation")

        self.optimizer = build_optimizer(parameters, config["optimizer"])
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())

    def loss(self, logits):
        log_probs = logits.log_softmax(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        return entropy, {"entropy": float(entropy.detach())}

    @torch.enable_grad()
    def step(self, images):
        for _ in range(self.config["steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss, metrics = self.loss(logits)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite adaptation loss")

            loss.backward()
            self.optimizer.step()
            self.updates += 1

        self.optimizer.zero_grad(set_to_none=True)

        if self.config["prediction"] == "after_update":
            with torch.no_grad():
                logits = self.model(images)

        metrics.update(loss=float(loss.detach()), updates=self.config["steps"])
        # before_update returns the final adaptation forward's logits, as Tent
        # does. With multiple steps these precede only the LAST optimizer step.
        return logits.detach(), metrics


class InfoMax(Tent):
    def loss(self, logits):
        conditional, metrics = super().loss(logits)

        marginal = logits.softmax(dim=-1).mean(dim=0)
        marginal_entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum()
        loss = conditional - self.config["diversity_weight"] * marginal_entropy

        metrics["marginal_entropy"] = float(marginal_entropy.detach())
        return loss, metrics


class Lambda(InfoMax):
    def __init__(self, model, config):
        layers = config["layers"]
        if layers == "all":
            layers = list(range(len(model.blocks)))
        if (not isinstance(layers, list) or not layers
                or any(type(i) is not int or not 0 <= i < len(model.blocks) for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("method.layers must be 'all' or distinct zero-based block indices")

        branches = config["branches"]
        if not all(type(value) is bool for value in branches.values()) or not any(branches.values()):
            raise ValueError("Enable at least one Lambda branch with true/false")
        if not 0 < config["lambda_max"] < float("inf"):
            raise ValueError("method.lambda_max must be finite and positive")

        for i in layers:
            model.blocks[i].attn = LambdaTTT(
                model.blocks[i].attn, config["lambda_max"], **branches)

        # Snapshot the model including zero theta, then enable only theta.
        Source.__init__(self, model, config)
        parameters = []
        for name, parameter in model.named_parameters():
            if name.rsplit(".", 1)[-1] in ("theta_swiglu", "theta_conv"):
                parameter.requires_grad_(True)
                self.parameter_names.append(name)
                parameters.append(parameter)

        self.optimizer = build_optimizer(parameters, config["optimizer"])
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())


METHODS = {"source": Source, "tent": Tent, "infomax": InfoMax, "lambda": Lambda}


def build_method(model, config):
    return METHODS[config["name"]](model, config)
