"""Small TTA implementations. Methods receive images, never target labels.

Tent here means entropy minimization on ViTTT's LayerNorm affine parameters.
InfoMax is a batch information-maximization baseline, not full SHOT.
Lambda shapes the inner loss while keeping the original weights frozen.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.lambda_ttt import LambdaTTT
from sam import SAM


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


class LambdaW0(Lambda):
    """InfoMax on theta and the selected branches' fast-weight initializers."""
    def __init__(self, model, config):
        super().__init__(model, config)
        parameters = []
        for name, module in model.named_modules():
            if not isinstance(module, LambdaTTT):
                continue
            weights = []
            if module.theta_swiglu is not None:
                weights.extend([("w1", module.w1), ("w2", module.w2)])
            if module.theta_conv is not None:
                weights.append(("w3", module.w3))
            for key, parameter in weights:
                parameter.requires_grad_(True)
                parameters.append(parameter)
                self.parameter_names.append(f"{name}.{key}")

        self.optimizer.add_param_group({"params": parameters, "lr": config["w0_lr"]})
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())


def softmax_entropy(logits):
    # Same expression as the official EATA / SAR / DeYO implementations.
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def norm_parameters(model, exclude_blocks=(), exclude_final_norm=False):
    parameters, names = [], []
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        if exclude_final_norm and name == "norm":
            continue
        if name.startswith("blocks.") and int(name.split(".")[1]) in exclude_blocks:
            continue
        for key, parameter in module.named_parameters(recurse=False):
            if key in ("weight", "bias"):
                parameter.requires_grad_(True)
                parameters.append(parameter)
                names.append(f"{name}.{key}")
    return parameters, names


def prediction_logits(method, images, logits):
    if method.config["prediction"] == "after_update":
        with torch.no_grad():
            return method.model(images).detach()
    return logits.detach()


class EATA(Source):
    """Official EATA loss and the authors' SAR-repository LayerNorm support."""
    def __init__(self, model, config):
        super().__init__(model, config)
        parameters, self.parameter_names = norm_parameters(model)
        self.optimizer = build_optimizer(parameters, config["optimizer"])
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())
        self.options = config["eata"]
        self.e_margin = self.options["e_margin"] * math.log(model.num_classes)
        self.current_model_probs = None
        self.fishers = None

    @torch.enable_grad()
    def estimate_fisher(self, loader):
        parameters = dict(self.model.named_parameters())
        self.fishers = {name: [torch.zeros_like(parameters[name]), parameters[name].detach().clone()]
                        for name in self.parameter_names}
        device = next(self.model.parameters()).device
        for images, _ in loader:
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images.to(device))
            F.cross_entropy(logits, logits.argmax(1)).backward()
            for name, (fisher, _) in self.fishers.items():
                fisher.add_(parameters[name].grad.detach().square())
        for fisher, _ in self.fishers.values():
            fisher.div_(len(loader))
        self.optimizer.zero_grad(set_to_none=True)

    @torch.enable_grad()
    def step(self, images):
        if self.fishers is None:
            raise RuntimeError("EATA requires estimate_fisher(clean_loader) before adaptation")
        updates = 0
        for _ in range(self.config["steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            entropies = softmax_entropy(logits)
            selected = torch.where(entropies < self.e_margin)[0]
            reliable = len(selected)
            if self.current_model_probs is not None:
                similarity = F.cosine_similarity(self.current_model_probs.unsqueeze(0),
                                                 logits[selected].softmax(1), dim=1)
                selected = selected[similarity.abs() < self.options["d_margin"]]

            metrics = {"n_reliable": reliable, "n_selected": len(selected),
                       "loss": None, "weighted_entropy": None, "fisher_penalty": None}
            if len(selected) == 0:
                continue
            with torch.no_grad():
                mean_probs = logits[selected].softmax(1).mean(0)
                if self.current_model_probs is None:
                    self.current_model_probs = mean_probs
                else:
                    self.current_model_probs = 0.9 * self.current_model_probs + (1 - 0.9) * mean_probs

            entropies = entropies[selected]
            weight = 1 / torch.exp(entropies.clone().detach() - self.e_margin)
            weighted_entropy = entropies.mul(weight).mean()
            penalty = logits.new_zeros(())
            for name, parameter in self.model.named_parameters():
                if name in self.fishers:
                    fisher, reference = self.fishers[name]
                    penalty = penalty + self.options["fisher_alpha"] * (
                        fisher * (parameter - reference).square()).sum()
            loss = weighted_entropy + penalty
            loss.backward()
            self.optimizer.step()
            updates += 1
            self.updates += 1
            metrics.update(loss=float(loss.detach()), weighted_entropy=float(weighted_entropy.detach()),
                           fisher_penalty=float(penalty.detach()))

        self.optimizer.zero_grad(set_to_none=True)
        metrics["updates"] = updates
        metrics.update(forward_calls=self.config["steps"] + (self.config["prediction"] == "after_update"),
                       backward_calls=updates)
        return prediction_logits(self, images, logits), metrics

    def reset(self):
        super().reset()
        self.current_model_probs = None

    def state_dict(self):
        raise NotImplementedError("EATA checkpoint support is deferred; use run.save_weights=none")


class SAR(Source):
    """Official reliable two-pass SAM update, with complete cross-domain resets."""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.options = config["sar"]
        parameters, self.parameter_names = norm_parameters(
            model, self.options["exclude_blocks"], self.options["exclude_final_norm"])
        optimizer = config["optimizer"]
        if optimizer["name"] != "sgd":
            raise ValueError("SAR uses SAM with a base SGD optimizer")
        self.optimizer = SAM(parameters, torch.optim.SGD, rho=self.options["rho"],
                             adaptive=self.options["adaptive"], lr=optimizer["lr"],
                             momentum=optimizer["momentum"], weight_decay=optimizer["weight_decay"])
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())
        self.margin = self.options["margin_e0"] * math.log(model.num_classes)
        self.ema = None

    @torch.enable_grad()
    def step(self, images):
        recoveries = 0
        for _ in range(self.config["steps"]):
            self.optimizer.zero_grad()
            logits = self.model(images)
            entropies = softmax_entropy(logits)
            first_ids = torch.where(entropies < self.margin)
            loss_first = entropies[first_ids].mean()
            loss_first.backward()
            self.optimizer.first_step(zero_grad=True)

            second_entropies = softmax_entropy(self.model(images))[first_ids]
            second_ids = torch.where(second_entropies < self.margin)
            loss_second = second_entropies[second_ids].mean()
            value = float(loss_second.detach())
            if not math.isnan(value):
                self.ema = value if self.ema is None else 0.9 * self.ema + (1 - 0.9) * value
            loss_second.backward()
            self.optimizer.second_step(zero_grad=True)
            self.updates += 1

            # Official internal recovery keeps base-SGD momentum and the new EMA.
            if self.ema is not None and self.ema < self.options["reset_constant_em"]:
                self.model.load_state_dict(self.initial_model, strict=True)
                self.optimizer.load_state_dict(deepcopy(self.initial_optimizer))
                recoveries += 1

        first_value = float(loss_first.detach())
        metrics = {"loss": value if math.isfinite(value) else None,
                   "entropy_first": first_value if math.isfinite(first_value) else None,
                   "n_reliable": len(first_ids[0]), "n_selected": len(second_ids[0]),
                   "entropy_ema": self.ema, "recoveries": recoveries, "updates": self.config["steps"]}
        metrics.update(forward_calls=2 * self.config["steps"] + (self.config["prediction"] == "after_update"),
                       backward_calls=2 * self.config["steps"])
        return prediction_logits(self, images, logits), metrics

    def reset(self):
        super().reset()
        # The official SAM state omits base-optimizer momentum. Clear it at task boundaries.
        self.optimizer.base_optimizer.state.clear()
        self.ema = None

    def state_dict(self):
        raise NotImplementedError("SAR checkpoint support is deferred; use run.save_weights=none")


class DeYO(Source):
    """Official entropy / PLPD filtering and additive loss reweighting."""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.options = config["deyo"]
        parameters, self.parameter_names = norm_parameters(
            model, self.options["exclude_blocks"], self.options["exclude_final_norm"])
        self.optimizer = build_optimizer(parameters, config["optimizer"])
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())
        self.entropy_threshold = self.options["deyo_margin"] * math.log(model.num_classes)
        self.entropy_reference = self.options["deyo_margin_e0"] * math.log(model.num_classes)

    def perturb(self, images):
        from einops import rearrange
        from torchvision.transforms import Resize

        images = images.detach().clone()
        if self.options["aug_type"] == "patch":
            patches = self.options["patch_len"]
            side = images.shape[-1]
            images = Resize(((side // patches) * patches, (side // patches) * patches))(images)
            images = rearrange(images, "b c (p h) (q w) -> b (p q) c h w", p=patches, q=patches)
            permutation = torch.argsort(torch.rand(images.shape[0], images.shape[1]), dim=-1)
            images = images[torch.arange(images.shape[0]).unsqueeze(-1), permutation]
            images = rearrange(images, "b (p q) c h w -> b c (p h) (q w)", p=patches, q=patches)
            return Resize((side, side))(images)
        if self.options["aug_type"] == "pixel":
            side = images.shape[-1]
            images = rearrange(images, "b c h w -> b c (h w)")
            images = images[:, :, torch.randperm(images.shape[-1])]
            return rearrange(images, "b c (h w) -> b c h w", h=side, w=side)
        if self.options["aug_type"] == "occ":
            size = self.options["occlusion_size"]
            row, column = self.options["row_start"], self.options["column_start"]
            mean = images.flatten(2).mean(2).unsqueeze(-1).unsqueeze(-1)
            images[:, :, row:row + size, column:column + size] = mean.expand(-1, -1, size, size)
            return images
        raise ValueError("deyo.aug_type must be patch, pixel or occ")

    @torch.enable_grad()
    def step(self, images):
        updates = 0
        perturbed_calls = 0
        for _ in range(self.config["steps"]):
            logits = self.model(images)
            self.optimizer.zero_grad()
            entropies = softmax_entropy(logits)
            if self.options["filter_ent"]:
                first_ids = torch.where(entropies < self.entropy_threshold)[0]
            else:
                first_ids = torch.where(entropies <= math.log(self.model.num_classes))[0]
            metrics = {"n_reliable": len(first_ids), "n_selected": 0, "loss": None, "plpd_mean": None}
            if len(first_ids) == 0:
                continue
            with torch.no_grad():
                perturbed_logits = self.model(self.perturb(images[first_ids]))
            perturbed_calls += 1
            probabilities = logits[first_ids].softmax(1)
            classes = probabilities.argmax(1, keepdim=True)
            plpd = (probabilities.gather(1, classes) - perturbed_logits.softmax(1).gather(1, classes)).flatten()
            if self.options["filter_plpd"]:
                selected = torch.where(plpd > self.options["plpd_threshold"])[0]
            else:
                selected = torch.where(plpd >= -2.0)[0]
            metrics.update(n_selected=len(selected), plpd_mean=float(plpd.detach().mean()))
            if len(selected) == 0:
                continue

            entropies = entropies[first_ids][selected]
            if self.options["reweight_ent"] or self.options["reweight_plpd"]:
                weight = (self.options["reweight_ent"] / torch.exp(entropies.clone().detach() - self.entropy_reference)
                          + self.options["reweight_plpd"] / torch.exp(-plpd[selected].clone().detach()))
                entropies = entropies * weight
            loss = entropies.mean()
            loss.backward()
            self.optimizer.step()
            updates += 1
            self.updates += 1
            metrics["loss"] = float(loss.detach())

        self.optimizer.zero_grad()
        metrics["updates"] = updates
        metrics.update(forward_calls=self.config["steps"] + perturbed_calls
                       + (self.config["prediction"] == "after_update"), backward_calls=updates)
        return prediction_logits(self, images, logits), metrics


class _LambdaObjective(Lambda):
    """Keep new variants' Lambda settings in their own checkpoint state."""
    def state_dict(self):
        state = super().state_dict()
        state["lambda_options"] = {key: deepcopy(self.config[key])
                                   for key in ("lambda_max", "branches", "layers")}
        return state

    def load_state_dict(self, state):
        # Legacy resume_config only compares Lambda settings for the old names.
        options = {key: self.config[key] for key in ("lambda_max", "branches", "layers")}
        if state["lambda_options"] != options:
            raise ValueError("Resume Lambda settings differ from the saved experiment")
        super().load_state_dict(state)


class LambdaTent(_LambdaObjective):
    """Tent's entropy objective, updating only Lambda theta."""
    loss = Tent.loss


class LambdaDeYO(_LambdaObjective):
    """DeYO's unchanged filtering and update, optimizing only Lambda theta."""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.options = config["deyo"]
        self.entropy_threshold = self.options["deyo_margin"] * math.log(model.num_classes)
        self.entropy_reference = self.options["deyo_margin_e0"] * math.log(model.num_classes)

    perturb = DeYO.perturb
    step = DeYO.step


METHODS = {"source": Source, "tent": Tent, "infomax": InfoMax, "lambda": Lambda,
           "lambda_w0": LambdaW0, "eata": EATA, "sar": SAR, "deyo": DeYO}
METHODS.update(lambda_tent=LambdaTent, lambda_deyo=LambdaDeYO)


def build_method(model, config):
    return METHODS[config["name"]](model, config)
