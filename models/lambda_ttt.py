"""Channel-wise shaping of the TTT inner loss; original forward stays intact."""
import torch
from torch import nn
from torch.nn import functional as F

from .ttt_block import TTT


class LambdaTTT(TTT):
    def __init__(self, original, lambda_max=1.0, swiglu=True, conv=True):
        # Reuse the loaded modules and weights without initializing a new model.
        nn.Module.__init__(self)
        self.dim = original.dim
        self.num_heads = original.num_heads
        self.scale = original.scale
        self.qkv, self.proj = original.qkv, original.proj
        self.w1, self.w2, self.w3 = original.w1, original.w2, original.w3
        self.lambda_max = lambda_max

        head_dim = self.dim // self.num_heads
        self.register_parameter("theta_swiglu", nn.Parameter(self.w1.new_zeros(
            1, self.num_heads, 1, head_dim)) if swiglu else None)
        self.register_parameter("theta_conv", nn.Parameter(self.w3.new_zeros(
            1, head_dim, 1, 1)) if conv else None)

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        if self.theta_swiglu is None:
            return super().inner_train_simplified_swiglu(k, v, w1, w2, lr)

        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        v_hat = z1 * a
        lam = self.lambda_max * self.theta_swiglu.tanh()

        # d[-v_hat*v + 0.5*lambda*v_hat^2] / d(v_hat).
        e = (lam * v_hat - v) / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - lr * g1, w2 - lr * g2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation="prod"):
        if self.theta_conv is None:
            return super().inner_train_3x3dwc(k, v, w, lr, implementation)

        B, C, H, W = k.shape
        v_hat = F.conv2d(k, w, padding=1, groups=C)
        lam = self.lambda_max * self.theta_conv.tanh()
        e = (lam * v_hat - v) / float(H * W) * self.scale

        if implementation == "conv":
            g = F.conv2d(k.reshape(1, B * C, H, W),
                         e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == "prod":
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys, xs = 1 + dy, 1 + dx
                    outs.append((k[:, :, ys:ys + H, xs:xs + W] * e).sum(dim=(-2, -1)))
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError(implementation)

        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        return w.repeat(B, 1, 1, 1) - lr * g
