import torch
import torch.nn as nn
from einops import rearrange, einsum
from collections.abc import Callable, Iterable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"] # Get the beta coefficients.
            eps = group["eps"] # Get the epsilon value.
            weight_decay = group["weight_decay"] # Get the weight decay.
        for p in group["params"]:
            if p.grad is None:
                continue
            if "t" not in self.state[p]:
                self.state[p]["t"] = 0
                self.state[p]["m"] = torch.zeros_like(p.data)
                self.state[p]["v"] = torch.zeros_like(p.data)
            state = self.state[p]
            m, v = state["m"], state["v"] 
            grad = p.grad.data
            state["t"] += 1
            state["m"] = m.mul(beta1).add(grad, alpha=1 - beta1)
            state["v"] = v.mul(beta2).add(grad.pow(2), alpha=1 - beta2)
            alpha = lr * ( (1 - beta2 ** state["t"]) ** 0.5 ) / (1 - beta1 ** state["t"])
            p.data -= alpha * state["m"] / (state["v"].sqrt() + eps) + lr * weight_decay * p.data

        return loss

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    stabilized_inputs = inputs - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(stabilized_inputs), dim=-1))
    loss = - torch.gather(stabilized_inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) + log_sum_exp
    return loss.mean()

def cos_schedule(t:int, alphamax: float, alpha_min: float, T_w: int, T_c: int) -> torch.Tensor:
    if t < T_w:
        return alphamax * t / T_w
    elif t > T_c:
        return alpha_min
    else:
        cos_inner = (math.pi * (t - T_w)) / (T_c - T_w)
        return alpha_min + 0.5 * (alphamax - alpha_min) * (1 + math.cos(cos_inner))
    
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)