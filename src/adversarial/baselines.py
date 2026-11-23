"""
Gradient-based baselines to compare AdvGAN against.

  FGSM — Goodfellow et al., 2015 (arXiv:1412.6572): one signed-gradient step of size eps.
  PGD  — Madry et al., 2018 (arXiv:1706.06083): many small steps, projected back into the eps-ball.

Both need white-box gradient access to the ViT for every image they attack.
Inputs/outputs are [0,1] pixel tensors; eps is in [0,1] units (e.g. 8/255).
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _step_loss(logits, y, target: Optional[int]):
    # untargeted: increase loss on the true label; targeted: decrease loss on the target label
    if target is None:
        return F.cross_entropy(logits, y)
    return -F.cross_entropy(logits, torch.full_like(y, target))


def fgsm(model, x: torch.Tensor, y: torch.Tensor, eps: float, target: Optional[int] = None) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    loss = _step_loss(model(x), y, target)
    grad, = torch.autograd.grad(loss, x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 10,
        alpha: Optional[float] = None, random_start: bool = True,
        target: Optional[int] = None) -> torch.Tensor:
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    x0 = x.clone().detach()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps) if random_start else x0.clone()
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = _step_loss(model(x_adv), y, target)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = (x0 + (x_adv - x0).clamp(-eps, eps)).clamp(0, 1)
    return x_adv.detach()
