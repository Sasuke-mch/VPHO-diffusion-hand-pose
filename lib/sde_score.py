"""
Optional score-based SDE / ODE utilities.

This file lets you:
1. train a real continuous-time score model later;
2. approximately reuse your current DDPM eps-prediction model as a score model.

Important: DDPMEpsToScoreAdapter is only a diagnostic/migration baseline.
For a true VPHO-style score-SDE model, retrain the diffusion branches with
score_matching_loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VPSDE:
    """Variance Preserving SDE."""

    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)[:, None] * x0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)).clamp_min(1e-8)
        return mean, std

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t).clamp_min(1e-8)
        return drift, diffusion

    def prior_sampling(self, shape, device):
        return torch.randn(*shape, device=device)


def score_matching_loss(x0, score_model, sde: VPSDE, eps=1e-5):
    """
    Denoising score matching objective.

    score_model(xt, t_float) should return score with the same shape as x0.
    """
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device) * (1.0 - eps) + eps
    mean, std = sde.marginal_prob(x0, t)
    z = torch.randn_like(x0)
    xt = mean + std[:, None] * z
    score = score_model(xt, t)
    target = -z / std[:, None]
    return ((score - target).pow(2) * std[:, None].pow(2)).sum(dim=-1).mean()


@torch.no_grad()
def probability_flow_ode_sampler(
    score_model,
    sde: VPSDE,
    shape,
    device,
    steps=80,
    T=1.0,
    eps=1e-3,
    x_init=None,
    solver="heun",
):
    """Simple Euler/Heun probability-flow ODE sampler."""
    x = sde.prior_sampling(shape, device) if x_init is None else x_init.clone()
    ts = torch.linspace(T, eps, steps, device=device)

    def ode_drift(x_cur, t_cur):
        drift, diffusion = sde.sde(x_cur, t_cur)
        score = score_model(x_cur, t_cur)
        return drift - 0.5 * diffusion[:, None].pow(2) * score

    for i in range(len(ts) - 1):
        t = torch.full((shape[0],), ts[i], device=device)
        dt = ts[i + 1] - ts[i]
        if solver == "euler":
            x = x + ode_drift(x, t) * dt
        elif solver == "heun":
            k1 = ode_drift(x, t)
            x_euler = x + k1 * dt
            t_next = torch.full((shape[0],), ts[i + 1], device=device)
            k2 = ode_drift(x_euler, t_next)
            x = x + 0.5 * (k1 + k2) * dt
        else:
            raise ValueError(f"unknown solver: {solver}")
    return x


class DDPMEpsToScoreAdapter(nn.Module):
    """
    Approximate adapter: wrap a DDPM eps-prediction model as a continuous score model.

    forward_fn(model, x_t, t_long) must call the model with your branch-specific conditions.
    """

    def __init__(self, eps_model: nn.Module, T: int, alpha_bar: torch.Tensor, interpolate=True):
        super().__init__()
        self.eps_model = eps_model
        self.T = int(T)
        self.interpolate = bool(interpolate)
        self.register_buffer("alpha_bar_buf", alpha_bar.float())

    def forward(self, x_t: torch.Tensor, t_cont: torch.Tensor, forward_fn):
        tau = t_cont.clamp(0.0, 1.0) * (self.T - 1)
        lo = tau.floor().long().clamp(0, self.T - 1)
        hi = tau.ceil().long().clamp(0, self.T - 1)
        w = (tau - lo.float())[:, None]

        eps_lo = forward_fn(self.eps_model, x_t, lo + 1)
        if self.interpolate:
            eps_hi = forward_fn(self.eps_model, x_t, hi + 1)
            eps = (1.0 - w) * eps_lo + w * eps_hi
            abar = (1.0 - w[:, 0]) * self.alpha_bar_buf[lo] + w[:, 0] * self.alpha_bar_buf[hi]
        else:
            eps = eps_lo
            abar = self.alpha_bar_buf[lo]

        std = torch.sqrt(1.0 - abar).clamp_min(1e-8)[:, None]
        return -eps / std
