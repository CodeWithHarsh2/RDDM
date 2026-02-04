import torch
import numpy as np

def make_schedules(T):
    alpha = torch.linspace(1.0, 0.0, T)
    beta2 = torch.linspace(0.0, 0.01, T)
    beta = torch.sqrt(beta2)

    alpha_bar = torch.cumsum(alpha, dim=0)
    beta_bar = torch.sqrt(torch.cumsum(beta2, dim=0))

    return alpha, beta, alpha_bar, beta_bar

def forward_diffusion(I0, Ires, t, alpha_bar, beta_bar):
    noise = torch.randn_like(I0)
    It = I0 + alpha_bar[t] * Ires + beta_bar[t] * noise
    return It, noise
