"""Sensitivity-Aware Differential Privacy (SADP) for ASFL-IoT.

Allocates DP perturbation noise adaptively based on layer hierarchy and sensitivity.
Addresses Section III-B (Eq. 5) of the IEEE ICES&T 2026 paper.
"""

from typing import Dict, Optional
import torch


def add_fixed_dp(
    params: Dict[str, torch.Tensor],
    noise_scale: float = 0.004,
) -> Dict[str, torch.Tensor]:
    """Apply uniform isotropic Gaussian noise across all model parameters (Baseline DP)."""
    return {k: v + torch.randn_like(v) * noise_scale for k, v in params.items()}


def add_sadp(
    params: Dict[str, torch.Tensor],
    base_noise: float = 0.008,
) -> Dict[str, torch.Tensor]:
    """Inject layer-sensitive differential privacy noise.

    Output head layers and deep weight matrices carry higher empirical inversion
    risk and receive calibrated noise budgets, while foundational feature layers
    retain lower perturbation to protect utility:
        - Head / Classifier layers: scale = base_noise * 0.60
        - Hidden weights: scale = base_noise * 0.80
        - Biases and shallow representations: scale = base_noise * 0.35
    """
    noisy: Dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if k.startswith("head."):
            scale = base_noise * 0.60
        elif "weight" in k:
            scale = base_noise * 0.80
        else:
            scale = base_noise * 0.35
        noisy[k] = v + torch.randn_like(v) * scale
    return noisy


def apply_dp_mode(
    params: Dict[str, torch.Tensor],
    mode: str = "none",
    fixed_noise: float = 0.004,
    sadp_base_noise: float = 0.008,
) -> Dict[str, torch.Tensor]:
    """Apply selected differential privacy scheme to parameter dictionary."""
    if mode == "fixed_dp":
        return add_fixed_dp(params, noise_scale=fixed_noise)
    elif mode == "sadp":
        return add_sadp(params, base_noise=sadp_base_noise)
    elif mode in ("none", "disabled"):
        return params
    else:
        raise ValueError(f"Unknown DP mode: '{mode}'. Supported: 'none', 'fixed_dp', 'sadp'.")
