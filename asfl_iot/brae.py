"""Byzantine-Robust Aggregation and Evaluation (BRAE) for ASFL-IoT.

Detects and filters malicious model updates via Median Absolute Deviation (MAD)
distance gating and cosine alignment screening.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from .models import flatten_params


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D parameter vectors."""
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    denom = (norm_a * norm_b).item() + 1e-12
    return float(torch.dot(a, b).item() / denom)


def brae_filter(
    global_params: Dict[str, torch.Tensor],
    client_params: List[Dict[str, torch.Tensor]],
    mad_beta: float = 3.0,
    cosine_min: float = -0.3,
) -> Tuple[List[int], Dict[str, Any]]:
    """Filter candidate client updates through dual MAD-norm and cosine direction tests.

    Parameters:
        global_params: Reference parameter dictionary from previous round t-1.
        client_params: List of uploaded parameter dictionaries from admitted clients.
        mad_beta: Multiplier for Median Absolute Deviation cut-off threshold.
        cosine_min: Minimum allowable cosine similarity against global gradient reference.

    Returns:
        keep_indices: Indices of client updates surviving the Byzantine filter.
        info: Dictionary reporting count of rejected/detected anomalous updates.
    """
    if len(client_params) <= 2:
        return list(range(len(client_params))), {"detected": 0}

    vecs = [flatten_params(p) for p in client_params]
    stack = torch.stack(vecs)

    # 1. Coordinate-wise median vector
    median_vec = stack.median(dim=0).values

    # 2. Euclidean distances to coordinate-wise median
    distances = torch.norm(stack - median_vec, dim=1).numpy()
    med = float(np.median(distances))
    mad = float(np.median(np.abs(distances - med))) + 1e-12
    keep_mad = distances <= (med + mad_beta * mad)

    # 3. Directional cosine alignment check against reference model
    global_vec = flatten_params(global_params)
    cosines = np.array([cosine_similarity(v, global_vec) for v in vecs])
    keep_cos = cosines >= cosine_min

    # Dual criterion survival
    keep = np.where(keep_mad & keep_cos)[0].tolist()
    detected = len(client_params) - len(keep)
    return keep, {"detected": detected, "distances": distances, "cosines": cosines}


def make_byzantine(
    params: Dict[str, torch.Tensor],
    attack: str = "sign_flip",
) -> Dict[str, torch.Tensor]:
    """Inject Byzantine poisoning into client model parameters.

    Supported attack modalities:
        - 'sign_flip': Inverts update direction with -3.0x magnification.
        - 'gaussian': Adds large-variance Gaussian noise (+ 5.0 * N(0, 1)).
        - 'model_replacement': Multiplies model parameters by 10.0x scale.
    """
    attacked: Dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if attack == "sign_flip":
            attacked[k] = -3.0 * v
        elif attack == "gaussian":
            attacked[k] = v + torch.randn_like(v) * 5.0
        elif attack == "model_replacement":
            attacked[k] = v * 10.0
        else:
            attacked[k] = v.clone()
    return attacked
