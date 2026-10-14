"""Model architectures and parameter manipulation utilities for ASFL-IoT."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """Multilayer Perceptron with explicit body (feature extractor) and head (classifier).

    This decoupling allows Personalized Federated Learning (PFL) where the feature
    extractor is shared globally while the classification head is retained locally.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 96):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden

        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def get_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract a detached, CPU-cloned state dictionary from a model."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_params(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    body_only: bool = False,
) -> None:
    """Load parameter dictionary into model, optionally updating body layers only."""
    sd = model.state_dict()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    for k, v in params.items():
        if body_only and k.startswith("head."):
            continue
        if k in sd:
            sd[k] = v.to(device)
    model.load_state_dict(sd)


def flatten_params(
    params: Dict[str, torch.Tensor],
    body_only: bool = False,
) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    vecs = []
    for k, v in params.items():
        if body_only and k.startswith("head."):
            continue
        vecs.append(v.reshape(-1).float())
    return torch.cat(vecs)


def average_params(
    params_list: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    body_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute weighted average of a list of model parameter dictionaries.

    If weights is None, uniform averaging is applied.
    """
    if not params_list:
        raise ValueError("Cannot average an empty list of parameters.")

    keys = list(params_list[0].keys())
    if body_only:
        keys = [k for k in keys if not k.startswith("head.")]

    n = len(params_list)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    avg_params: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = torch.zeros_like(params_list[0][k], dtype=torch.float32)
        for p, w in zip(params_list, weights):
            acc += p[k].float() * float(w)
        avg_params[k] = acc
    return avg_params
