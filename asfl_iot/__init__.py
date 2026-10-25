"""ASFL-IoT: Risk-Coupled Adaptive Secure Federated Learning for Trust-Aware IoT Systems.

Official Python package for the IEEE ICES&T 2026 publication.
"""

from .config import DEFAULT_CONFIG, get_default_config, load_config, save_config
from .models import SmallMLP, get_params, set_params, flatten_params, average_params
from .dataset import load_dataset, load_digits_dataset, load_synthetic_iot, dirichlet_partition, make_loader
from .datt import compute_datt_threshold, datt_filter, static_trust_filter
from .brae import brae_filter, make_byzantine, cosine_similarity
from .sadp import add_sadp, add_fixed_dp, apply_dp_mode
from .asyncflp import compute_staleness_weights, is_quorum_satisfied
from .pfl import evaluate_personalized_clients
from .federated import run_federated, set_seed
from .metrics import evaluate_model, compute_summary_table, format_mean_std

__version__ = "1.0.0"
__all__ = [
    "DEFAULT_CONFIG",
    "get_default_config",
    "load_config",
    "save_config",
    "SmallMLP",
    "get_params",
    "set_params",
    "flatten_params",
    "average_params",
    "load_dataset",
    "load_digits_dataset",
    "load_synthetic_iot",
    "dirichlet_partition",
    "make_loader",
    "compute_datt_threshold",
    "datt_filter",
    "static_trust_filter",
    "brae_filter",
    "make_byzantine",
    "cosine_similarity",
    "add_sadp",
    "add_fixed_dp",
    "apply_dp_mode",
    "compute_staleness_weights",
    "is_quorum_satisfied",
    "evaluate_personalized_clients",
    "run_federated",
    "set_seed",
    "evaluate_model",
    "compute_summary_table",
    "format_mean_std",
]
