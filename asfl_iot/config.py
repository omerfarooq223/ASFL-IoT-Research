"""Configuration and hyperparameter defaults for ASFL-IoT."""

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "seeds": [1, 7, 21, 42, 100],
    "num_clients": 10,
    "rounds": 20,
    "local_epochs": 2,
    "batch_size": 32,
    "lr": 0.02,
    "dirichlet_alpha": 0.5,
    "max_train_samples": 3000,
    "trust_threshold": 0.55,
    "datt_k": 1.5,
    "datt_rho": 0.3,
    "staleness_lambda": 0.3,
    "byzantine_fraction": 0.3,
    "dp_epsilon": 0.7,
    "dp_delta": 1e-5,
    "sync_quorum": 0.8,
    "fixed_dp_noise": 0.004,
    "sadp_noise": 0.008,
    "datt_drift_penalty": 0.6,
    "datt_anomaly_penalty": 0.50,
    "datt_iqr_scale": 0.50,
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default ASFL-IoT configuration."""
    return DEFAULT_CONFIG.copy()


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file, overlaying onto default values."""
    config = get_default_config()
    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    config.update(user_config)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration dictionary to a formatted JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
