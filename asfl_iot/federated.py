"""Core federated learning simulation orchestrator for ASFL-IoT.

Coordinates the 5-stage risk-coupled round lifecycle:
1. Client Admission: Risk-coupled DATT vs Static Trust Threshold
2. Connectivity & Staleness: AsyncFLP staleness decay vs Synchronous Quorum Blocking
3. Byzantine Defense: BRAE median-MAD and cosine filtering against poisoning attacks
4. Privacy Protection: Sensitivity-Aware Differential Privacy (SADP) vs Fixed DP
5. Personalization: PFL shared body + local classification heads vs monolithic global model
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .asyncflp import compute_staleness_weights, is_quorum_satisfied
from .brae import brae_filter, make_byzantine
from .config import DEFAULT_CONFIG
from .datt import datt_filter, static_trust_filter
from .dataset import dirichlet_partition, load_dataset, make_loader
from .metrics import evaluate_model
from .models import SmallMLP, average_params, get_params, set_params
from .sadp import apply_dp_mode


def set_seed(seed: int) -> None:
    """Set random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2,
    lr: float = 0.02,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Execute local SGD training on private client data."""
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()


def run_federated(
    dataset_name: str = "digits",
    method: str = "fedavg",
    seed: int = 42,
    alpha: float = 0.5,
    rounds: Optional[int] = None,
    hidden: int = 96,
    num_clients: Optional[int] = None,
    packet_loss: float = 0.0,
    byz_frac: float = 0.0,
    attack: str = "sign_flip",
    use_static_trust: bool = True,
    use_datt: bool = False,
    use_async: bool = False,
    sync_blocking: bool = False,
    use_brae: bool = False,
    dp_mode: str = "none",
    use_pfl: bool = False,
    return_state: bool = False,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Execute one federated simulation under the specified security and network conditions.

    Returns:
        df: DataFrame tracking round-by-round accuracy, macro_f1, online/used clients, detected Byzantine nodes.
        state (optional): Internal model state dictionary for downstream personalized evaluation.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    set_seed(seed)
    rounds = cfg["rounds"] if rounds is None else rounds
    num_clients = cfg["num_clients"] if num_clients is None else num_clients

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset & partition non-IID
    X_train, X_test, y_train, y_test, num_classes, input_dim = load_dataset(dataset_name, seed)
    parts = dirichlet_partition(y_train, num_clients=num_clients, alpha=alpha, seed=seed)

    global_model = SmallMLP(input_dim, num_classes, hidden=hidden).to(device)
    global_params = get_params(global_model)
    client_heads: List[Optional[Dict[str, torch.Tensor]]] = [None] * num_clients

    prev_theta = float(cfg["trust_threshold"])
    prev_mu = float(cfg["trust_threshold"])
    staleness = np.zeros(num_clients, dtype=np.float64)
    history: List[Dict[str, Any]] = []
    skipped_rounds = 0

    num_byz = int(num_clients * byz_frac)
    byz_clients = set(np.random.choice(num_clients, num_byz, replace=False)) if num_byz > 0 else set()

    start_time = time.time()
    for r in range(rounds):
        # 1. Simulate trust scores and anomaly telemetry
        trust_scores = np.random.normal(0.72, 0.08, size=num_clients)
        anomaly_scores = np.random.beta(1.2, 10.0, size=num_clients) * 0.35

        if byz_frac > 0:
            for cid in byz_clients:
                # Trust-warming behavior: adversaries slowly climb trust score
                trust_scores[cid] = np.random.normal(0.58 + 0.005 * r, 0.06)
                anomaly_scores[cid] = np.random.beta(7.0, 2.0) * 0.65 + 0.25

        trust_scores = np.clip(trust_scores, 0.0, 1.0)
        anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

        # 2. Admission control (DATT vs Static vs Permissive)
        if use_datt:
            admitted, prev_theta, prev_mu = datt_filter(
                trust_scores,
                anomaly_scores,
                prev_theta=prev_theta,
                prev_mu=prev_mu,
                drift_penalty=cfg["datt_drift_penalty"],
                anomaly_penalty=cfg["datt_anomaly_penalty"],
                iqr_scale=cfg["datt_iqr_scale"],
                rho=cfg["datt_rho"],
            )
        elif use_static_trust:
            admitted = static_trust_filter(trust_scores, threshold=cfg["trust_threshold"])
        else:
            admitted = list(range(num_clients))

        # 3. Radio link packet loss
        online = [cid for cid in admitted if np.random.rand() >= packet_loss]

        # 4. Synchronous blocking check
        if sync_blocking:
            if not is_quorum_satisfied(len(online), num_clients, quorum_fraction=cfg["sync_quorum"]):
                skipped_rounds += 1
                staleness += 1.0
                metrics = evaluate_model(global_model, X_test, y_test, device=device)
                metrics.update({
                    "round": r + 1, "seed": seed, "dataset": dataset_name, "method": method,
                    "online_clients": len(online), "used_clients": 0, "detected": 0,
                    "packet_loss": packet_loss, "byz_frac": byz_frac, "alpha": alpha,
                    "skipped": 1, "elapsed_sec": time.time() - start_time,
                })
                history.append(metrics)
                continue

        if len(online) == 0:
            skipped_rounds += 1
            staleness += 1.0
            continue

        # 5. Client local training
        client_params: List[Dict[str, torch.Tensor]] = []
        client_ids: List[int] = []

        for cid in online:
            idx = parts[cid]
            if len(idx) == 0:
                continue

            local_model = SmallMLP(input_dim, num_classes, hidden=hidden).to(device)

            if use_pfl and client_heads[cid] is not None:
                # Personalization: load global representation body + private local head
                set_params(local_model, global_params, body_only=True)
                sd = local_model.state_dict()
                for k, v in client_heads[cid].items():
                    sd[k] = v.to(device)
                local_model.load_state_dict(sd)
            else:
                set_params(local_model, global_params, body_only=False)

            train_local(
                local_model,
                X_train[idx],
                y_train[idx],
                epochs=cfg["local_epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                device=device,
            )

            c_params = get_params(local_model)

            if use_pfl:
                # Retain classification head parameters locally
                client_heads[cid] = {k: v for k, v in c_params.items() if k.startswith("head.")}

            # Inject Byzantine attack if client is compromised
            if cid in byz_clients:
                c_params = make_byzantine(c_params, attack=attack)

            client_params.append(c_params)
            client_ids.append(cid)

        if len(client_params) == 0:
            skipped_rounds += 1
            staleness += 1.0
            continue

        # 6. Byzantine-robust filtering (BRAE)
        detected = 0
        if use_brae:
            keep_idx, info = brae_filter(global_params, client_params)
            detected = info.get("detected", 0)
            client_params = [client_params[i] for i in keep_idx]
            client_ids = [client_ids[i] for i in keep_idx]

        if len(client_params) == 0:
            skipped_rounds += 1
            staleness += 1.0
            continue

        # 7. Asynchronous staleness weighting (AsyncFLP)
        if use_async:
            weights = compute_staleness_weights(
                staleness,
                client_ids,
                staleness_lambda=cfg["staleness_lambda"],
            ).tolist()
        else:
            weights = None

        # 8. Model aggregation (only body is aggregated if PFL is active)
        new_global = average_params(client_params, weights=weights, body_only=use_pfl)
        if use_pfl:
            merged = global_params.copy()
            merged.update(new_global)
            global_params = merged
        else:
            global_params = new_global

        # 9. Differential privacy injection (SADP vs Fixed DP)
        global_params = apply_dp_mode(
            global_params,
            mode=dp_mode,
            fixed_noise=cfg["fixed_dp_noise"],
            sadp_base_noise=cfg["sadp_noise"],
        )

        set_params(global_model, global_params, body_only=False)

        # Update staleness
        staleness += 1.0
        for cid in client_ids:
            staleness[cid] = 0.0

        # Round evaluation
        metrics = evaluate_model(global_model, X_test, y_test, device=device)
        metrics.update({
            "round": r + 1, "seed": seed, "dataset": dataset_name, "method": method,
            "online_clients": len(online), "used_clients": len(client_params), "detected": detected,
            "packet_loss": packet_loss, "byz_frac": byz_frac, "alpha": alpha,
            "skipped": 0, "elapsed_sec": time.time() - start_time,
        })
        history.append(metrics)

    if len(history) == 0:
        history = [{
            "seed": seed, "dataset": dataset_name, "method": method, "round": 0,
            "accuracy": float("nan"), "macro_f1": float("nan"),
            "online_clients": 0, "used_clients": 0, "detected": 0,
            "packet_loss": packet_loss, "byz_frac": byz_frac,
            "alpha": alpha, "skipped": skipped_rounds,
            "elapsed_sec": time.time() - start_time,
        }]

    df = pd.DataFrame(history)

    if return_state:
        state = {
            "global_params": global_params,
            "client_heads": client_heads,
            "parts": parts,
            "data": (X_train, X_test, y_train, y_test, num_classes, input_dim),
        }
        return df, state

    return df
