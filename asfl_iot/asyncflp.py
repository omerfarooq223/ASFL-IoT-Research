"""Asynchronous Federated Learning Protocol (AsyncFLP) for ASFL-IoT.

Mitigates intermittent wireless connectivity and packet loss through staleness decay.
Addresses Section III-A (Eq. 4) of the IEEE ICES&T 2026 paper.
"""

import math
from typing import List, Optional
import numpy as np


def compute_staleness_weights(
    staleness_vector: np.ndarray,
    active_client_ids: List[int],
    staleness_lambda: float = 0.3,
) -> np.ndarray:
    """Compute normalized exponential staleness decay aggregation weights.

    Formula (Eq. 4 in manuscript):
        q_i^{(t)} = exp(-lambda * tau_i^{(t)})
        omega_i^{(t)} = q_i^{(t)} / sum_j q_j^{(t)}

    Parameters:
        staleness_vector: Array containing accumulated rounds of staleness per client.
        active_client_ids: Indices of clients contributing updates in this round.
        staleness_lambda: Exponential decay coefficient prioritizing fresher updates.

    Returns:
        Normalized weights summing to 1.0 across active clients.
    """
    if len(active_client_ids) == 0:
        return np.array([], dtype=np.float64)

    raw_weights = np.array([
        math.exp(-staleness_lambda * float(staleness_vector[cid]))
        for cid in active_client_ids
    ], dtype=np.float64)

    total = float(raw_weights.sum())
    if total <= 1e-12:
        return np.ones(len(active_client_ids), dtype=np.float64) / len(active_client_ids)
    return raw_weights / total


def is_quorum_satisfied(
    online_count: int,
    total_clients: int,
    quorum_fraction: float = 0.8,
) -> bool:
    """Determine whether the online client count satisfies the synchronous aggregation quorum."""
    required = int(math.ceil(quorum_fraction * total_clients))
    return online_count >= required
