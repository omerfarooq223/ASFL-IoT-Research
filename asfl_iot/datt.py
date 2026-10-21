"""Dynamic Adaptive Trust Thresholding (DATT) for ASFL-IoT.

Coordinates admission control under trust warming and network volatility.
Addresses Section III-A (Eq. 2-3) of the IEEE ICES&T 2026 paper.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def compute_datt_threshold(
    trust_scores: Union[List[float], np.ndarray],
    anomaly_scores: Optional[Union[List[float], np.ndarray]] = None,
    prev_theta: float = 0.55,
    prev_mu: float = 0.55,
    drift_penalty: float = 0.6,
    anomaly_penalty: float = 0.50,
    iqr_scale: float = 0.50,
    rho: float = 0.30,
) -> Tuple[float, float]:
    """Compute the dynamic risk-coupled admission threshold for round t.

    Parameters:
        trust_scores: Historical and current observed trust values across clients [0, 1].
        anomaly_scores: Update geometry anomaly indicators [0, 1].
        prev_theta: Admission threshold from round t-1.
        prev_mu: Fleet average trust from round t-1.
        drift_penalty: Weight lambda_d penalizing trust drift.
        anomaly_penalty: Weight lambda_a penalizing update anomaly.
        iqr_scale: Volatility sensitivity factor kappa scaling the Interquartile Range.
        rho: Inertia damping factor for threshold momentum.

    Returns:
        theta: Calibrated admission threshold for round t.
        mu: Current round fleet mean trust.
    """
    trust_scores = np.asarray(trust_scores, dtype=np.float64)
    if anomaly_scores is None:
        anomaly_scores = np.zeros_like(trust_scores)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)

    q1, q3 = np.percentile(trust_scores, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    mu = float(trust_scores.mean())
    volatility = iqr_scale * iqr
    drift = max(0.0, prev_mu - mu)

    # Risk-adjusted scores
    risk_adjusted = trust_scores - anomaly_penalty * anomaly_scores
    median_val = float(np.median(risk_adjusted))

    theta = median_val - volatility + drift_penalty * drift + rho * (prev_theta - prev_mu)
    theta = float(np.clip(theta, 0.15, 0.85))
    return theta, mu


def static_trust_filter(
    trust_scores: Union[List[float], np.ndarray],
    threshold: float = 0.55,
) -> List[int]:
    """Filter clients using a fixed static trust threshold (Baseline SEPP-IoT)."""
    return [i for i, t in enumerate(trust_scores) if float(t) >= threshold]


def datt_filter(
    trust_scores: Union[List[float], np.ndarray],
    anomaly_scores: Optional[Union[List[float], np.ndarray]] = None,
    prev_theta: float = 0.55,
    prev_mu: float = 0.55,
    drift_penalty: float = 0.6,
    anomaly_penalty: float = 0.50,
    iqr_scale: float = 0.50,
    rho: float = 0.30,
) -> Tuple[List[int], float, float]:
    """Admit only clients whose risk-adjusted trust score meets or exceeds the DATT threshold.

    Returns:
        admitted_indices: List of admitted client indices for aggregation.
        new_theta: Updated threshold for round t.
        new_mu: Updated mean trust for round t.
    """
    theta, mu = compute_datt_threshold(
        trust_scores,
        anomaly_scores=anomaly_scores,
        prev_theta=prev_theta,
        prev_mu=prev_mu,
        drift_penalty=drift_penalty,
        anomaly_penalty=anomaly_penalty,
        iqr_scale=iqr_scale,
        rho=rho,
    )

    trust_arr = np.asarray(trust_scores, dtype=np.float64)
    if anomaly_scores is None:
        risk_adjusted = trust_arr
    else:
        risk_adjusted = trust_arr - anomaly_penalty * np.asarray(anomaly_scores, dtype=np.float64)

    admitted = [i for i, s in enumerate(risk_adjusted) if s >= theta]
    return admitted, theta, mu
