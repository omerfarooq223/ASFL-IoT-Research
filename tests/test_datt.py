import numpy as np
from asfl_iot.datt import compute_datt_threshold, datt_filter, static_trust_filter


def test_datt_threshold():
    trust = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1]
    anomaly = [0.05, 0.05, 0.1, 0.08, 0.1, 0.15, 0.6, 0.7, 0.8, 0.9]
    theta, mu = compute_datt_threshold(trust, anomaly)
    assert 0.15 <= theta <= 0.85
    assert 0.0 <= mu <= 1.0


def test_datt_filter():
    trust = np.array([0.9, 0.85, 0.8, 0.3, 0.2])
    anomaly = np.array([0.05, 0.05, 0.05, 0.8, 0.9])
    admitted, theta, mu = datt_filter(trust, anomaly)
    # The first three benign clients should be admitted
    assert 0 in admitted
    assert 1 in admitted
    assert 2 in admitted
    # Highly anomalous malicious clients should be filtered out
    assert 3 not in admitted
    assert 4 not in admitted
