import numpy as np
from asfl_iot.dataset import load_digits_dataset, load_synthetic_iot, dirichlet_partition, make_loader


def test_load_digits():
    X_tr, X_te, y_tr, y_te, n_cls, dim = load_digits_dataset(seed=42)
    assert len(X_tr) > 0 and len(X_te) > 0
    assert n_cls == 10
    assert dim == 64


def test_load_synthetic_iot():
    X_tr, X_te, y_tr, y_te, n_cls, dim = load_synthetic_iot(seed=42)
    assert len(X_tr) == 3000
    assert len(X_te) == 1000
    assert n_cls == 4
    assert dim == 30


def test_dirichlet_partition():
    y = np.random.randint(0, 10, size=500)
    parts = dirichlet_partition(y, num_clients=5, alpha=0.5, seed=42, min_size=5)
    assert len(parts) == 5
    for p in parts:
        assert len(p) >= 5
    # Total count matches
    assert sum(len(p) for p in parts) == len(y)
