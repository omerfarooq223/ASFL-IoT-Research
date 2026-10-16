"""Data loading, non-IID Dirichlet partitioning, and loader utilities for ASFL-IoT."""

from typing import List, Tuple
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_digits_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load, standardize, and split the handwritten digits dataset.

    Returns:
        X_train, X_test, y_train, y_test, num_classes (10), input_dim (64)
    """
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, 10, X.shape[1]


def load_synthetic_iot(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Generate an imbalanced multi-class synthetic IoT dataset representing edge sensory anomalies.

    Class imbalance mimics rare-event sensing: [94%, 4%, 1.5%, 0.5%].
    Total: 4,000 samples (3,000 train, 1,000 test).

    Returns:
        X_train, X_test, y_train, y_test, num_classes (4), input_dim (30)
    """
    X, y = make_classification(
        n_samples=4000,
        n_features=30,
        n_informative=16,
        n_redundant=6,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.94, 0.04, 0.015, 0.005],
        flip_y=0.01,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1000, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, 4, X.shape[1]


def load_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Factory helper to load either 'digits' or 'iot' dataset."""
    if name == "digits":
        return load_digits_dataset(seed)
    elif name in ("iot", "synthetic_iot"):
        return load_synthetic_iot(seed)
    else:
        raise ValueError(f"Unknown dataset name: '{name}'. Supported: 'digits', 'iot'.")


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 5,
) -> List[List[int]]:
    """Partition dataset indices among clients using a Dirichlet distribution for non-IID skew.

    Smaller alpha leads to higher label concentration (more severe non-IID data).
    Guarantees each client receives at least min_size samples.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)

    for _ in range(200):
        parts: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            splits = np.split(idx, cuts)
            for cid, split in enumerate(splits):
                parts[cid].extend(split.tolist())

        if min(len(p) for p in parts) >= min_size:
            return parts

    # Fallback to stratified uniform chunks if extreme sampling produced empty client sets
    indices = np.arange(len(y))
    rng.shuffle(indices)
    return [c.tolist() for c in np.array_split(indices, num_clients)]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from feature and label arrays."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
