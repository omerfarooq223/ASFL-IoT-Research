"""Evaluation metrics, confidence intervals, and statistical aggregators for ASFL-IoT."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from .dataset import make_loader


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Evaluate accuracy and macro-averaged F1 score on a given dataset.

    Returns:
        dict with 'accuracy' and 'macro_f1'
    """
    if len(y) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}

    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    model.eval()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    preds: List[int] = []
    targets: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.argmax(dim=-1).cpu().numpy().tolist())
            targets.extend(yb.numpy().tolist())

    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": f1}


def compute_summary_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Aggregate simulation runs across seeds, computing mean, standard deviation,

    and two-sided 95% Student-t / asymptotic normal confidence intervals (ci95 = 1.96 * std / sqrt(n)).
    """
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = len(sub)
        row["n"] = n
        for m in metric_cols:
            if m not in sub.columns:
                continue
            mean_val = float(sub[m].mean())
            std_val = float(sub[m].std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std_val / math.sqrt(n)) if n > 1 else 0.0
            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val
            row[f"{m}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format mean +/- standard deviation string for academic tables."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"
