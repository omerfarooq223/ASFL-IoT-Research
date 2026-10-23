"""Personalized Federated Learning (PFL) utilities for ASFL-IoT.

Preserves client-specific classification heads locally while sharing
foundational representation layers across edge devices.
Addresses Section III-B (Eq. 6) of the IEEE ICES&T 2026 paper.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from .models import SmallMLP, set_params
from .metrics import evaluate_model


def evaluate_personalized_clients(
    state: Dict[str, Any],
    use_pfl: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate each client's personalized performance on held-out private data.

    Parameters:
        state: State dictionary returned by run_federated(..., return_state=True).
        use_pfl: Whether to attach client's local personalized head.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        summary: Dictionary with mean_client_accuracy, worst_client_accuracy,
                 client_accuracy_std, and mean_client_macro_f1.
        client_df: DataFrame with individual client metrics.
    """
    global_params = state["global_params"]
    client_heads = state["client_heads"]
    parts = state["parts"]
    X_train, X_test, y_train, y_test, num_classes, input_dim = state["data"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_metrics: List[Dict[str, Any]] = []

    for cid, idx in enumerate(parts):
        if len(idx) < 5:
            continue
        Xc, yc = X_train[idx], y_train[idx]
        if len(np.unique(yc)) < 2:
            continue

        bincounts = np.bincount(yc)
        can_stratify = len(bincounts) >= 2 and min(bincounts[bincounts > 0]) >= 2

        tr_idx, te_idx = train_test_split(
            np.arange(len(yc)),
            test_size=0.35,
            random_state=cid + 123,
            stratify=yc if can_stratify else None,
        )

        model = SmallMLP(input_dim, num_classes).to(device)
        set_params(model, global_params)

        if use_pfl and client_heads[cid] is not None:
            sd = model.state_dict()
            for k, v in client_heads[cid].items():
                sd[k] = v.to(device)
            model.load_state_dict(sd)

        metrics = evaluate_model(model, Xc[te_idx], yc[te_idx], device=device)
        metrics.update({
            "client": cid,
            "client_size": len(idx),
        })
        client_metrics.append(metrics)

    client_df = pd.DataFrame(client_metrics)
    summary = {
        "mean_client_accuracy": float(client_df["accuracy"].mean()) if not client_df.empty else float("nan"),
        "worst_client_accuracy": float(client_df["accuracy"].min()) if not client_df.empty else float("nan"),
        "client_accuracy_std": float(client_df["accuracy"].std()) if not client_df.empty else float("nan"),
        "mean_client_macro_f1": float(client_df["macro_f1"].mean()) if not client_df.empty else float("nan"),
    }
    return summary, client_df
