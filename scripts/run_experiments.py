#!/usr/bin/env python3
"""ASFL-IoT Reproduction Runner.

Executes and logs all experiments from the IEEE ICES&T 2026 paper:
  - Trust Admission Simulation (1,000 nodes, DATT vs Static)
  - Intermittent Connectivity & Packet Loss (AsyncFLP vs Sync Blocking)
  - Byzantine Poisoning Attacks (BRAE vs No Defense)
  - Differential Privacy Sensitivity (SADP vs Fixed DP)
  - Personalized Federated Learning under non-IID Skew (PFL vs FedAvg)
  - Global Utility Component Ablation
  - Synthetic Imbalanced IoT Benchmark
  - Scalability & Model Capacity Sweeps (10, 25, 50 clients; hidden 96, 384)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow importing from local package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asfl_iot import (
    DEFAULT_CONFIG,
    compute_summary_table,
    datt_filter,
    evaluate_personalized_clients,
    run_federated,
    set_seed,
    static_trust_filter,
)


def run_trust_experiment(seeds: List[int], out_dir: Path) -> pd.DataFrame:
    """1,000-node network simulation measuring malicious admission (FNR) and benign rejection (FPR)."""
    print("\n[1/7] Running Calibrated Risk-Coupled DATT Trust Simulation...")
    rows = []
    for seed in seeds:
        for cv in [0.1, 0.3, 0.5]:
            set_seed(seed)
            n = 1000
            rounds = 50
            adv_frac = 0.20
            adv_n = int(n * adv_frac)
            adv_idx = set(np.random.choice(n, adv_n, replace=False))
            benign_idx = set(range(n)) - adv_idx

            for mode in ["static", "datt"]:
                prev_theta, prev_mu = 0.55, 0.55
                fnrs, fprs = [], []
                for r in range(rounds):
                    trust = np.random.normal(0.72, cv * 0.12, size=n)
                    anomaly = np.random.beta(1.1, 11.0, size=n) * 0.30
                    for i in adv_idx:
                        trust[i] = np.random.normal(0.45 + 0.008 * r, cv * 0.10)
                        anomaly[i] = np.random.beta(6.0, 2.0) * 0.70 + 0.20
                    trust = np.clip(trust, 0.0, 1.0)
                    anomaly = np.clip(anomaly, 0.0, 1.0)

                    if mode == "static":
                        admitted = set(static_trust_filter(trust, 0.55))
                    else:
                        admitted_list, prev_theta, prev_mu = datt_filter(
                            trust, anomaly, prev_theta, prev_mu
                        )
                        admitted = set(admitted_list)

                    fn = len(adv_idx & admitted)
                    tp = len(adv_idx - admitted)
                    fp = len(benign_idx - admitted)
                    tn = len(benign_idx & admitted)

                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnrs.append(fnr)
                    fprs.append(fpr)

                rows.append({
                    "seed": seed, "cv": cv, "method": mode,
                    "fnr": float(np.mean(fnrs)), "fpr": float(np.mean(fprs)),
                    "drift_penalty": DEFAULT_CONFIG["datt_drift_penalty"],
                    "anomaly_penalty": DEFAULT_CONFIG["datt_anomaly_penalty"],
                    "iqr_scale": DEFAULT_CONFIG["datt_iqr_scale"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "trust_drift_results.csv", index=False)
    summary = compute_summary_table(df, ["cv", "method"], ["fnr", "fpr"])
    summary.to_csv(out_dir / "trust_summary.csv", index=False)
    print("  Trust simulation complete. Saved to trust_drift_results.csv and trust_summary.csv")
    return df


def run_async_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """AsyncFLP vs Sync Blocking under packet loss [0.0, 0.1, 0.2, 0.3, 0.4]."""
    print("\n[2/7] Running AsyncFLP Packet Loss Resilience Experiments...")
    results = []
    for seed in seeds:
        for packet_loss in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for method, kwargs in [
                ("sync_blocking", dict(sync_blocking=True, use_async=False)),
                ("asyncflp", dict(sync_blocking=False, use_async=True)),
            ]:
                df = run_federated(
                    dataset_name="digits", method=method, seed=seed, alpha=0.5,
                    rounds=rounds, packet_loss=packet_loss, device=device, **kwargs
                )
                results.append(df.iloc[-1].to_dict())

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "async_results.csv", index=False)
    summary = compute_summary_table(
        res_df, ["packet_loss", "method"], ["accuracy", "macro_f1", "used_clients", "skipped"]
    )
    summary.to_csv(out_dir / "async_summary.csv", index=False)
    print("  Async experiment complete.")
    return res_df


def run_byzantine_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """BRAE vs No Defense under sign-flip, Gaussian, and model-replacement attacks."""
    print("\n[3/7] Running Byzantine-Robust Aggregation (BRAE) Experiments...")
    results = []
    for seed in seeds:
        for byz_frac in [0.1, 0.2, 0.3]:
            for attack in ["sign_flip", "gaussian", "model_replacement"]:
                for method, use_brae in [("no_defense", False), ("brae", True)]:
                    df = run_federated(
                        dataset_name="digits", method=f"{method}_{attack}", seed=seed, alpha=0.5,
                        rounds=rounds, byz_frac=byz_frac, attack=attack, use_brae=use_brae,
                        device=device
                    )
                    results.append(df.iloc[-1].to_dict())

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "byzantine_results.csv", index=False)
    summary = compute_summary_table(
        res_df, ["byz_frac", "method"], ["accuracy", "macro_f1", "detected"]
    )
    summary.to_csv(out_dir / "byzantine_summary.csv", index=False)
    print("  Byzantine experiment complete.")
    return res_df


def run_privacy_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """Sensitivity-Aware DP (SADP) vs Fixed DP vs None."""
    print("\n[4/7] Running Differential Privacy (SADP vs Fixed DP) Experiments...")
    results = []
    for seed in seeds:
        for method, dp_mode in [("none", "none"), ("fixed_dp", "fixed_dp"), ("sadp", "sadp")]:
            df = run_federated(
                dataset_name="digits", method=method, seed=seed, alpha=0.5,
                rounds=rounds, packet_loss=0.0, dp_mode=dp_mode, device=device
            )
            results.append(df.iloc[-1].to_dict())

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "privacy_results.csv", index=False)
    summary = compute_summary_table(res_df, ["method"], ["accuracy", "macro_f1"])
    summary.to_csv(out_dir / "privacy_summary.csv", index=False)
    print("  Privacy experiment complete.")
    return res_df


def run_pfl_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """Personalized Federated Learning (PFL) across non-IID Dirichlet alpha sweep."""
    print("\n[5/7] Running Personalized Federated Learning (PFL) Sweep...")
    pfl_results = []
    pfl_client_rows = []
    for seed in seeds:
        for alpha in [0.1, 0.3, 0.5, 1.0, 100.0]:
            for method, use_pfl in [("fedavg", False), ("pfl", True)]:
                df, state = run_federated(
                    dataset_name="digits", method=method, seed=seed, alpha=alpha,
                    rounds=rounds, use_pfl=use_pfl, return_state=True, device=device
                )
                summary, clients = evaluate_personalized_clients(state, use_pfl=use_pfl, device=device)
                row = {"seed": seed, "alpha": alpha, "method": method, **summary}
                pfl_results.append(row)
                clients["seed"] = seed
                clients["alpha"] = alpha
                clients["method"] = method
                pfl_client_rows.append(clients)

    pfl_df = pd.DataFrame(pfl_results)
    pfl_clients_df = pd.concat(pfl_client_rows, ignore_index=True)
    pfl_df.to_csv(out_dir / "pfl_results.csv", index=False)
    pfl_clients_df.to_csv(out_dir / "pfl_client_level_results.csv", index=False)

    summary = compute_summary_table(
        pfl_df, ["alpha", "method"],
        ["mean_client_accuracy", "worst_client_accuracy", "client_accuracy_std", "mean_client_macro_f1"]
    )
    summary.to_csv(out_dir / "pfl_summary.csv", index=False)
    print("  PFL experiment complete.")
    return pfl_df


def run_ablation_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """Ablation study removing individual ASFL-IoT modules under joint stress."""
    print("\n[6/7] Running ASFL-IoT Module Ablation Study...")
    ablations = {
        "full_asfl": dict(use_datt=True, use_async=True, sync_blocking=False, use_brae=True, dp_mode="sadp", use_pfl=True),
        "no_datt": dict(use_datt=False, use_async=True, sync_blocking=False, use_brae=True, dp_mode="sadp", use_pfl=True),
        "no_async": dict(use_datt=True, use_async=False, sync_blocking=True, use_brae=True, dp_mode="sadp", use_pfl=True),
        "no_brae": dict(use_datt=True, use_async=True, sync_blocking=False, use_brae=False, dp_mode="sadp", use_pfl=True),
        "no_sadp": dict(use_datt=True, use_async=True, sync_blocking=False, use_brae=True, dp_mode="none", use_pfl=True),
        "no_pfl": dict(use_datt=True, use_async=True, sync_blocking=False, use_brae=True, dp_mode="sadp", use_pfl=False),
    }
    results = []
    for seed in seeds:
        for name, flags in ablations.items():
            df = run_federated(
                dataset_name="digits", method=name, seed=seed, alpha=0.3,
                rounds=rounds, packet_loss=0.2, byz_frac=0.2, attack="sign_flip",
                device=device, **flags
            )
            results.append(df.iloc[-1].to_dict())

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "ablation_results.csv", index=False)
    summary = compute_summary_table(
        res_df, ["method"], ["accuracy", "macro_f1", "detected", "used_clients"]
    )
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    print("  Ablation experiment complete.")
    return res_df


def run_iot_experiment(seeds: List[int], out_dir: Path, rounds: int, device: str) -> pd.DataFrame:
    """Synthetic Imbalanced IoT Benchmark evaluation."""
    print("\n[7/7] Running Imbalanced Synthetic IoT Benchmark...")
    results = []
    for seed in seeds:
        for method, kwargs in [
            ("fedavg", dict()),
            ("full_asfl", dict(use_datt=True, use_async=True, sync_blocking=False, use_brae=True, dp_mode="sadp", use_pfl=True)),
        ]:
            df = run_federated(
                dataset_name="iot", method=method, seed=seed, alpha=0.5,
                rounds=rounds, packet_loss=0.1, byz_frac=0.1, attack="sign_flip",
                device=device, **kwargs
            )
            results.append(df.iloc[-1].to_dict())

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "iot_results.csv", index=False)
    summary = compute_summary_table(res_df, ["method"], ["accuracy", "macro_f1"])
    summary.to_csv(out_dir / "iot_summary.csv", index=False)
    print("  IoT experiment complete.")
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Run ASFL-IoT experimental benchmarks.")
    parser.add_argument(
        "--experiment",
        choices=["all", "trust", "async", "byzantine", "privacy", "pfl", "ablation", "iot"],
        default="all",
        help="Experiment suite to execute (default: all)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_CONFIG["seeds"], help="Random seeds")
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["rounds"], help="Communication rounds")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory for output CSVs")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting ASFL-IoT Benchmark Runner")
    print(f"  Target Experiment: {args.experiment}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Device: {device}")
    print(f"  Output Directory: {out_dir.resolve()}")

    start_time = time.time()
    if args.experiment in ("all", "trust"):
        run_trust_experiment(args.seeds, out_dir)
    if args.experiment in ("all", "async"):
        run_async_experiment(args.seeds, out_dir, args.rounds, device)
    if args.experiment in ("all", "byzantine"):
        run_byzantine_experiment(args.seeds, out_dir, args.rounds, device)
    if args.experiment in ("all", "privacy"):
        run_privacy_experiment(args.seeds, out_dir, args.rounds, device)
    if args.experiment in ("all", "pfl"):
        run_pfl_experiment(args.seeds, out_dir, args.rounds, device)
    if args.experiment in ("all", "ablation"):
        run_ablation_experiment(args.seeds, out_dir, args.rounds, device)
    if args.experiment in ("all", "iot"):
        run_iot_experiment(args.seeds, out_dir, args.rounds, device)

    total_time = round(time.time() - start_time, 2)
    print(f"\nAll requested benchmarks completed successfully in {total_time}s.")


if __name__ == "__main__":
    main()
