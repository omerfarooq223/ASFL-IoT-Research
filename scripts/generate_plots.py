#!/usr/bin/env python3
"""Script to generate all publication-ready figures for ASFL-IoT.

Produces both PDF vector graphics and PNG high-resolution images:
  1. trust_fnr_fpr_vs_cv: Risk-Coupled DATT vs Static Admission Threshold
  2. async_packet_loss_accuracy: AsyncFLP vs Synchronous Blocking
  3. byzantine_accuracy: BRAE vs No Defense under Poisoning Attacks
  4. pfl_alpha_sweep: Personalized FL vs Global Model under non-IID Skew
  5. ablation_accuracy: Impact of individual security and privacy components
  6. scalability_macro_f1: Scalability across client fleet sizes and model capacities
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Set academic aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_trust(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "trust_drift_results.csv"
    if not csv_path.exists():
        print(f"Skipping trust plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    metrics = [("fnr", "Malicious Admission (FNR)"), ("fpr", "Benign Rejection (FPR)")]
    for (metric, title), ax in zip(metrics, axes):
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            g = sub.groupby("cv")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                g["cv"], g["mean"], yerr=g["std"],
                marker="o", capsize=4, label=method.upper() if method == "datt" else "Static"
            )
        ax.set_xlabel("Network Volatility CV")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend(frameon=True)

    fig.suptitle("DATT vs Static Trust Admission", y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"trust_fnr_fpr_vs_cv.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved trust_fnr_fpr_vs_cv.{pdf,png}")


def plot_async(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "async_results.csv"
    if not csv_path.exists():
        print(f"Skipping async plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    method_labels = {"sync_blocking": "Sync Quorum (Blocking)", "asyncflp": "AsyncFLP (Proposed)"}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        g = sub.groupby("packet_loss")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            g["packet_loss"] * 100, g["mean"], yerr=g["std"],
            marker="s" if method == "asyncflp" else "o",
            capsize=4, label=method_labels.get(method, method)
        )
    ax.set_xlabel("Packet Loss Rate (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("AsyncFLP under Intermittent Connectivity")
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"async_packet_loss_accuracy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved async_packet_loss_accuracy.{pdf,png}")


def plot_byzantine(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "byzantine_results.csv"
    if not csv_path.exists():
        print(f"Skipping Byzantine plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    tmp = df[df["method"].str.contains("sign_flip")]
    for method in sorted(tmp["method"].unique()):
        sub = tmp[tmp["method"] == method]
        g = sub.groupby("byz_frac")["accuracy"].agg(["mean", "std"]).reset_index()
        label = "BRAE (Defense Active)" if "brae" in method else "No Defense"
        ax.errorbar(
            g["byz_frac"] * 100, g["mean"], yerr=g["std"],
            marker="^" if "brae" in method else "x",
            capsize=4, label=label
        )
    ax.set_xlabel("Byzantine Client Ratio (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("BRAE Defense under Sign-Flip Attack")
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"byzantine_accuracy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved byzantine_accuracy.{pdf,png}")


def plot_pfl(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "pfl_results.csv"
    if not csv_path.exists():
        print(f"Skipping PFL plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    method_labels = {"fedavg": "FedAvg (Monolithic)", "pfl": "PFL (Local Heads)"}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        g = sub.groupby("alpha")["mean_client_accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            g["alpha"].astype(str), g["mean"], yerr=g["std"],
            marker="d" if method == "pfl" else "o",
            capsize=4, label=method_labels.get(method, method)
        )
    ax.set_xlabel("Dirichlet Concentration Alpha (Lower = Higher Non-IID Skew)")
    ax.set_ylabel("Mean Client Accuracy")
    ax.set_title("Personalized FL under Statistical Heterogeneity")
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"pfl_alpha_sweep.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pfl_alpha_sweep.{pdf,png}")


def plot_ablation(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "ablation_results.csv"
    if not csv_path.exists():
        print(f"Skipping ablation plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    g = df.groupby("method")["accuracy"].agg(["mean", "std"]).reset_index()
    order = ["full_asfl", "no_datt", "no_async", "no_brae", "no_sadp", "no_pfl"]
    g["sort_key"] = g["method"].map(lambda x: order.index(x) if x in order else 99)
    g = g.sort_values("sort_key")

    display_names = {
        "full_asfl": "Full ASFL-IoT",
        "no_datt": "w/o DATT",
        "no_async": "w/o AsyncFLP",
        "no_brae": "w/o BRAE",
        "no_sadp": "w/o SADP",
        "no_pfl": "w/o PFL",
    }
    labels = [display_names.get(m, m) for m in g["method"]]
    ax.bar(labels, g["mean"], yerr=g["std"], capsize=4, color="#1B4965", alpha=0.85)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("ASFL-IoT Component Ablation under Joint Stress")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"ablation_accuracy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ablation_accuracy.{pdf,png}")


def plot_scalability(res_dir: Path, fig_dir: Path) -> None:
    csv_path = res_dir / "generalization_summary.csv"
    if not csv_path.exists():
        print(f"Skipping scalability plot: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    scale_only = df[df["experiment"] == "scalability"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, dataset_name in enumerate(["digits", "iot"]):
        for col, rounds in enumerate([20, 50]):
            ax = axes[row, col]
            sub = scale_only[
                (scale_only["dataset"] == dataset_name) &
                (scale_only["rounds"] == rounds)
            ]
            for method in ["fedavg", "asfl_core"]:
                m = sub[sub["method"] == method]
                label = "ASFL-Core" if method == "asfl_core" else "FedAvg Baseline"
                ax.errorbar(
                    m["clients"], m["macro_f1_mean"], yerr=m["macro_f1_std"],
                    marker="o", capsize=4, label=label
                )
            ax.set_title(f"{dataset_name.capitalize()} Dataset ({rounds} Rounds)")
            ax.set_xlabel("Number of Edge Clients")
            ax.set_ylabel("Macro-F1 Score")
            ax.legend(frameon=True)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"scalability_macro_f1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scalability_macro_f1.{pdf,png}")


def main():
    parser = argparse.ArgumentParser(description="Generate ASFL-IoT publication figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with CSV results")
    parser.add_argument("--fig-dir", type=str, default="results/figures", help="Directory for generated plots")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from: {res_dir.resolve()} -> {fig_dir.resolve()}")
    plot_trust(res_dir, fig_dir)
    plot_async(res_dir, fig_dir)
    plot_byzantine(res_dir, fig_dir)
    plot_pfl(res_dir, fig_dir)
    plot_ablation(res_dir, fig_dir)
    plot_scalability(res_dir, fig_dir)
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
