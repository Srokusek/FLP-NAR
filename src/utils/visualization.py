"""
Functions in this file were created with the help of GenAI

Visualization utilities for the FLP-NAR project.

Functions for plotting training history, inference results, and solution details.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from typing import Dict, List, Optional, Tuple


def plot_training_history(
    history: Dict,
    loss_config: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Plot training history: individual losses, ratios, and total weighted loss.
    
    Args:
        history: Dict with "train" (list) and "val" (dict[scale_key -> list]) keys
        loss_config: Optional dict of loss weights to determine which losses to plot
        figsize: Figure size tuple
    
    Returns:
        Tuple of (fig_losses, fig_total) matplotlib figures
    """
    if loss_config is None:
        loss_config = {}
    
    epochs = range(1, len(history["train"]) + 1)
    train_h = history["train"]
    val_h = history["val"]
    
    # Extract loss keys (non-ratio keys with positive weights)
    loss_keys = [
        k for k in loss_config.keys()
        if k not in ("optimum_diff", "dual_diff") and loss_config.get(k, 0) > 0
    ]
    ratio_keys = ["optimum_diff", "dual_diff"]
    
    n_loss = len(loss_keys)
    n_ratio = len(ratio_keys)
    total_plots = n_loss + n_ratio
    
    # Create subplots for individual losses and ratios
    fig, axes = plt.subplots(2, max(2, (total_plots + 1) // 2), figsize=figsize)
    axes = axes.flatten() if total_plots > 1 else [axes]
    
    # Plot individual losses
    for ax, key in zip(axes[:n_loss], loss_keys):
        train_vals = [e.get(key, 0) for e in train_h]
        ax.plot(epochs, train_vals, label="train", linewidth=1.5, color="blue")
        
        for vk, vh in val_h.items():
            val_vals = [e.get(key, 0) for e in vh]
            ax.plot(epochs, val_vals, label=f"val {vk}", linewidth=1, alpha=0.7)
        
        ax.set(xlabel="Epoch", ylabel=key, title=key)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Plot ratio metrics (optimum_diff, dual_diff)
    for ax, key in zip(axes[n_loss:n_loss+n_ratio], ratio_keys):
        train_vals = [e.get(key, 0) for e in train_h]
        ax.plot(epochs, train_vals, label="train", linewidth=1.5, color="blue")
        
        for vk, vh in val_h.items():
            val_vals = [e.get(key, 0) for e in vh]
            ax.plot(epochs, val_vals, label=f"val {vk}", linewidth=1, alpha=0.7)
        
        ax.axhline(1.0, color="green", ls="--", alpha=0.5, label="optimal")
        ax.set(xlabel="Epoch", ylabel=key, title=key)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for ax in axes[total_plots:]:
        ax.axis("off")
    
    fig.suptitle("Training History: Individual Losses and Ratios")
    plt.tight_layout()
    
    # Create separate figure for total weighted loss
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    def _weighted_total(losses: dict, weights: dict) -> float:
        return sum(weights.get(k, 0) * losses.get(k, 0) for k in weights if k != "_weights")
    
    train_total = [
        _weighted_total(e, e.get("_weights", loss_config))
        for e in train_h
    ]
    ax2.plot(epochs, train_total, label="train total", linewidth=2, color="blue")
    
    for vk, vh in val_h.items():
        # Use weights from corresponding training epoch
        vt = [
            _weighted_total(e, train_h[i].get("_weights", loss_config))
            for i, e in enumerate(vh)
        ]
        ax2.plot(epochs, vt, label=f"val {vk}", linewidth=1.5, alpha=0.8)
    
    ax2.set(xlabel="Epoch", ylabel="Weighted Loss", title="Total Weighted Loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, fig2


def plot_inference_results_vs_optimum(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (11, 4.5),
) -> plt.Figure:
    """
    Plot inference results comparing predicted solution to exact optimum.
    
    Args:
        results_df: DataFrame from run_inference() with columns:
                    size, predicted, optimum, n_fac_opened, n_fac_target,
                    and optionally solution_time and exact_time
    
    Returns:
        Matplotlib figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Predicted vs Optimum (scatter)
    for s in sorted(results_df["size"].unique()):
        sub = results_df[results_df["size"] == s]
        axes[0].scatter(sub["optimum"], sub["predicted"], label=s, alpha=0.6, s=40)
    
    mx = max(results_df["optimum"].max(), results_df["predicted"].max())
    axes[0].plot([0, mx], [0, mx], "k--", alpha=0.5, linewidth=2, label="Perfect")
    axes[0].set(xlabel="Exact Optimum", ylabel="Predicted Cost", 
                title="Predicted vs Exact Optimum")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Computation time comparison (aggregated by size)
    # Use aggregated bars + speedup line for readability.
    _plot_time_comparison_by_size(
        ax=axes[1],
        results_df=results_df,
        baseline_col="exact_time",
        baseline_label="Exact Solver",
    )
    
    fig.suptitle("Inference Results vs Exact Optimum")
    plt.tight_layout()
    
    return fig


def plot_inference_results_vs_dual(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (11, 4.5),
) -> plt.Figure:
    """
    Plot inference results comparing predicted solution to the JV algorithm baseline.
    
    Args:
        results_df: DataFrame from run_inference() with columns:
                    size, predicted, dual_bound,
                    and optionally solution_time and dual_time
    
    Returns:
        Matplotlib figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Predicted vs JV Algorithm (scatter)
    for s in sorted(results_df["size"].unique()):
        sub = results_df[results_df["size"] == s]
        axes[0].scatter(sub["dual_bound"], sub["predicted"], label=s, alpha=0.6, s=40)
    
    mx = max(results_df["dual_bound"].max(), results_df["predicted"].max())
    axes[0].plot([0, mx], [0, mx], "k--", alpha=0.5, linewidth=2, label="Perfect")
    axes[0].set(xlabel="JV Algorithm Cost", ylabel="Predicted Cost", 
                title="Predicted vs JV Algorithm")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # 2. Computation time comparison (aggregated by size)
    _plot_time_comparison_by_size(
        ax=axes[1],
        results_df=results_df,
        baseline_col="dual_time",
        baseline_label="JV Algorithm",
    )
    
    fig.suptitle("Inference Results vs JV Algorithm")
    plt.tight_layout()
    
    return fig


def plot_facility_count_prediction(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (5.5, 4.5),
) -> plt.Figure:
    """Plot only the predicted-vs-target facility count scatter."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not {"n_fac_target", "n_fac_opened"}.issubset(results_df.columns):
        ax.text(
            0.5,
            0.5,
            "Facility count data not available\n(required: n_fac_target, n_fac_opened)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Facility Count Prediction")
        ax.axis("off")
        return fig

    ax.scatter(
        results_df["n_fac_target"],
        results_df["n_fac_opened"],
        alpha=0.65,
        s=40,
        color="steelblue",
    )
    mf = max(results_df["n_fac_target"].max(), results_df["n_fac_opened"].max()) + 1
    ax.plot([0, mf], [0, mf], "k--", alpha=0.5, linewidth=2)
    ax.set(
        xlabel="Target # Facilities",
        ylabel="Predicted # Facilities",
        title="Facility Count Prediction",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_time_comparison_by_size(
    ax: "plt.Axes",
    results_df: pd.DataFrame,
    baseline_col: str,
    baseline_label: str,
    model_col: str = "solution_time",
) -> None:
    """Plot readable runtime comparison aggregated by size with speedup."""
    required_cols = {"size", model_col, baseline_col}
    if not required_cols.issubset(results_df.columns):
        ax.text(
            0.5,
            0.5,
            f"Computation time data not available\n(required: {baseline_col}, {model_col})",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Computation Time Comparison")
        ax.axis("off")
        return

    df = results_df[["size", model_col, baseline_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        ax.text(0.5, 0.5, "No valid timing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Computation Time Comparison")
        ax.axis("off")
        return

    summary = df.groupby("size", sort=False).agg(
        model_mean=(model_col, "mean"),
        model_std=(model_col, "std"),
        base_mean=(baseline_col, "mean"),
        base_std=(baseline_col, "std"),
    )

    # Use geometric size ordering if possible (e.g. 10x10, 20x20, ...).
    def _size_key(size_label: str):
        parts = str(size_label).split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            n1, n2 = int(parts[0]), int(parts[1])
            return (n1 * n2, n1, n2)
        return (10**18, str(size_label), "")

    order = sorted(summary.index, key=_size_key)
    summary = summary.loc[order]

    for col in ("model_std", "base_std"):
        summary[col] = summary[col].fillna(0.0)

    x = np.arange(len(summary))
    width = 0.38

    ax.bar(
        x - width / 2,
        summary["base_mean"],
        width,
        yerr=summary["base_std"],
        capsize=3,
        color="lightgray",
        edgecolor="gray",
        alpha=0.95,
        label=f"{baseline_label} mean",
    )
    ax.bar(
        x + width / 2,
        summary["model_mean"],
        width,
        yerr=summary["model_std"],
        capsize=3,
        color="teal",
        edgecolor="black",
        alpha=0.85,
        label="Model mean",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Size")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title("Computation Time by Size")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    speedup = summary["base_mean"] / np.maximum(summary["model_mean"], 1e-12)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        speedup,
        color="darkred",
        marker="o",
        linewidth=1.7,
        markersize=4,
        label="Speedup (baseline/model)",
    )
    ax2.axhline(1.0, color="darkred", linestyle="--", alpha=0.45)
    ax2.set_ylabel("Speedup (x)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")


def plot_results_summary(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot a summary of key metrics by problem size.
    
    Args:
        results_df: DataFrame from run_inference()
    
    Returns:
        Matplotlib figure
    """
    summary = results_df.groupby("size").agg(
        opt_ratio_mean=("opt_ratio", "mean"),
        opt_ratio_std=("opt_ratio", "std"),
        gap_pct_mean=("opt_gap_pct", "mean"),
        gap_pct_median=("opt_gap_pct", "median"),
        count=("opt_ratio", "count"),
    ).round(4)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(summary))
    width = 0.35
    
    ax.bar(x - width/2, summary["opt_ratio_mean"], width, 
           yerr=summary["opt_ratio_std"], label="Mean Opt Ratio", 
           capsize=5, alpha=0.8, color="steelblue")
    ax.bar(x + width/2, 1.0 + summary["gap_pct_mean"] / 100, width, 
           label="Mean Opt Ratio (from gap)", alpha=0.8, color="coral")
    
    ax.axhline(1.0, color="green", ls="--", alpha=0.5, linewidth=2, label="Optimal")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Optimality Ratio")
    ax.set_title("Solution Quality Summary by Size")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def inspect_solution(
    result: Dict,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Visualize a single solution: facility decisions, assignment residuals, and targets.
    
    Args:
        result: Single result dict from model.inference(), containing:
                - y_target, y_res, repaired_opened: facility-level
                - x_res, x_res_target, assignment: client-facility assignments
                - n_fac, n_cli: problem dimensions
                - optimum, pred_cost, opt_ratio: cost metrics
                - dual_bound, dual_ratio: dual bound metrics
    
    Returns:
        Matplotlib figure with 3 subplots
    """
    n_fac = result["n_fac"]
    n_cli = result["n_cli"]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # --- Subplot 1: Facility decisions ---
    ax = axes[0]
    x_pos = np.arange(n_fac)
    w = 0.35
    
    y_target = result["y_target"][:n_fac]
    y_res = result["y_res"][:n_fac]
    repaired_opened = result["repaired_opened"][:n_fac]
    
    ax.bar(x_pos - w/2, y_target, w, label="Target", alpha=0.7, color="orange")
    
    colors = ["green" if o else "blue" for o in repaired_opened]
    ax.bar(x_pos + w/2, y_res, w, label="y_res", alpha=0.7, color=colors)
    
    ax.axhline(1e-8, color="red", ls="--", alpha=0.5, linewidth=1, label="eps")
    ax.set(xlabel="Facility Index", ylabel="Value")
    
    n_opened = int(repaired_opened.sum())
    n_target = int((y_target > 0.5).sum())
    ax.set_title(f"Facility Decisions: Opened {n_opened} (target: {n_target})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    
    if n_fac <= 50:
        ax.set_xticks(x_pos[::max(1, n_fac // 20)])
    
    # --- Subplot 2: Assignment residuals (predicted) ---
    ax = axes[1]
    x_res_data = result["x_res"].reshape(n_cli, n_fac)
    vmin = float(x_res_data.min())
    vmax = float(x_res_data.max())
    
    # Use TwoSlopeNorm if values cross zero, else default
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(vmax, 1e-8))
    
    im = ax.imshow(x_res_data, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set(xlabel="Facility Index", ylabel="Client Index", title="Predicted Assignment Residuals")
    plt.colorbar(im, ax=ax, label="Residual")
    
    # --- Subplot 3: Target assignments ---
    ax = axes[2]
    x_res_target = result["x_res_target"]
    assignment_one_hot = torch.nn.functional.one_hot(
        torch.from_numpy(x_res_target).long(), 
        num_classes=n_fac
    ).numpy().astype(float)
    
    ax.imshow(assignment_one_hot, aspect="auto", cmap="coolwarm")
    ax.set(xlabel="Facility Index", ylabel="Client Index", title="Target Assignments")
    ax.grid(True, alpha=0.3)
    
    # Add title with performance metrics
    gap_pct = (result["opt_ratio"] - 1.0) * 100
    dual_gap = (result["dual_ratio"] - 1.0) * 100
    title_str = (
        f"Gap: {gap_pct:.2f}% | "
        f"Pred: {result['pred_cost']:.4f} | "
        f"Opt: {result['optimum']:.4f} | "
        f"Dual: {result['dual_bound']:.4f}"
    )
    fig.suptitle(title_str, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    return fig


def inspect_solutions_batch(
    results: List[Dict],
    n_inspect: int = 5,
    sort_by: str = "opt_ratio",
) -> None:
    """
    Inspect multiple solutions, showing best and worst by specified metric.
    
    Args:
        results: List of result dicts from model.inference()
        n_inspect: Number of best and worst to visualize
        sort_by: Metric to sort by ("opt_ratio", "dual_ratio", "opt_gap_pct", etc.)
    """
    sorted_results = sorted(results, key=lambda r: r.get(sort_by, 0))
    
    print("=" * 70)
    print(f"BEST predictions (sorted by {sort_by})")
    print("=" * 70)
    for r in sorted_results[:n_inspect]:
        gap_pct = (r["opt_ratio"] - 1) * 100
        print(f"  Gap: {gap_pct:7.2f}% | Pred: {r['pred_cost']:8.4f} | "
              f"Opt: {r['optimum']:8.4f} | Dual: {r['dual_bound']:8.4f}")
        inspect_solution(r)
        plt.show()
    
    print("\n" + "=" * 70)
    print(f"WORST predictions (sorted by {sort_by})")
    print("=" * 70)
    for r in sorted_results[-n_inspect:]:
        gap_pct = (r["opt_ratio"] - 1) * 100
        print(f"  Gap: {gap_pct:7.2f}% | Pred: {r['pred_cost']:8.4f} | "
              f"Opt: {r['optimum']:8.4f} | Dual: {r['dual_bound']:8.4f}")
        inspect_solution(r)
        plt.show()


def plot_size_scalability(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Study scalability: how performance changes with problem size.
    
    Args:
        results_df: DataFrame from run_inference()
    
    Returns:
        Matplotlib figure
    """
    # Extract numeric sizes from size string (e.g., "10x10" -> 100 nodes)
    results_df = results_df.copy()
    results_df["n_nodes"] = results_df["n_fac"] + results_df["n_cli"]
    results_df["problem_scale"] = results_df["n_fac"] * results_df["n_cli"]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Gap vs problem scale
    ax = axes[0]
    for size in sorted(results_df["size"].unique()):
        sub = results_df[results_df["size"] == size]
        ax.scatter(sub["problem_scale"], sub["opt_gap_pct"], 
                  label=size, alpha=0.6, s=50)
    
    ax.set(xlabel="Problem Scale (n_fac × n_cli)", ylabel="Optimality Gap (%)",
           title="Solution Quality vs Problem Scale (scatter)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean gap trend
    ax = axes[1]
    size_summary = results_df.groupby("size").agg({
        "problem_scale": "first",
        "opt_gap_pct": ["mean", "std"]
    }).sort_values(("problem_scale", "first"))
    
    x = range(len(size_summary))
    means = size_summary[("opt_gap_pct", "mean")].values
    stds = size_summary[("opt_gap_pct", "std")].values
    
    ax.errorbar(x, means, yerr=stds, marker="o", capsize=5, linewidth=2, 
               markersize=8, label="Mean ± Std")
    ax.axhline(0, color="green", ls="--", alpha=0.5, linewidth=2, label="Optimal")
    ax.set(xlabel="Size Index (sorted by scale)", ylabel="Optimality Gap (%)",
           title="Scalability Trend")
    ax.set_xticks(x)
    ax.set_xticklabels(size_summary.index, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig
