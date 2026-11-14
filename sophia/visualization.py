"""
SOPHIA Visualization Module: RDM Heatmaps and 2nd-Order RSA Fusion Plots.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rdm(rdm, conditions, title, cmap="viridis", save_path=None, vmin=0.0, vmax=None):
    """
    Plots a single Representational Dissimilarity Matrix with condition tick labels.
    """
    if vmax is None:
        vmax = np.percentile(rdm, 99)

    # Format condition labels: clean underscores and trial counters if needed
    clean_labels = [c.replace("_", " ").title() for c in conditions]

    plt.figure(figsize=(10, 8), dpi=300)
    ax = sns.heatmap(
        rdm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=clean_labels,
        yticklabels=clean_labels,
    )

    # Styling ticks
    plt.title(title, fontsize=14, pad=15)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved RDM plot to: {save_path}")

    plt.close()


def plot_rsa_fusion(
    vec_fmri,
    vec_meeg,
    rho,
    p_val,
    title="Project SOPHIA: Representational Similarity Analysis",
    xlabel="M/EEG Temporal Dissimilarity (Mu/Beta ERD)",
    ylabel="fMRI Spatial Dissimilarity (Cortical Vertices)",
    save_path=None,
):
    """
    Plots the scatter plot comparing pairwise dissimilarities across modalities.
    Matches publication standard styling.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Regression plot with scatter points
    sns.regplot(
        x=vec_meeg,
        y=vec_fmri,
        scatter_kws={"color": "#208b8e", "alpha": 0.35, "s": 15},
        line_kws={"color": "red", "linewidth": 2.2},
    )

    plt.title(f"{title}\nrho = {rho:.4f}, p = {p_val:.4e}", fontsize=12, pad=12)
    plt.xlabel(xlabel, fontsize=10, labelpad=8)
    plt.ylabel(ylabel, fontsize=10, labelpad=8)

    # Gridlines matching reference
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved RSA Fusion plot to: {save_path}")

    plt.close()