"""
Representational Similarity Analysis (RSA) Fusion Engine.
Aligns heterogeneous feature matrices across trials and computes second-order RDM correlations.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def compute_rdm(feature_matrix, metric="correlation"):
    """
    Computes a symmetric Representational Dissimilarity Matrix (RDM).
    
    Parameters:
    - feature_matrix (np.ndarray): Shape (n_trials, n_features)
    - metric (str): Distance metric (default: 'correlation' -> 1 - Pearson r)
    
    Returns:
    - rdm (np.ndarray): Shape (n_trials, n_trials)
    - rdm_vec (np.ndarray): Condensed upper-triangular distance vector
    """
    rdm_vec = pdist(feature_matrix, metric=metric)
    rdm = squareform(rdm_vec)
    return rdm, rdm_vec


def align_and_intersect_trials(mat1, labels1, mat2, labels2):
    """
    Finds common condition labels between two modalities and aligns matrix rows.
    
    Parameters:
    - mat1 (np.ndarray): (N1, D1) array
    - labels1 (list): List of N1 condition strings
    - mat2 (np.ndarray): (N2, D2) array
    - labels2 (list): List of N2 condition strings
    
    Returns:
    - mat1_aligned, mat2_aligned, common_labels
    """
    # Canonical label normalization (lowercase, stripped) for robust matching
    norm1 = [str(l).strip().lower() for l in labels1]
    norm2 = [str(l).strip().lower() for l in labels2]

    # Find mutual conditions preserving consistent deterministic order
    common_labels = sorted(list(set(norm1) & set(norm2)))
    
    if not common_labels:
        raise ValueError("Zero overlapping conditions found between modalities.")

    # Find row indices corresponding to the common conditions
    arr1, arr2 = np.array(norm1), np.array(norm2)
    mat1_aligned = np.vstack([mat1[arr1 == lbl, :].mean(axis=0) for lbl in common_labels])
    mat2_aligned = np.vstack([mat2[arr2 == lbl, :].mean(axis=0) for lbl in common_labels])

    return mat1_aligned, mat2_aligned, common_labels


def rsa_fusion(mat_fmri, labels_fmri, mat_meeg, labels_meeg, metric="correlation"):
    """
    Performs full cross-modal RSA fusion.
    
    Returns:
    - results (dict): Dictionary with Spearman rho, p-value, aligned RDMs, and labels.
    """
    # 1. Align common trials
    mat_fmri_a, mat_meeg_a, common_labels = align_and_intersect_trials(
        mat_fmri, labels_fmri, mat_meeg, labels_meeg
    )

    n_common = len(common_labels)
    print(f"Aligned {n_common} mutual condition trials across fMRI and M/EEG.")

    # 2. Compute 1st-order RDMs (1 - Pearson correlation)
    rdm_fmri, vec_fmri = compute_rdm(mat_fmri_a, metric=metric)
    rdm_meeg, vec_meeg = compute_rdm(mat_meeg_a, metric=metric)

    # 3. Compute 2nd-order RSA metric (Spearman rank correlation)
    rho, p_val = spearmanr(vec_fmri, vec_meeg)

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "n_conditions": n_common,
        "conditions": common_labels,
        "rdm_fmri": rdm_fmri,
        "rdm_meeg": rdm_meeg,
        "vec_fmri": vec_fmri, 
        "vec_meeg": vec_meeg,
    }