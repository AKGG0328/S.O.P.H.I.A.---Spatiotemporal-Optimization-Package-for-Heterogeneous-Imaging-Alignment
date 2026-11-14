"""
SOPHIA Quickstart Example (Synthetic Multimodal Dataset)
Demonstrates end-to-end RDM computation, cross-modal trial alignment,
second-order RSA fusion, and publication-quality visualization without
requiring external BIDS downloads.
"""
from pathlib import Path
import numpy as np

from sophia import (
    compute_rdm,
    plot_rdm,
    plot_rsa_fusion,
    rsa_fusion,
)


def main():
    rng = np.random.default_rng(2026)
    output_dir = Path("results/quickstart_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define 12 simulated experimental action categories across 2 runs
    base_actions = [
        "archery", "basketball", "bowling", "climbing",
        "cycling", "fencing", "golf", "gymnastics",
        "rowing", "skating", "swimming", "tennis",
    ]
    n_actions = len(base_actions)

    # Latent representational geometry shared across modalities
    latent_patterns = rng.normal(size=(n_actions, 24))

    # 2. Simulate M/EEG spectral features (64 channels x 2 bands = 128 features)
    # Include 2 trials per action with shuffled order to mimic multi-run BIDS data
    meeg_labels = base_actions * 2
    meeg_proj = rng.normal(size=(24, 128))
    mat_meeg = np.vstack([latent_patterns, latent_patterns]) @ meeg_proj
    mat_meeg += rng.normal(scale=0.6, size=mat_meeg.shape)

    meeg_perm = rng.permutation(len(meeg_labels))
    mat_meeg = mat_meeg[meeg_perm]
    meeg_labels = [meeg_labels[i] for i in meeg_perm]

    # 3. Simulate fMRI CIFTI surface beta series (91,282 grayordinates -> 2,000 ROIs)
    fmri_labels = base_actions * 2
    fmri_proj = rng.normal(size=(24, 2000))
    mat_fmri = np.vstack([latent_patterns, latent_patterns]) @ fmri_proj
    mat_fmri += rng.normal(scale=0.8, size=mat_fmri.shape)

    fmri_perm = rng.permutation(len(fmri_labels))
    mat_fmri = mat_fmri[fmri_perm]
    fmri_labels = [fmri_labels[i] for i in fmri_perm]

    # 4. Run SOPHIA Cross-Modal Alignment & Second-Order RSA Fusion
    results = rsa_fusion(
        mat_fmri=mat_fmri,
        labels_fmri=fmri_labels,
        mat_meeg=mat_meeg,
        labels_meeg=meeg_labels,
        metric="correlation",
    )

    print(f"Aligned {results['n_conditions']} mutual action categories.")
    print(f"Cross-Modal Spearman rho : {results['spearman_rho']:.4f}")
    print(f"Significance p-value     : {results['p_value']:.4e}")

    # 5. Generate Publication Plots
    plot_rdm(
        rdm=results["rdm_meeg"],
        conditions=results["conditions"],
        title="Simulated M/EEG RDM (Mu/Beta ERD)",
        save_path=output_dir / "quickstart_rdm_meeg.png",
    )
    plot_rdm(
        rdm=results["rdm_fmri"],
        conditions=results["conditions"],
        title="Simulated fMRI RDM (Cortical Grayordinates)",
        save_path=output_dir / "quickstart_rdm_fmri.png",
    )
    plot_rsa_fusion(
        vec_fmri=results["vec_fmri"],
        vec_meeg=results["vec_meeg"],
        rho=results["spearman_rho"],
        p_val=results["p_value"],
        title="SOPHIA Quickstart: Cross-Modal RSA Fusion",
        save_path=output_dir / "quickstart_rsa_fusion.png",
    )
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()