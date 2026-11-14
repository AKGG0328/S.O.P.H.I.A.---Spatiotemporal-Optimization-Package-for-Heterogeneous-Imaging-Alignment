import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sophia.fmri_processing import process_cifti_fmri
from sophia.meeg_processing import process_meeg
from sophia.visualization import plot_rdm, plot_rsa_fusion


def test_fmri_processing_cache_bypass(tmp_path):
    dataset_dir = "ds_mock_fmri"
    out_dir = (
        tmp_path / dataset_dir / "derivatives" / "sophia"
        / "sub-01" / "ses-action01" / "func"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_mat = out_dir / "sub-01_ses-action01_task-action_run-01_betas.npy"
    cached_lbl = out_dir / "sub-01_ses-action01_task-action_run-01_conditions.tsv"
    np.save(cached_mat, np.zeros((5, 100)))
    pd.DataFrame({"condition": [f"act_{i}" for i in range(5)]}).to_csv(
        cached_lbl, sep="\t", index=False
    )

    mat_path, lbl_path = process_cifti_fmri(
        bids_root=tmp_path,
        dataset_dir=dataset_dir,
        subject_id="01",
        session="action01",
        task_name="action",
        run="1",
    )
    assert mat_path == cached_mat
    assert lbl_path == cached_lbl


def test_meeg_processing_cache_bypass(tmp_path):
    dataset_dir = "ds_mock_meeg"
    out_dir = (
        tmp_path / dataset_dir / "derivatives" / "sophia"
        / "sub-01" / "ses-eeg" / "eeg"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_feat = out_dir / "sub-01_ses-eeg_task-action_run-01_features.npy"
    cached_lbl = out_dir / "sub-01_ses-eeg_task-action_run-01_conditions.tsv"
    np.save(cached_feat, np.zeros((5, 128)))
    pd.DataFrame({"condition": [f"act_{i}" for i in range(5)]}).to_csv(
        cached_lbl, sep="\t", index=False
    )

    feat_path, lbl_path = process_meeg(
        bids_root=tmp_path,
        dataset_dir=dataset_dir,
        subject_id="01",
        task_name="action",
        session="eeg",
        modality="eeg",
        run="01",
    )
    assert feat_path == cached_feat
    assert lbl_path == cached_lbl


def test_visualization_outputs(tmp_path):
    rdm = np.array([[0.0, 0.4, 0.8], [0.4, 0.0, 0.5], [0.8, 0.5, 0.0]])
    conditions = ["kickball_1", "swimming_1", "archery_1"]

    rdm_png = tmp_path / "rdm.png"
    plot_rdm(rdm, conditions, title="Test RDM", save_path=rdm_png)
    assert rdm_png.exists() and rdm_png.stat().st_size > 0

    vec = np.array([0.4, 0.8, 0.5])
    fusion_png = tmp_path / "fusion.png"
    plot_rsa_fusion(vec, vec * 0.9, rho=0.95, p_val=0.001, save_path=fusion_png)
    assert fusion_png.exists() and fusion_png.stat().st_size > 0