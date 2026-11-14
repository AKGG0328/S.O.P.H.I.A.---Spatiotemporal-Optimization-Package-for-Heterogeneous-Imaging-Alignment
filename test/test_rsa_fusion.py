import numpy as np
import pytest
from sophia.rsa_fusion import (
    align_and_intersect_trials,
    compute_rdm,
    rsa_fusion,
)


def test_compute_rdm_properties():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(10, 32))
    rdm, rdm_vec = compute_rdm(features, metric="correlation")

    assert rdm.shape == (10, 10)
    assert rdm_vec.shape == (10 * 9 // 2,)
    assert np.allclose(np.diag(rdm), 0.0)
    assert np.allclose(rdm, rdm.T)


def test_align_and_intersect_trials_handles_duplicates_and_order():
    mat1 = np.array([
        [1.0, 0.0],   # archery (trial 1)
        [3.0, 2.0],   # archery (trial 2) -> mean should be [2.0, 1.0]
        [5.0, 5.0],   # swimming
        [9.0, 9.0],   # unmatched_only_in_1
    ])
    labels1 = ["Archery", " archery ", "SWIMMING", "unmatched_1"]

    mat2 = np.array([
        [10.0, 20.0, 30.0],  # swimming
        [40.0, 50.0, 60.0],  # archery
        [70.0, 80.0, 90.0],  # unmatched_only_in_2
    ])
    labels2 = ["swimming", "archery", "unmatched_2"]

    m1_a, m2_a, common = align_and_intersect_trials(mat1, labels1, mat2, labels2)

    assert common == ["archery", "swimming"]
    assert m1_a.shape == (2, 2)
    assert m2_a.shape == (2, 3)
    assert np.allclose(m1_a[0], [2.0, 1.0])
    assert np.allclose(m2_a[0], [40.0, 50.0, 60.0])


def test_align_and_intersect_trials_raises_on_disjoint():
    mat1 = np.ones((2, 4))
    mat2 = np.ones((2, 4))
    with pytest.raises(ValueError, match="Zero overlapping conditions"):
        align_and_intersect_trials(mat1, ["a", "b"], mat2, ["c", "d"])


def test_rsa_fusion_heterogeneous_dimensions():
    rng = np.random.default_rng(123)
    n_conditions = 15
    conditions = [f"action_{i:02d}" for i in range(n_conditions)]

    mat_meeg = rng.normal(size=(n_conditions, 64))
    random_projection = rng.normal(size=(64, 500))
    mat_fmri = mat_meeg @ random_projection

    perm = rng.permutation(n_conditions)
    mat_fmri_shuffled = mat_fmri[perm]
    labels_fmri_shuffled = [conditions[i] for i in perm]

    results = rsa_fusion(mat_fmri_shuffled, labels_fmri_shuffled, mat_meeg, conditions)

    assert results["n_conditions"] == n_conditions
    assert results["rdm_fmri"].shape == (n_conditions, n_conditions)
    assert results["rdm_meeg"].shape == (n_conditions, n_conditions)
    assert results["spearman_rho"] > 0.85
    assert results["p_value"] < 1e-4