import pytest
import sophia
from sophia.utils import find_run_file


def test_package_exports():
    assert hasattr(sophia, "__version__")
    expected = {
        "process_cifti_fmri",
        "process_meeg",
        "compute_rdm",
        "align_and_intersect_trials",
        "rsa_fusion",
        "find_run_file",
        "plot_rdm",
        "plot_rsa_fusion",
    }
    assert expected.issubset(set(sophia.__all__))


def test_find_run_file_padded_and_unpadded(tmp_path):
    padded_file = tmp_path / "sub-01_ses-action01_task-action_run-01_events.tsv"
    padded_file.touch()

    found_from_str = find_run_file(
        tmp_path, prefix="sub-01_ses-action01_task-action", suffix="events.tsv", run="1"
    )
    found_from_int = find_run_file(
        tmp_path, prefix="sub-01_ses-action01_task-action", suffix="events.tsv", run=1
    )
    assert found_from_str == padded_file
    assert found_from_int == padded_file

    unpadded_file = tmp_path / "ses-action01_task-action_run-2_Atlas.dtseries.nii"
    unpadded_file.touch()

    found_unpadded = find_run_file(
        tmp_path, prefix="ses-action01_task-action", suffix="Atlas.dtseries.nii", run="02"
    )
    assert found_unpadded == unpadded_file


def test_find_run_file_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_run_file(tmp_path, prefix="sub-99", suffix="events.tsv", run=1)