"""
SOPHIA: Spatiotemporal Optimization Package for Heterogeneous Imaging Alignment.
"""

from sophia.fmri_processing import process_cifti_fmri
from sophia.meeg_processing import process_meeg
from sophia.rsa_fusion import (
    align_and_intersect_trials,
    compute_rdm,
    rsa_fusion,
)
from sophia.utils import find_run_file
from sophia.visualization import plot_rdm, plot_rsa_fusion

__version__ = "0.1.0"

__all__ = [
    "process_cifti_fmri",
    "process_meeg",
    "compute_rdm",
    "align_and_intersect_trials",
    "rsa_fusion",
    "find_run_file",
    "plot_rdm",
    "plot_rsa_fusion",
]