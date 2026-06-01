import os
import mne
import nilearn
from nilearn import image

# Define your BIDS root paths
MEEG_ROOT = os.path.expanduser('/home/idhuang/bhs2026/project_sophia/data/ds007353')
FMRI_ROOT = os.path.expanduser('/home/idhuang/bhs2026/project_sophia/data/ds004488')

print("--- Project SOPHIA Workspace Verification ---")

# 1. Verify Directories Exist
print(f"M/EEG Dataset Path Exists: {os.path.exists(MEEG_ROOT)}")
print(f"fMRI Dataset Path Exists: {os.path.exists(FMRI_ROOT)}")

# 2. Locate sub-01 files dynamically
# Note: BIDS data folders are divided into functional (func), anatomical (anat), or meg/eeg
print("\nChecking subject 01 folders...")
if os.path.exists(MEEG_ROOT):
    print("MEEG sub-01 contents:", os.listdir(os.path.join(MEEG_ROOT, 'sub-01')))
if os.path.exists(FMRI_ROOT):
    print("fMRI sub-01 contents:", os.listdir(os.path.join(FMRI_ROOT, 'sub-01')))

print("\nReady to begin preprocessing!")