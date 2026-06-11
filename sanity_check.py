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


# epoch check line
import mne

# 1. Point to the exact file you just saved
saved_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_task-action-epo.fif'

print(f"Loading saved epochs from: {saved_file}")
# preload=False is a nice trick here; it reads the headers and metadata instantly 
# without loading the heavy brainwaves into RAM.
test_epochs = mne.read_epochs(saved_file, preload=False, verbose=False)

# 2. The Moment of Truth
print("\n--- Inspecting Saved Metadata ---")

if test_epochs.metadata is not None:
    print(" SUCCESS: Metadata was successfully loaded from disk!\n")
    
    print("--- First 3 Rows of the Dataframe ---")
    print(test_epochs.metadata[['event_name', 'session', 'run', 'class_name']].head(3))
    
    print(f"\n--- Total Epochs: {len(test_epochs)} ---")
    
    print("\n--- Unique Video Labels Found ---")
    print(test_epochs.metadata['class_name'].value_counts())
else:
    print(" WARNING: The metadata is missing. The binding did not save.")