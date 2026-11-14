import mne
import numpy as np
import os 
import os.path as op


#bypass mri freesurfer process with fsaverage, mne built-in brain template


print("--- Initializing Source Localization ---")

# 1. Load the Epoched Data
epoch_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_task-action-epo.fif'
print(f"Loading epoched data: {epoch_file}")
epochs = mne.read_epochs(epoch_file, preload=True)
evoked = epochs.average()

# 2. Setup the 3D Head Model (Using the 'fsaverage' template bypass)
print("\nFetching standard 3D brain mesh (fsaverage)...")
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

# Set up the physical properties of fake head
subject = 'fsaverage'
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
#model = mne.make_bem_model(subject=subject, ico=4, subjects_dir=subjects_dir, conductivity=(0.3,))
#bem = mne.make_bem_solution(model)

bem_path = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
bem = mne.read_bem_solution(bem_path)

# 3. Calculate the Forward Model 
# real subject require a 'trans' file aligning the BIDS MEG sensors to the MRI.
# generate identity matrix here to force the template to work.
# fake trans
trans = mne.transforms.Transform('head', 'mri') 
print("\nCalculating the Forward Solution (Sensors -> Skull -> Cortex)...")
fwd = mne.make_forward_solution(evoked.info, trans=trans, src=src, bem=bem, eeg=True, meg=False, ignore_ref=True)

# --- meg 1 layer bem . eeg 3 layer bem ---
# => modality check (same as load_meeg)
# --- underconstruction ---


# 4. Calculate the Noise Covariance
print("Calculating Noise Covariance from baseline...")
noise_cov = mne.compute_covariance(epochs, tmax=0.0, method='shrunk', rank=None)

# 5. Build and Apply the Inverse Operator
print("\nBuilding the Inverse Operator...")
from mne.minimum_norm import make_inverse_operator, apply_inverse

inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)

print("Applying the Inverse Operator (Pushing waves into the 3D brain)...")
# We use dSPM (dynamic statistical parametric mapping) to turn raw voltage into statistical z-scores
stc = apply_inverse(evoked, inverse_operator, lambda2=1.0 / 9.0, method='dSPM')

print(f"\nSource Localization Complete.")
print(f"The 2D sensors have been mapped to {stc.data.shape[0]} 3D cortical locations!")

#mne.viz.set_3d_options(depth_peeling=False, antialias=False)

# 6. Plot the 3D Brain Activity
print("\nRendering 3D Cortex. Focus on the visual/motor peak at 0.150 seconds.")
brain = stc.plot(
    subject=subject, 
    subjects_dir=subjects_dir, 
    hemi='both', 
    views='lateral', 
    initial_time=0.150, 
    time_viewer=True
)

# Pause the script so the 3D window stays open
input("Press Enter in this terminal to close the 3D window and end the script...") 