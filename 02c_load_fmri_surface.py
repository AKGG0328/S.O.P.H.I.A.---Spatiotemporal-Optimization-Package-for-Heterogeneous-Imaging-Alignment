import os.path as op
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

print("Initializing fMRI Surface Loader...")

# path
fs_dir = "/home/idhuang/bhs2026/project_sophia/data/ds004488"
cifti_path = op.join(fs_dir, 'derivatives', 'ciftify', 'sub-01', 'results', 'ses-action01_task-action_cycle-1_beta.dscalar.nii')
label_path = op.join(fs_dir, 'derivatives', 'ciftify', 'sub-01', 'results', 'ses-action01_task-action_cycle-1_label.txt')

#  Load the CIFTI image into memory
print(f"Loading pre-calculated GLM betas from: {cifti_path}")
cifti_img = nib.load(cifti_path)

#  Extract the raw numerical matrix
# .get_fdata() strips away the file headers and just gives you the math
fmri_feature_matrix = cifti_img.get_fdata()

#  Load the event labels to prepare for RSA sorting
with open(label_path, 'r') as file:
    fmri_labels = [line.strip() for line in file.readlines()]

print("\n--- fMRI Extraction Complete ---")
print(f"fMRI Feature Matrix Shape: {fmri_feature_matrix.shape}")
print(f"Number of loaded labels: {len(fmri_labels)}")


# visual proof
print("\nPreparing 3D Surface Visualization for Trial 1...")

surface_dir = op.join(fs_dir, 'derivatives', 'ciftify', 'sub-01', 'standard_fsLR_surface')
left_mesh = op.join(surface_dir, 'sub-01.L.midthickness.32k_fs_LR.surf.gii')
right_mesh = op.join(surface_dir, 'sub-01.R.midthickness.32k_fs_LR.surf.gii')


cifti_img = nib.load(cifti_path)
bm_axis = cifti_img.header.get_axis(1)

# Helper Function to reconstruct the Medial Wall
def map_to_full_surface(compressed_array, hemisphere_name):
    # Create an empty array of 32,492 zeros (the exact size of the fsLR 32k mesh)
    full_surface = np.zeros(32492)
    
    # Find exactly which values belong to this hemisphere
    mask = (bm_axis.name == hemisphere_name)
    
    # Find exactly which vertices on the 32k mesh these values belong to
    vertex_indices = bm_axis.vertex[mask]
    
    # Map the compressed data directly onto the correct triangles
    full_surface[vertex_indices] = compressed_array[mask]
    return full_surface


# (180, 59412) just want the first row.
trial_1_data = fmri_feature_matrix[0, :]
trial_1_name = fmri_labels[0]

sulc_img = nib.load(op.join(surface_dir, 'sub-01.sulc.32k_fs_LR.dscalar.nii'))
sulc_array = sulc_img.get_fdata()[0]

# Apply the decoder ring to get perfect 32,492-length arrays
left_data  = map_to_full_surface(trial_1_data, 'CIFTI_STRUCTURE_CORTEX_LEFT')
right_data = map_to_full_surface(trial_1_data, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
left_sulc  = sulc_array[:32492]
right_sulc = sulc_array[32492:]

# 5. Plot the Brain
print(f"Plotting Activation for: {trial_1_name}")
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121, projection='3d')
plotting.plot_surf(
    left_mesh, left_data, axes=ax1, cmap='cold_hot', 
    threshold=15, bg_map=left_sulc, view='lateral', title=f"Left Hemisphere\n{trial_1_name}"
)

ax2 = fig.add_subplot(122, projection='3d')
plotting.plot_surf(
    right_mesh, right_data, axes=ax2, cmap='cold_hot', 
    threshold=15, bg_map=right_sulc, view='lateral', title=f"Right Hemisphere\n{trial_1_name}"
)

plt.show()