import os.path as op
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import re


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

# Plot the Brain
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

'''

print("\n--- Generating fMRI Surface RDM ---")

# You already loaded fmri_feature_matrix (Shape: 180 x 59412) from the CIFTI file.
# The M/EEG only has 90 trials. We slice the first 90 rows to perfectly align the datasets chronologically.
fmri_features_90 = fmri_feature_matrix[:90, :]

print(f"Aligned fMRI Feature Shape: {fmri_features_90.shape} (90 Trials x 59,412 Vertices)")

print("\nCalculating 90x90 fMRI Surface Correlation Matrix...")
# Calculate the Representational Dissimilarity Matrix across the 59,412 cortical vertices
spatial_rdm_vector = pdist(fmri_features_90, metric='correlation')
spatial_rdm_matrix = squareform(spatial_rdm_vector)

# Save the Spatial RDM to disk
output_rdm_file = '/home/idhuang/bhs2026/project_sophia/data/ds004488/sub-01_fmrisurface_rdm.npy'
np.save(output_rdm_file, spatial_rdm_matrix)
print(f" fMRI Surface RDM saved to: {output_rdm_file}")

# Plot the fMRI Spatial RDM
plt.figure(figsize=(10, 8))
sns.heatmap(spatial_rdm_matrix, cmap='magma', xticklabels=10, yticklabels=10)
plt.title("fMRI Single-Trial Spatial RDM (Cortical Vertices)")
plt.xlabel("Chronological Trial Number (1-90)")
plt.ylabel("Chronological Trial Number (1-90)")
plt.show()

'''

print("\n--- Generating ALIGNED fMRI Surface RDM ---")
meeg_labels = np.load('/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_labels.npy', allow_pickle=True)

# Helper functions for text cleaning
def normalize(text):
    return str(text).lower().replace(' ', '').replace('_', '').replace('-', '')

def extract_fmri_name(raw_string):
    match = re.search(r'^v_(.*?)_id_', raw_string)
    if match:
        return match.group(1)
    return raw_string

# Create a normalized whitelist from the M/EEG labels
meeg_whitelist = set([normalize(L) for L in meeg_labels])

# Extract and normalize the 180 fMRI names
# (fmri_labels is already loaded earlier in your script from the .txt file)
fmri_extracted_names = [extract_fmri_name(L) for L in fmri_labels]
fmri_normalized_names = np.array([normalize(L) for L in fmri_extracted_names])

# FILTERING: Find the exact indices of the 90 overlapping videos
# We iterate through the 180 fMRI names and save the index ONLY if it's in the whitelist
valid_indices = [i for i, name in enumerate(fmri_normalized_names) if name in meeg_whitelist]

print(f"Filtering fMRI data: Dropping extra videos, keeping exactly {len(valid_indices)} overlapping videos...")

# Slice the fMRI features and labels using ONLY the valid indices
# (fmri_feature_matrix is your 180 x 59412 array loaded from the CIFTI file)
fmri_features_filtered = fmri_feature_matrix[valid_indices, :]
fmri_labels_filtered = np.array(fmri_extracted_names)[valid_indices] 

# ALPHABETICAL SORTING (The Mathematical Lock)
print("Sorting the remaining 90 fMRI trials alphabetically to guarantee M/EEG alignment...")
sort_indices = np.argsort(fmri_labels_filtered)

fmri_features_sorted = fmri_features_filtered[sort_indices]
fmri_labels_sorted = fmri_labels_filtered[sort_indices]

# Generate the Representational Dissimilarity Matrix (RDM)
print(f"Calculating 90x90 fMRI Surface Correlation Matrix... (Matrix Shape: {fmri_features_sorted.shape})")
spatial_rdm_vector = pdist(fmri_features_sorted, metric='correlation')
spatial_rdm_matrix = squareform(spatial_rdm_vector)

# Save to disk 
output_rdm_file = '/home/idhuang/bhs2026/project_sophia/data/ds004488/sub-01_fmrisurface_rdm.npy'
np.save(output_rdm_file, spatial_rdm_matrix)
print(f" Aligned & Sorted fMRI Surface RDM saved to: {output_rdm_file}")

# Plot the finalized fMRI grid
plt.figure(figsize=(12, 10))
sns.heatmap(spatial_rdm_matrix, cmap='magma', 
            xticklabels=fmri_labels_sorted, yticklabels=fmri_labels_sorted)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title("fMRI Single-Trial Spatial RDM (Filtered & Alphabetically Aligned)")
plt.tight_layout()
plt.show()