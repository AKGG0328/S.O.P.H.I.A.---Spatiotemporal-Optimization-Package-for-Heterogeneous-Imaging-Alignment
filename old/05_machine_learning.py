import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

print("--- Initializing  M/EEG Single-Trial RDM ---")

# Load  Data
feature_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_features.npy'
label_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_labels.npy' # Optional: if you saved y to a file

print(f"Loading M/EEG feature matrix from: {feature_file}...")
X = np.load(feature_file)
labels = np.load(label_file, allow_pickle=True)

# The shape should be (90, 124)
print(f"Feature Matrix Shape: {X.shape}") 

print("\nSorting trials alphabetically to guarantee fMRI alignment...")
# np.argsort finds the exact row indices needed to put the labels in A-Z order
sort_indices = np.argsort(labels)

# Reorder both the features and the labels simultaneously using the exact same indices
X_sorted = X[sort_indices]
labels_sorted = labels[sort_indices]

# Generate the Representational Dissimilarity Matrix (RDM)
print("\nCalculating 90x90 M/EEG Correlation Matrix...")

# pdist calculates the exact 1D correlation distance between every single trial's brain state
# squareform folds that 1D list into the final 90x90 grid
rdm_vector = pdist(X_sorted, metric='correlation')
meeg_rdm_matrix = squareform(rdm_vector)

output_rdm_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_rdm.npy'
np.save(output_rdm_file, meeg_rdm_matrix)
print(f" M/EEG Temporal RDM saved to: {output_rdm_file}")

#  Plot the RDM to visually inspect the data
plt.figure(figsize=(10, 8))
# We show every 10th label on the axes to keep the plot readable
sns.heatmap(meeg_rdm_matrix, cmap='viridis', xticklabels=labels_sorted, yticklabels=labels_sorted,)

plt.title("M/EEG Single-Trial Temporal RDM (Mu/Beta ERD)")
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()