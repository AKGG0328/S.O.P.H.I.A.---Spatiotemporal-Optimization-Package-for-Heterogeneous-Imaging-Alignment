import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Initializing  RSA Fusion ---")

# load the two matrices you built
meeg_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_rdm.npy'
fmri_file = '/home/idhuang/bhs2026/project_sophia/data/ds004488/sub-01_fmrisurface_rdm.npy'

print("Loading M/EEG Temporal RDM...")
meeg_rdm = np.load(meeg_file)

print("Loading fMRI Spatial RDM...")
fmri_rdm = np.load(fmri_file)

# Extract the lower triangle
# We only want to compare the unique combinations (Trial 1 vs Trial 2). 
# We ignore the duplicate upper triangle and the 0.0 diagonal.
indices = np.tril_indices(90, k=-1)

meeg_distances = meeg_rdm[indices]
fmri_distances = fmri_rdm[indices]

print(f"Extracted {len(meeg_distances)} unique pairwise comparisons from both modalities.")

# Calculate Spearman Rank Correlation
# We use Spearman because we care about the *rank order* of the distances, 
# not the absolute linear scaling between M/EEG and fMRI physics.
print("\nCalculating Spearman Rank Correlation...")
correlation, p_value = spearmanr(meeg_distances, fmri_distances)

print("\n==========================================")
print(f" RSA Spearman Correlation (rho): {correlation:.4f}")
print(f" Statistical Significance (p-value): {p_value:.4e}")
print("==========================================\n")

# Plot the final RSA Scatter Plot
plt.figure(figsize=(8, 6))
sns.regplot(
    x=meeg_distances, 
    y=fmri_distances, 
    scatter_kws={'alpha': 0.3, 's': 15, 'color': 'teal'},
    line_kws={'color': 'red', 'linewidth': 2}
)

plt.title(f"Project SOPHIA: Representational Similarity Analysis\nrho = {correlation:.4f}, p = {p_value:.4e}")
plt.xlabel("M/EEG Temporal Dissimilarity (Mu/Beta ERD)")
plt.ylabel("fMRI Spatial Dissimilarity (Cortical Vertices)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()