import os 
import glob 
from nilearn import image, plotting 
import matplotlib.pyplot as plt

print("--- Initializing fMRI Environment ---")

FMRI_ROOT = '/home/idhuang/bhs2026/project_sophia/data/ds004488'

print("Searching for functional NIfTI files...")
search_pattern = os.path.join(FMRI_ROOT, 'sub-01', '**', '*bold.nii*')
func_files = glob.glob(search_pattern, recursive = True)
if not func_files:
    print("Error: Could not find any BOLD fMRI files. Check your folder structure.")
else:
    func_file = func_files[0] #aghh fucking typo
    print(f" Found fMRI Cargo: {os.path.basename(func_file)}")

    print("Loading heavy 4D data into RAM...")
    fmri_image = image.load_img(func_file)

    print(f"Image Matrix Shape: {fmri_image.shape}")

    print("\nCalculating mean anatomical image for visual verification... ")
    mean_img = image.mean_img(fmri_image)

    print("Plotting axial slices")
    plotting.plot_epi(
        mean_img,
        title = 'Mean fMRI BOLD Image (sub-01)',
        display_mode = 'z',
        cut_coords = 5
    )

    plt.show()