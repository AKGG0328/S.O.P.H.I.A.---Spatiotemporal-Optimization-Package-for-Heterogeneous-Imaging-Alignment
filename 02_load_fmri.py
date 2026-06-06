import os 
import glob 
from nilearn import image, plotting 
import matplotlib.pyplot as plt
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel


'''
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
        # title = 'Mean fMRI BOLD Image (sub-01)',
        display_mode = 'z',
        cut_coords = 5
    )

    plt.show()

'''

'''
#General Linear Model: predict and check
print("--- Initializing fMRI GLM Pipeline")
search_pattern = os.path.join(FMRI_ROOT, 'sub-01', '**', '*bold.nii*')
func_files = glob.glob(search_pattern, recursive=True)

for func_path in func_files:
    # build the exact name of the matching events file by replacing '_bold.nii.gz' with '_events.tsv'
    event_path = func_path.replace('_bold.nii.gz', '_events.tsv')
    
    if not os.path.exists(event_path):
        print(f"WARNING: No matching event file found for {func_path}. Skipping...")
        continue
        
    print(f"Loading Functional Data: {os.path.basename(func_path)}")
    print(f"Loading Event Timestamps: {os.path.basename(event_path)}")
    break #one test 

fmri_img = image.load_img(func_path)
events = pd.read_csv(event_path, sep='\t')

#force trial type 
clean_events = events[['onset', 'duration', 'trial_type']].copy()
clean_events['trial_type'] = 'video_on'

#build GLM 
#Hemodynamic Response Fucntion
print("\nBuilding the Design Matrix and HRF Prdiction")
fmri_glm = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    minimize_memory=False
)


print("Running GLM equation on 4D brain matrix. This might take a moment...")
fmri_glm = fmri_glm.fit(fmri_img, clean_events)

print("Extracting 3D map of 'Video On' network...")
z_map = fmri_glm.compute_contrast('video_on', output_type='z_score')

print("Plotting the active spatial networks")
plotting.plot_glass_brain(
    z_map,
    threshold=3.0,
    display_mode='lyrz',
    plot_abs=False,
    cmap='cold_hot'
)
plt.show()

'''

#fmriprep standard pipeline
print("--- Initializing fMRIPrep GLM Pipeline ---")
BIDS_ROOT = '/home/idhuang/bhs2026/project_sophia/data/ds004488'
PREP_ROOT = os.path.join(BIDS_ROOT, 'derivatives', 'fmriprep')

print("Hunting for fMRIPrep outputs...")
func_pattern = os.path.join(PREP_ROOT, 'sub-01', '**', '*run-1_desc-preproc_bold.nii.gz')
func_files = glob.glob(func_pattern, recursive=True)

confound_pattern = os.path.join(PREP_ROOT, 'sub-01', '**', '*run-1_desc-confounds_timeseries.tsv')
confound_files = glob.glob(confound_pattern, recursive=True)

events_files = glob.glob(os.path.join(BIDS_ROOT,  'sub-01', '**', '*run-01_events.tsv'), recursive=True)

if not func_files or not confound_files:
    raise FileNotFoundError("Could not find fMRIPrep functional or confound files.")

func_file = func_files[0]
confound_file = confound_files[0]
events_file = events_files[0]

print(f" Loading Preprocessed BOLD: {os.path.basename(func_file)}")
fmri_img = image.load_img(func_file)

print(f" Loading Events: {os.path.basename(events_file)}")
events = pd.read_csv(events_file, sep='\t')
#print("Available trial types in this run:", events['trial_type'].unique())
clean_events = events[['onset', 'duration', 'trial_type']].copy()
clean_events['trial_type'] = 'video_on'


print(f" Loading Head Motion Confounds: {os.path.basename(confound_file)}")
confound_all = pd.read_csv(confound_file, sep='\t')
 
basic_motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
confound_clean = confound_all[basic_motion].fillna(0)

print("\nBuilding the Design Matrix and HRF Prdiction")
fmri_glm = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    minimize_memory=False
)


print("Running GLM equation...")
fmri_glm = fmri_glm.fit(fmri_img, clean_events, confounds=confound_clean)

print("Extracting 3D map of 'Video On' network...")
z_map = fmri_glm.compute_contrast('video_on', output_type='z_score')

print("Plotting the active spatial networks")
plotting.plot_glass_brain(
    z_map,
    threshold=4.0,
    display_mode='lyrz',
    plot_abs=False,
    cmap='cold_hot'
)
plt.show()