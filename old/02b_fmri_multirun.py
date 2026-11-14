import os 
import glob 
from nilearn import  plotting 
import matplotlib.pyplot as plt
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel

print("--- Initializing Multi-Run fMRIP Pipeline ---")
BIDS_ROOT = '/home/idhuang/bhs2026/project_sophia/data/ds004488'
PREP_ROOT = os.path.join(BIDS_ROOT, 'derivatives', 'fmriprep')

func_files = sorted(glob.glob(os.path.join(PREP_ROOT, 'sub-01', '**', '*desc-preproc_bold.nii*'), recursive=True))
confound_files = sorted(glob.glob(os.path.join(PREP_ROOT, 'sub-01', '**', '*desc-confounds_timeseries.tsv'), recursive=True))
events_files = sorted(glob.glob(os.path.join(BIDS_ROOT, 'sub-01', '**', '*events.tsv'), recursive=True))

total_runs = len(func_files)
print(f"Found {total_runs} brain scans, {len(confound_files)} confound files, and {len(events_files)} event files.")

if not (total_runs == len(confound_files) == len(events_files)) or total_runs == 0:
    raise ValueError(" File counts do not match or are missing! The pipeline cannot synchronize the runs.")

# 2. Build the lists for the GLM
func_img_list = []
events_list = []
confounds_list = []
basic_motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

print("\nProcessing Data Arrays...")
for i in range(total_runs):
    print(f"  -> Prepping Run {i+1} of {total_runs}")
    
    func_img_list.append(func_files[i]) 
    
    events_raw = pd.read_csv(events_files[i], sep='\t')
    clean_events = events_raw[['onset', 'duration', 'trial_type']].copy()
    clean_events['trial_type'] = 'video_on' 
    events_list.append(clean_events)
    
    conf = pd.read_csv(confound_files[i], sep='\t')
    confounds_list.append(conf[basic_motion].fillna(0))

# 3. GLM
print("\nBuilding the Multi-Run Design Matrix...")
fmri_glm = FirstLevelModel(
    t_r=2.0, 
    noise_model='ar1', 
    standardize=False, 
    hrf_model='spm',
    minimize_memory=False
)

# Bypass lists stitches all runs into massive timeline
print(f"Running GLM math across all {total_runs} runs. This will take a few minutes...")
fmri_glm = fmri_glm.fit(func_img_list, events_list, confounds=confounds_list)

# 4. Extract the Definitive Map
print("Extracting definitively highly-powered spatial network...")
z_map = fmri_glm.compute_contrast('video_on', output_type='z_score')

# 5. Plot the Final Results
print("Plotting the active spatial networks.")
plotting.plot_glass_brain(
    z_map, 
    threshold=5.0, 
    display_mode='lyrz', 
    plot_abs=False,
    title=f'Definitive Action Perception Network ({total_runs} Runs)',
    cmap='cold_hot'
)
plt.show()