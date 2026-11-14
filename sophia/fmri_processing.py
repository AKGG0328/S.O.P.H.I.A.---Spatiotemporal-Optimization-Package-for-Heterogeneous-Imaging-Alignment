import nibabel as nib
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
import numpy as np
from pathlib import Path
from sophia.utils import find_run_file


def process_cifti_fmri(bids_root, dataset_dir, subject_id, session="action01", 
                       task_name="action", run="1", tr=2.0, 
                       target_event="video on", condition_col="stim_file"):
    
    print(f"\n--- Starting fMRI Beta-Series (LSA) for sub-{subject_id}, run-{run} ---")
    
    # 1. FILE PATHS 
    bids_path = Path(bids_root) / dataset_dir

    run_str = f"{int(run):02d}"
    out_dir = bids_path / "derivatives" / "sophia" / f"sub-{subject_id}" / f"ses-{session}" / "func"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_out = out_dir / f"sub-{subject_id}_ses-{session}_task-{task_name}_run-{run_str}_betas.npy"
    labels_out = out_dir / f"sub-{subject_id}_ses-{session}_task-{task_name}_run-{run_str}_conditions.tsv"

    if matrix_out.exists() and labels_out.exists():
        print(f"SUCCESS: Found existing fMRI betas at {matrix_out.name}. Bypassing GLM fitting...")
        return matrix_out, labels_out

    cifti_dir = bids_path / "derivatives" / "ciftify" / f"sub-{subject_id}" / "results"
    func_dir = bids_path / f"sub-{subject_id}" / f"ses-{session}" / "func"

    cifti_file = find_run_file(
        directory=cifti_dir,
        prefix=f"ses-{session}_task-{task_name}",
        suffix="Atlas.dtseries.nii",
        run=run,
    )
    events_file = find_run_file(
        directory=func_dir,
        prefix=f"sub-{subject_id}_ses-{session}_task-{task_name}",
        suffix="events.tsv",
        run=run,
    )

    print(f"Found CIFTI : {cifti_file.name}")
    print(f"Found Events: {events_file.name}")

    # 2. LOAD CIFTI DATA
    print("Loading CIFTI surface data...")
    cifti_img = nib.load(cifti_file)
    fmri_data = cifti_img.get_fdata() 
    n_scans = fmri_data.shape[0]
    frame_times = np.arange(n_scans) * tr
    
    # 3. UNIVERSAL LSA EVENT MAPPING
    print(f"Parsing events: Extracting action labels from '{condition_col}'...")
    events = pd.read_csv(events_file, sep='\t')

    # Filter by target event if specified (e.g., isolate video presentation rows)
    if "event_type" in events.columns and target_event:
        events = events[events["event_type"] == target_event].copy()
    elif "trial_type" in events.columns and target_event in events[
        "trial_type"
    ].values:
        events = events[events["trial_type"] == target_event].copy()
    
    # 3a. Clean the nested strings (e.g., "Category/filename.mp4" -> "Category")
    # We use a lambda function to split by the slash and grab the first element
    clean_labels = events[condition_col].astype(str).apply(lambda x: x.split('/')[0])
    
    # 3b. Append the unique trial counter for LSA (e.g., "Category_1")
    lsa_labels = clean_labels + "_" + events.groupby(clean_labels).cumcount().add(1).astype(str)
    
    # 3c. Overwrite trial_type so Nilearn treats each as an independent GLM predictor
    events['trial_type'] = lsa_labels.values

    # Nilearn expects minimal columns: 'onset', 'duration', 'trial_type'
    glm_events = events[["onset", "duration", "trial_type"]]

    # 4. DESIGN MATRIX
    print("Building First-Level Design Matrix...")
    design_matrix = make_first_level_design_matrix(
        frame_times, 
        glm_events, 
        hrf_model='glover', 
        drift_model='cosine'
    )
    
    # 5. RUN GLM ON SURFACE
    print("Fitting GLM to surface grayordinates...")
    labels, results = run_glm(fmri_data, design_matrix.values, noise_model='ar1')
    
    # 6. EXTRACT AND SAVE BETA MAPS
    print("Extracting trial-specific Beta maps...")
    
    conditions = list(design_matrix.columns)
    # Filter out Nilearn's automatically generated cosine (drift) and constant regressors, keeping only video trials 
    trial_indices = [i for i, c in enumerate(conditions) if "cosine" not in c and c != "constant"]
    trial_conditions = [conditions[i] for i in trial_indices]
    
    # run_glm returns a dictionary of results keyed by spatial labels. 
    # For a unified surface mesh, all valid grayordinates fall under the first label.
    model = results[labels[0]] 
    all_betas = model.theta

    # Slice out only the trial regressors: shape -> (n_trials, n_grayordinates)
    beta_matrix = all_betas[trial_indices, :]

    np.save(matrix_out, beta_matrix)
    pd.DataFrame({"condition": trial_conditions}).to_csv(labels_out, sep="\t", index=False)

    print(
        f"Successfully saved beta matrix {beta_matrix.shape} to {matrix_out}"
    )
    return matrix_out, labels_out