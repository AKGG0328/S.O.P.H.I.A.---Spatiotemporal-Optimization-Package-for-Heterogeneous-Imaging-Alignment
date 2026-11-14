import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore

def process_meeg(bids_root, dataset_dir, subject_id, task_name="action",   
                    session="eeg", modality="eeg", run="01",
                    l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.0, 
                    ica_components=15, ica_random_state=97, eog_ch='Fp1'
                    ,target_event=None, condition_col="class_name"):
    
    
    """
    Loads and preprocesses M/EEG data for a given subject.
    
    Parameters:
    - bids_root (str or Path): The main data folder (e.g., 'data/')
    - dataset_dir (str): The specific dataset folder (e.g., 'ds007353')
    - subject_id (str): The subject number (e.g., '01')
    - task_name (str): The BIDS task identifier
    
    Returns:
    - epochs (mne.Epochs): The processed epoch data, or None if failed.
    """
    print(f"\n--- Starting M/EEG processing for sub-{subject_id} , run-{run} ---")

    try:
        # 1. 
        bids_path_root = Path(bids_root) / dataset_dir
        run_int = int(run)
        run_str = f"{run_int:02d}" 

        # 2. Construct the derivative output path & BYPASS CHECK
        out_dir = (bids_path_root/ "derivatives"/ "sophia"/ f"sub-{subject_id}"/ f"ses-{session}"/ modality)
        out_dir.mkdir(parents=True, exist_ok=True)

        features_out = (out_dir/ f"sub-{subject_id}_ses-{session}_task-{task_name}_run-{run_str}_features.npy")
        labels_out = (out_dir/ f"sub-{subject_id}_ses-{session}_task-{task_name}_run-{run_str}_conditions.tsv")
        
        if features_out.exists() and labels_out.exists():
            print(f"SUCCESS: Found cached MEEG features at {features_out.name}.")
            return features_out, labels_out

        # 3. detail loading 
        detailed_events_file = (bids_path_root / "derivatives" / "detailed_events" / f"sub-{subject_id}_events.tsv")
        if detailed_events_file.exists():
            meta_df = pd.read_csv(detailed_events_file, sep="\t")
            run_mask = (meta_df["run"] == run_int) & (meta_df["session"].isin([session, modality]))
            run_meta = meta_df[run_mask].reset_index(drop=True)
            if run_meta.empty:
                run_meta = meta_df[meta_df["run"] == run_int].reset_index(drop=True)
        else:
            std_events = bids_path_root / f"sub-{subject_id}" / f"ses-{session}" / modality / f"sub-{subject_id}_ses-{session}_task-{task_name}_run-{run_str}_events.tsv"
            run_meta = pd.read_csv(std_events, sep="\t")
            if target_event and "trial_type" in run_meta.columns:
                run_meta = run_meta[run_meta["trial_type"].astype(str).str.lower() == target_event.lower()].reset_index(drop=True)
        print(f"Loaded {len(run_meta)} event conditions for run {run_int}.")

        # 4. Automatically handles .set, .fif, etc. using MNE-BIDS
        bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task_name,
        run=run_str,
        datatype=modality,
        root=bids_path_root,
        )
        print(f"Loading data from BIDS path: {bids_path.basename}...")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        
        # 5. Standard pipeline
        print(f"Filtering data from {l_freq} to {h_freq} Hz...")
        raw.filter(l_freq=l_freq, h_freq=h_freq)

        # automatic ICA for artifact rejection
        print("Fitting ICA for artifact rejection...")
        # n_components=15 is a standard, safe number for typical EEG setups
        ica = mne.preprocessing.ICA(n_components=ica_components, max_iter='auto', random_state=ica_random_state)
        ica.fit(raw)
        
        print("Automatically detecting EOG (blink) artifacts...")
        # If the dataset lacks a dedicated 'eog' channel type, we use Fp1 as a proxy
        target_eog = eog_ch if eog_ch in raw.ch_names else None
        if target_eog:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=target_eog)
            ica.exclude = eog_indices
        print(f"Dropping ICA components: {ica.exclude}")
        ica.apply(raw)
        
        print("Extracting events from BIDS annotations...")
        events, event_dict = mne.events_from_annotations(raw)

        if target_event:
            matched_keys = [k for k in event_dict.keys() if target_event.lower() in k.lower()]
            if not matched_keys:
                raise ValueError(
                    f"Target event '{target_event}' not found in annotations. Available:"
                    f" {list(event_dict.keys())}"
                )
            selected_event_id = {k: event_dict[k] for k in matched_keys}
            print(f"Filtering epochs for target event(s): {list(selected_event_id.keys())}")
        else:
            # Generic mode: keep everything except known hardware/boundary artifacts
            ignored_keywords = ["bad", "boundary", "edge"]
            selected_event_id = {
                k: v
                for k, v in event_dict.items()
                if not any(ign in k.lower() for ign in ignored_keywords)
            }
            print(
                "No target_event specified. Using all non-artifact events:"
                f" {list(selected_event_id.keys())}"
            )
        
        
        print(f"Slicing epochs from {tmin}s to {tmax}s...")
        epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=tmin, tmax=tmax, 
                            baseline=(None, 0), preload=True)

        """
        evoke construction site  //for if a trial does repetitive action through one run  //for another day
        
        print("Calculating and saving Evoked average...")
        evoked = epochs.average()
        evoked.save(evoked_path.fpath, overwrite=True)
        
        """
        # Filter the raw events array to target events first so indices match run_meta 1-to-1
        target_codes = list(selected_event_id.values())
        events = events[np.isin(events[:, 2], target_codes)]

        print(f"Slicing epochs from {tmin}s to {tmax}s...")
        epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=tmin, tmax=tmax, 
                            baseline=(None, 0), preload=True)

        # Align run_meta using the exact indices of surviving epochs
        if len(epochs.selection) <= len(run_meta):
            run_meta = run_meta.iloc[epochs.selection].reset_index(drop=True)
        else:
            min_len = min(len(epochs), len(run_meta))
            epochs = epochs[:min_len]
            run_meta = run_meta.iloc[:min_len].copy()

        # 6. Action labels & PSD calculation
        clean_labels = (run_meta[condition_col].astype(str).str.strip().str.replace(" ", "_"))
        trial_conditions = (clean_labels + "_" + run_meta.groupby(clean_labels).cumcount().add(1).astype(str)).tolist()

        action_epochs = epochs.copy().crop(tmin=0.5, tmax=2.0)
        spectrum = action_epochs.compute_psd(method="welch", fmin=8.0, fmax=30.0)
        psd_data, freqs = spectrum.get_data(return_freqs=True)

        mu = psd_data[:, :, (freqs >= 8.0) & (freqs <= 12.0)].mean(axis=-1)
        beta = psd_data[:, :, (freqs >= 15.0) & (freqs <= 30.0)].mean(axis=-1)
        feature_matrix_z = zscore(np.hstack([mu, beta]), axis=0)

        # 7. Save outputs
        np.save(features_out, feature_matrix_z)
        pd.DataFrame({"condition": trial_conditions}).to_csv(labels_out, sep="\t", index=False)

        return features_out, labels_out

    except Exception as e:
        print(f"\n[CRITICAL ERROR in process_meeg for sub-{subject_id}]: {e}")
        raise e  # Re-raises so caller receives clear traceback