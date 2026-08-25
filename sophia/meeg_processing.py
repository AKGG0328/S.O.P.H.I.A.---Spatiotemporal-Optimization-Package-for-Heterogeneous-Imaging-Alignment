import mne
from mne_bids import BIDSPath, read_raw_bids

def process_meeg(bids_root, dataset_dir, subject_id, task_name="action",   
                    session="eeg", modality="eeg", run="01",
                    l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.0, stim_channel='STI101'):
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
    print(f"\n--- Starting M/EEG processing for sub-{subject_id} ---")

    try:
        # 1. Construct the official MNE-BIDS path object
        bids_path = BIDSPath(
            subject=subject_id,
            session=session,
            task=task_name,
            run=run,
            datatype=modality,
            root=f"{bids_root}/{dataset_dir}"
        )
        
        print(f"Loading data from BIDS path: {bids_path.basename}...")
        
        # 2. Automatically handles .set, .fif, etc. using MNE-BIDS
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        
        # 3. Your standard pipeline math
        print(f"Filtering data from {l_freq} to {h_freq} Hz...")
        raw.filter(l_freq=l_freq, h_freq=h_freq)

        '''
        cleaning construction site 
        '''
        
        print(f"Extracting events from channel {stim_channel}...")  #problem currently with stim channel name
        events = mne.find_events(raw, stim_channel=stim_channel)
        
        print(f"Slicing epochs from {tmin}s to {tmax}s...")
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, 
                            baseline=(None, 0), preload=True)

        '''
        evoke construction site 
        '''
        
        print(f"Successfully processed sub-{subject_id}")
        return epochs 
        
    except Exception as e:
        print(f"CRITICAL ERROR processing sub-{subject_id}: {e}")
        return None