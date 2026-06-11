import os
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# Point to your MEEG dataset
BIDS_ROOT = '/home/idhuang/bhs2026/project_sophia/data/ds007353/'

print("Initializing BIDS path...")

MODALITY = 'eeg'

#  Define exactly what we want to load
if MODALITY == 'meg':
    bids_path = BIDSPath(
        subject='01',
        session='meg',  # <--- NEW: Tells it to open the 'ses-meg' folder
        task='action', 
        run='04',
        datatype='meg', 
        root=BIDS_ROOT
    )
elif MODALITY == 'eeg':
    bids_path = BIDSPath(
        subject='01',
        session='eeg',
        task='action', 
        run='04',
        datatype='eeg', 
        root=BIDS_ROOT
    )





# Read the raw data
print(f"Loading data for {bids_path.basename}...")
raw = read_raw_bids(bids_path=bids_path, verbose=False)

#  Print the metadata and plot the sensors
print("\n--- Data Successfully Loaded ---")
print(raw.info)

# Pick a few channels to print so we don't flood the terminal
print("\nFirst 10 Channel Names:")
print(raw.ch_names[:10])

print("\n--- Starting Preprocessing ---")

print("Loading data into RAM...")
raw.load_data()

print("\n--- Hardware Check & Referencing ---")
channel_types = raw.get_channel_types()

if 'eeg' in channel_types:
    print(" EEG channels detected in the dataset. Applying Common Average Reference (CAR)...")
    #  this function should(?) ONLY alters EEG channels. 
    raw.set_eeg_reference('average', projection=False)
else:
    print(" Pure MEG dataset detected. Skipping EEG re-referencing.")


#-

#iir to handle short data => fir met 397 length too short instead of 793
print("Applying 60Hz Notch Filter...")
raw.notch_filter(freqs=60.0, method='iir')

print("Applying 1-40Hz Bandpass Filter...")
raw.filter(l_freq=1.0, h_freq=40.0, method='iir')

print("Plotting the cleaned signals...")
raw.plot(duration=5.0, n_channels=20)
#raw.plot(duration=5.0, n_channels=20, block=True)



if MODALITY == 'eeg':
    print("\n--- EEG Specific Cleaning ---")
    
    # Ensure physical 3D locations of the sensors
    # just a safety net so Topomaps don't crash
    print("Setting standard 10-05 sensor montage...")
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage) 

    print("Running ICA to separate brainwaves from eye blinks...")
    # find the 15 strongest electrical patterns in the data
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw)
    
    # Plot the ICA components to visually find exclude stuff
    print("Displaying ICA components.")
    ica.plot_components(inst=raw)
    
    ica.exclude = [0, 1, 2] #manually chosen
    
    print(f"Applying ICA. Removing components: {ica.exclude}")
    ica.apply(raw) #somehow doesn't work here? assume becuase python save timing?


print("\n--- Starting Epoching ---")
events, event_id = mne.events_from_annotations(raw)
tmin = -0.2  # Start 0.2 seconds before the stimulus
tmax = 2.0   # End 2.0 seconds after the stimulus
# ^ a little narrow for universal use tho

print(f"Slicing data from {tmin}s to {tmax}s around each event...")
epochs = mne.Epochs(
    raw, 
    events=events, 
    event_id=event_id['video on'],  #isolates one stimulus for example
    tmin=tmin, 
    tmax=tmax, 
    baseline=(None, 0), # Correct the baseline using the 200ms before the video
    preload=True,       # Load all slices into memory
    verbose=False
)

print("\n--- Epoching Complete ---")
print(f"Total number of action trials extracted: {len(epochs)}")
#print("Event types found in this run:", list(event_id.keys()))
if MODALITY == 'eeg':
    ica.apply(epochs) #apply directly to epoch here? i think fix my prob

print("\n--- Starting Averaging ---")

# Average the 90 epochs together
evoked = epochs.average()

print("Plotting the Evoked Response...")
evoked.plot(spatial_colors=True, titles='Average Brain Response to Action Videos')

if MODALITY == 'eeg':
    evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type='eeg')
elif  MODALITY == 'meg':
    evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type='mag')#mag/eeg


# Save the clean epochs to disk
epochs.save('/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_task-action-epo.fif', overwrite=True)

