import os
import mne
from mne_bids import BIDSPath, read_raw_bids

# 1. Point to your MEEG dataset
MEEG_ROOT = '/home/idhuang/bhs2026/project_sophia/data/ds007353/'

print("Initializing BIDS path...")

# 2. Define exactly what we want to load
bids_path = BIDSPath(
    subject='01',
    session='meg',  # <--- NEW: Tells it to open the 'ses-meg' folder
    task='action', 
    run='04',
    datatype='meg', 
    root=MEEG_ROOT
)

# 3. Read the raw data
print(f"Loading data for {bids_path.basename}...")
raw = read_raw_bids(bids_path=bids_path, verbose=False)

# 4. Print the metadata and plot the sensors!
print("\n--- Data Successfully Loaded ---")
print(raw.info)

# Pick a few channels to print so we don't flood the terminal
print("\nFirst 10 Channel Names:")
print(raw.ch_names[:10])

print("\n--- Starting Preprocessing ---")

print("Loading data into RAM...")
raw.load_data()


#iir to handle short data => fir met 397 length too short instead of 793
print("Applying 60Hz Notch Filter...")
raw.notch_filter(freqs=60.0, method='iir')

print("Applying 1-40Hz Bandpass Filter...")
raw.filter(l_freq=60.0, h_freq=40.0, method='iir')

print("Plotting the cleaned signals...")
#raw.plot(duration=5.0, n_channels=20)
raw.plot(duration=5.0, n_channels=20, block=True)

