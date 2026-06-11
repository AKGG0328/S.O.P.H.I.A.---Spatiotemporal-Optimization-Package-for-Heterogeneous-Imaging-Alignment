import numpy as np
import mne 
import matplotlib.pyplot as plt

print("--- Initializing Feature Extraction ---")
RUN = '04'
epoch_file = '/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_task-action-epo.fif'
print(f"Loading epoched data: {epoch_file}")
epochs = mne.read_epochs(epoch_file, preload=True)

video_labels = epochs.metadata['class_name'].values
# 2. Define the Mathematical Parameters for Morlet Wavelets
#  8Hz up to 30Hz cover Mu and Beta for human action dataset 
freqs = np.arange(8, 31, 1) 
# Standard Morlet parameter: higher freq need more cycles to balance time v frequency res
n_cycles = freqs / 2.0 

# 3. Run the Time-Frequency Analysis 
print("\nCalculating Morlet Wavelets across 90 trials. This requires heavy compute...")
power = mne.time_frequency.tfr_morlet(
    epochs, 
    freqs=freqs, 
    n_cycles=n_cycles, 
    # picks=['O1', 'O2', 'Oz', 'C3', 'C4', 'Cz'],
    use_fft=True, 
    decim= 2, #not bad potato
    return_itc=False, 
    average=False,
    n_jobs=-1 
)

# 4. Baseline Correction / ERD Drop)
print("Applying baseline correction (-0.2s to 0.0s)...")
# mode='percent' changes raw voltage into a percentage drop from the resting state
power.apply_baseline(mode='percent', baseline=(-0.2, 0.0))

# 5. Extract the Final Features (The Cognitive Window)
print("\nExtracting 2D feature matrix for RSA...")

# Isolate the Beta Band (15-30 Hz) and the Cognitive Time Window (0.5s to 2.0s)
beta_power = power.copy().crop(tmin=0.5, tmax=2.0, fmin=15.0, fmax=30.0)
mu_power = power.copy().crop(tmin=0.5, tmax=2.0, fmin=8.0, fmax=12.0)
# Average the power across the time window and frequency band
#  4D matrix => clean 2D feature table: (Trials x Sensors)  feature matrix
feature_beta = beta_power.data.mean(axis=(2, 3))
feature_mu = mu_power.data.mean(axis=(2, 3))  
feature_matrix = np.concatenate((feature_beta,  feature_mu), axis=1)

print(f"\n Feature Extraction Complete.")
print(f"Final MEEG Matrix Shape: {feature_matrix.shape}")
print(f"Final Label Count: {len(video_labels)}")

# 6. Plot the visual proof of ERD for the first trial, on a central motor sensor
sensor_to_plot = 'C3' if 'C3' in epochs.ch_names else epochs.ch_names[0]
print(f"Plotting Time-Frequency map for sensor {sensor_to_plot}...")
power[0].plot(picks=[sensor_to_plot], vlim=(-1.0, 1.0), title=f"Trial 1: ERD on {sensor_to_plot}")
#beta_power[0].plot(picks=[sensor_to_plot], vlim=(-1.0, 1.0), title=f"Trial 1: SVM Beta Feature Only")

plt.show()

output_dir = '/home/idhuang/bhs2026/project_sophia/data/ds007353/'
np.save(f'{output_dir}sub-01_meeg_features.npy', feature_matrix) 
np.save(f'{output_dir}sub-01_meeg_labels.npy', video_labels)

print(f"Final Matrix Shape: {feature_matrix.shape}")