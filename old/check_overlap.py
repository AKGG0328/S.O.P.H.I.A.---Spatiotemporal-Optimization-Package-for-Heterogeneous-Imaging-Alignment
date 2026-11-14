import numpy as np
import re
from collections import Counter

# Load the M/EEG Labels
meeg_labels = np.load('/home/idhuang/bhs2026/project_sophia/data/ds007353/sub-01_meeg_labels.npy', allow_pickle=True)

# Load the fMRI Labels
label_path = '/home/idhuang/bhs2026/project_sophia/data/ds004488/derivatives/ciftify/sub-01/results/ses-action01_task-action_cycle-1_label.txt'
with open(label_path, 'r') as file:
    fmri_labels_raw = [line.strip() for line in file.readlines()]

def extract_fmri_name(raw_string):
    # Look for the text between 'v_' and '_id_'
    match = re.search(r'^v_(.*?)_id_', raw_string)
    if match:
        return match.group(1)
    return raw_string

def normalize(text):
    return str(text).lower().replace(' ', '').replace('_', '').replace('-', '')

fmri_labels_extracted = [extract_fmri_name(label) for label in fmri_labels_raw]


print("--- Visual Inspection ---")
print(f"M/EEG Sample: {meeg_labels[:5]}")
print(f"fMRI Sample:  {fmri_labels_extracted[:5]}\n")


# Calculate the exact mathematical intersection
# This finds the video names that exist in BOTH lists
meeg_norm = np.array([normalize(L) for L in meeg_labels])
fmri_norm = np.array([normalize(L) for L in fmri_labels_extracted])

'''
#  Check the mathematical intersection
common_videos = np.intersect1d(meeg_norm, fmri_norm)

print("--- Final Overlap Diagnostic ---")
print(f"Perfect Matches Found: {len(common_videos)}")

if len(common_videos) == 90:
    print("\nSUCCESS! All 90 M/EEG videos exist in the fMRI dataset.")
elif len(common_videos) > 0:
    print(f"\n Partial match. Only {len(common_videos)} videos overlap.")
else:
    print("\n 0 Matches. The datasets are entirely different.")
'''

meeg_counts = Counter(meeg_norm)
fmri_counts = Counter(fmri_norm)

print("--- Trial Count Diagnostic ---")
perfect_match = True
mismatches = []

#  Check every video in the M/EEG list against the fMRI list
for video, meeg_count in meeg_counts.items():
    fmri_count = fmri_counts.get(video, 0) # Returns 0 if the fMRI doesn't have it at all
    
    if meeg_count != fmri_count:
        perfect_match = False
        mismatches.append(f"'{video}': M/EEG played {meeg_count} times, fMRI played {fmri_count} times")

#  The Verdict
if perfect_match:
    print(f" FLAWLESS ALIGNMENT: All {len(meeg_counts)} unique M/EEG videos have the EXACT same trial count in the fMRI dataset.")
    
    # Check if fMRI simply has extra videos we need to drop
    fmri_only_videos = set(fmri_counts.keys()) - set(meeg_counts.keys())
    if fmri_only_videos:
        print(f" Note: fMRI has {len(fmri_only_videos)} extra videos.")
else:
    print(" MISMATCH DETECTED: The number of trials per video do not line up.")
    for m in mismatches[:10]: # Print the first 10 errors so we can see what went wrong
        print(m)