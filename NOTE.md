Project Sophia Dev Log 
------------------------------------------------------------

Roadmap 
- [x] Data ingestion
- [x] Pull BIDS dataset
- [x] M/EEG prototypying
- [x] fMRI spatial prototyping	
- [x] intergrate for **fmriprep preproceesed data** reading
- [x] Source Localization
- [ ] SVM skip for now 
	- [x] Feature extraction: time-frequency domain
	- [ ] ML process 
- [x] fMRI RDM
- [x] RSA



### 2026/06/02 02:48
- sanity_check.py for data checking 
- MEEG data preprocessing started
- Signal filtering completed 
- Set to Epoch slicing 

### 2026/06/03 12:14
- epoch slicing & evoke
- did not push
- **M/EEG preprocessing prototype complete**
- game plan :
1. freq domain for evoked data to feed svm via morlet wavelets or hilbert transformaton
2. nii.gz for source localization 
3. fMRI parallel track to check fMRI dataset


### 2026/06/05 12:12
- added road map to NOTE.md
- a great battle with typos was fought on this day
- ingest 4D spatial data
- Not much was achieved today :(
- set to GLM

### 2026/06/06 18:15
- GLM works
- *look into fmriprep and datapreprocessing*

### 2026/06/07 01:29
- transioning fmri pipeline to fmriprep standard

### 2026/06/11 13:09
- add epoch save to 01
- add more `f` for dual modality 
- time-freq analysis for meeg data 
- fmri finalization 
	- loop for average and 3D map 
	- write 02b for clean multirun
- meeg source localization
	- anatomy 3D model build
	-- struggling right now :（

###  2026/06/11 19:40
- figuring out fake head 
- HEAD (fsaverage bypass source localization)
- enter machine learning phase 
- minor problem in event alignment
- shift fmri work to derivate preprocessed data
- meeg video labeling
- add epoch label confirm to sanity check
- mod 03 with label

###  2026/06/12 09:28
- fmri rdm from 02b
- meeg rdm @05
- ml on hold 
- set for rsa fusion

### 2026/06/12 14:36
- rsa fusion
- work on read me 
- save visualization
- threshold mod on 02, 02b(comparison testing, no specific meaning)

### 2026/08/25 09:42
- Environment Port 
- Gameplan 2.0

### 2026/08/25 13:35
- complete data reading logic
- apply save logic/bypass system

### 2026/08/28 07:19
- ICA
- event slicing 
- fmri processing

### 2026/09/21 11:25
- been some time since last opened
- logic & roadmap check

### 2026/10/05 15:55
- gonna try to finish fmri part 
- work on meeg_processing from epoch .fif => .npy 
- epoched condition alignment fix 
- RSA fusion and vizualization
- next step : 1.tune category 2.wrap up 3.try larger dataset on big drive :)

### 2026/11/14 13:30
- core patching 
- public API
