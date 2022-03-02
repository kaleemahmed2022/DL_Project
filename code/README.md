# DL_Project


## Preprocessing

Preprocessing is executed by preprocessing.py, which contains a suite of functions for preprocessing.

Key functions to be run (require no inputs) are:

### Setup before preprocessing

The initial dataset (audio files in .m4a format) is held within the directory ./dataset/raw/.

The naming convention follows the format ./dataset/raw/id/context/filename.m4a.

### 1) dataset_to_wav()

This function takes raw .m4a files from ./dataset/raw/ , and builds an identical dataset in ./dataset/processed/ but with
all .m4a files converted to .wav.

### 2) gen_phases

This function generates a file 'phase_map.csv' in .dataset/processed/ which contains a complete indexation of the contents
of ./dataset/processed, along with a variable 'phase' which identifies which contents will be used for training (phase=1),
validation (phase=2) and testing (phase=4).

### 3) check_sample_rates()

This function checks all the wav files inside .dataset/processed/ and plots a histogram of the sample rates. NB: we
require all data to be the same samplerate (16kHz)

### 4) wav_to_spectogram()

Function to convert the .wav files in ./dataset/processed/ into spectograms. Spectograms must be in pytorch binary file
format.

### 4) normalise_spectograms()

Function to perform normlaisation on the spectograms (held in ./dataset/processed/) and to perform a normlaisation on them.
The function overwrites the existing spectograms with their new, normalised version, also saving them as torch binary
files.


### Other

Additional preprocessing steps can be included in this script. Eg: additional function to convert the wav files into
a spectogram in a torch binary file format.

## Dataloaders

In dataloader2.py, we have a dataloader for the VOX dataset. It requires spectograms to be already computed, normalised,
and stored at a given path. It also requires a "phase_map.csv" file to index the relevant files. Spectograms must be stored
in pytorch's binary file format.




