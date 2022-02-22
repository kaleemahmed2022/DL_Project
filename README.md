# DL_Project

## Dataset

The dataset is stored in ./dataset/, with the original data downloaded from VOX in ./dataset/raw/

Our code takes mono audio files in .WAV format, which are stored in an identcal structure in ./dataset/processed/
(see guide on preprocessing)

## Preprocessing

Preprocessing is executed by preprocessing.py, which is a stand-alone script that can be run. This script takes
raw .m4a files from ./dataset/raw/ , and builds an identical dataset in ./dataset/processed/ but with all .m4a files
converted to .wav

Additional preprocessing steps can be included in this script


