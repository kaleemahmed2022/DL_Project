'''
quick script to take .m4a data directly from the vox dataset (in ./dataset/raw/) and convert to mono WAV
format. Can also include further preproessing steps here if necessary.
'''
import os

os.system('conda install -c main ffmpeg')  # need this for pydub to function
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import wave
import torch
from torchvision import transforms


def mkdir_if_not_exists(path):
    '''

    Args:
        path: path to check the existance of. Must be a directory path without a file at end

    Returns: None, but creates the path if it doesn't exist

    '''
    path_splits = path.split('/')
    path_splits.remove('.')
    incremental_path = '.'  # we need to iterate through all the subdirectories in 'path' to incrememtally create them
    for subpath in path_splits:

        incremental_path = os.path.join(incremental_path,
                                        subpath)  # build the incrememtal path, check if it exists and build
        if not os.path.exists(incremental_path):
            os.mkdir(incremental_path)
        else:
            pass

    return


def m4a_to_wav(input_path, output_path):
    '''

    Args:
        input_path: path to the m4a target file
        output_path: path where the wav file will be exported

    Returns: None

    '''

    track = AudioSegment.from_file(input_path, format='m4a')  # read the m4a file
    file_handle = track.export(output_path, format='wav')  # export the wav
    return


def wav_to_spectrogram(input_path, output_path,
                       sr = 16e3,
                       Ts = 10,
                       Tw = 25):
    '''

    Args:
        input_path: path to the wav file
        output_path: normalized spectrogram as a tensor
        sr = sample rate of wav requiried
        Ts = time step (ms) of sliding fft window
        Tw = window size of fft window (ms)

    Returns: None

    '''

    Ns = int(float(Ts)/1000 * sr) # need to specify the size of the fft windows
    Nw = int(float(Tw)/1000 * sr)

    x, sr = librosa.load(input_path, sr=sr, mono=True, duration=3)
    X = librosa.stft(x, hop_length=Ns, win_length=Nw) # FFT in complex numbers
    Xdb = np.abs(X) # abs value to take amplitude data of complex matrix
    data = (Xdb - Xdb.mean()) / X.std()
    torch.save(torch.tensor(data), output_path) # the prior transform was messing with the datashape
    return


def gen_phases(DATAPATH, train_split=0.7, valid_split=0.15, test_split=0.15):
    '''

    generates iden_split.txt at DATAPATH/iden_split.txt.

    calculates phases based on provided splits using a random generator.
    A single id and context directory will always be in the same phase!!!! (prevents dataleakage)
    Every id will have a combination of all 3 phases
    '''

    # normalise splits
    splits = [train_split, valid_split, test_split]
    assert sum(splits) == 1.

    iden_split = pd.DataFrame(columns=['phase', 'path', 'id', 'context'])
    ids = os.listdir(DATAPATH)
    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(DATAPATH, id))
        phases = list(np.random.choice([1, 2, 3], p=splits, size=len(contexts)))
        id_int = int(id.replace('id', '')) - 1

        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(DATAPATH, id, ctx)

            phase = phases.pop(0)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                filepath = os.path.join(id, ctx, f)
                iden_split.loc[len(iden_split)] = [phase, filepath, id_int, ctx]
    iden_split.to_csv(os.path.join(DATAPATH, 'phase_map.csv'))
    return


# def convert_sample_rate(filename):
#    '''
#    NOT NEEDED??
#    converts and overwrites a .wav file at ./filename to 16kHz
#    Args:
#        filename: path + name (ending in .wav) of file to be converted
#    Returns:
#    '''
#    rootdir = './dataset/processed/'
#    ids = os.listdir(rootdir)
#    for id in tqdm(ids):
#
#        contexts = os.listdir(os.path.join(rootdir, id))
#        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
#        for ctx in contexts:
#            path = os.path.join(rootdir, id, ctx)
#            files = os.listdir(path)
#            for f in files:
#                librosa.load(os.path.join(path, f), sr=16000)


def check_sample_rates():
    '''

    checks the sample rate of all .wav files inside rootdir
    '''

    sample_rates = pd.DataFrame(index=None, columns=['id', 'context', 'file', 'sample rate'])
    rootdir = './dataset/processed/'
    ids = os.listdir(rootdir)
    if 'iden_split.csv' in ids: ids.remove('iden_split.csv')
    for id in tqdm(ids):
        contexts = os.listdir(os.path.join(rootdir, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:
            path = os.path.join(rootdir, id, ctx)
            files = os.listdir(path)
            for f in files:
                with wave.open(os.path.join(path, f), "rb") as wave_file:
                    sr = wave_file.getframerate()
                    sample_rates.loc[len(sample_rates)] = [id, ctx, f, sr]
    sample_rates.hist(column='sample rate')
    return


def dataset_to_wav():
    '''

    Scans through all subdirectories of ./dataset/raw/, and recreates them in ./dataset/processed/ (writing new
    directories if needed), and converting the m4a files into wav format
    '''

    rootdir = './dataset/raw/'
    outdir = './dataset/processed/'
    ids = os.listdir(rootdir)
    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(rootdir, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(rootdir, id, ctx)
            procpath = os.path.join(outdir, id, ctx)
            mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                m4a_to_wav(os.path.join(rawpath, f),
                           os.path.join(procpath, f[:-3] + 'wav'))
    return


def dataset_to_pt():
    '''

    Scans through all subdirectories of ./dataset/processed/, and recreates them in ./dataset/spectrogram/ (writing new
    directories if needed), and converting the wav files into pt files with spectrograms

    '''
    rootdir = './dataset/processed/'
    outdir = './dataset/spectrograms/'
    ids = os.listdir(rootdir)
    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(rootdir, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(rootdir, id, ctx)
            procpath = os.path.join(outdir, id, ctx)
            mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                wav_to_spectrogram(os.path.join(rawpath, f),
                                   os.path.join(procpath, f[:-3] + 'pt'))

    return


def normalise_spectograms(spect, rootdir='./dataset/processed/'):
    '''

    Normalises the amplitudes in a spectogram by (val-mean)/stdev

    Args:
        spect (str): path to spectogram file in binary torch format (.pt)
        rootdir (str): string to the root of the data directory

    Rewrites after normalising the spectogram. Same place it was read from

    '''

    ids = os.listdir(rootdir)
    if 'iden_split.csv' in ids: ids.remove('iden_split.csv')
    for id in tqdm(ids):
        contexts = os.listdir(os.path.join(rootdir, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:
            path = os.path.join(rootdir, id, ctx)
            files = os.listdir(path)
            for f in files:
                filepath = os.path.join(rootdir, id, ctx, f)
                data = torch.load(filepath)
                data = (data - data.mean()) / data.std()
                torch.save(data, filepath)
    return


if __name__ == '__main__':
    #    dataset_to_wav()
    dataset_to_pt()
#    gen_phases('./dataset/processed/', train_split=0.7, valid_split=0.15, test_split=0.15)
# check_sample_rates()
