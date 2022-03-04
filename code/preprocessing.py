import os
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import librosa.display
import wave
import torch
import time
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def mkdir_if_not_exists(path):
    '''

    Args:
        path: path to check the existance of. Must be a directory path without a file at end

    Returns: None, but creates the path if it doesn't exist

    '''
    path_splits = path.split('/')
    #if '.' in path_splits: path_splits.remove('.')
    if '' in path_splits: path_splits[0] = '/'
    incremental_path = ''  # we need to iterate through all the subdirectories in 'path' to incrememtally create them
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

def m4a_to_specs(input_path, output_path,
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
    X = librosa.stft(x, hop_length=Ns, win_length=Nw, n_fft=1024) # FFT in complex numbers
    Xdb = np.abs(X) # abs value to take amplitude data of complex matrix
    img = librosa.display.specshow(Xdb, y_axis='linear', x_axis='time',sr=sr)
    # plt.savefig(output_path)
    Xlog = np.log(Xdb)
    log_img = librosa.display.specshow(Xlog, y_axis='linear', x_axis='time',sr=sr)
    plt.savefig(output_path)
    data = (Xlog - Xlog.mean()) / Xlog.std()
    # NB: unsqueeze needed to replicate dimension of number of channel (just 1 in our case):
    torch.save(torch.tensor(data).unsqueeze(0), output_path) # the prior transform was messing with the datashape
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
    X = librosa.stft(x, hop_length=Ns, win_length=Nw, n_fft=1024) # FFT in complex numbers
    Xdb = np.abs(X) # abs value to take amplitude data of complex matrix
    # img = librosa.display.specshow(Xdb, y_axis='linear', x_axis='time',sr=sr)
    Xlog = np.log(Xdb)
    # log_img = librosa.display.specshow(Xlog, y_axis='linear', x_axis='time',sr=sr)
    data = (Xlog - Xlog.mean()) / Xlog.std()
    # NB: unsqueeze needed to replicate dimension of number of channel (just 1 in our case):
    # torch.save(img, output_path+'png')
    # torch.save(log_img, output_path+'png')
    torch.save(torch.tensor(data).unsqueeze(0), output_path) # the prior transform was messing with the datashape
    return

def noise_tests(input_path,
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

    Ns = int(float(Ts) / 1000 * sr)  # need to specify the size of the fft windows
    Nw = int(float(Tw) / 1000 * sr)

    # x, sr = librosa.load(input_path, sr=sr, mono=True, duration=3)
    x, sr = librosa.load(input_path, sr=sr, mono=True)
    # print(x)
    dur = librosa.get_duration(x,sr)
    noised = x + 0.009 * np.random.normal(0, 1, len(x))
    # print(noised)
    time_stretched = librosa.effects.time_stretch(x, 0.4)
    # print(time_stretched)
    return dur

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
    if 'phase_map.csv' in ids: ids.remove('phase_map.csv')

    for id in tqdm(ids):  # run a proc bar just to keep track
        if "icon" not in id.lower() and "id" in id.lower():
            contexts = os.listdir(os.path.join(DATAPATH, id))
            phases = list(np.random.choice([1, 2, 3], p=splits, size=len(contexts)))
            id_int = int(id.replace('id', '')) - 1

            if '.DS_Store' in contexts: contexts.remove('.DS_Store')
            for ctx in contexts:
                if 'icon' not in ctx.lower():

                    rawpath = os.path.join(DATAPATH, id, ctx)

                    phase = phases.pop(0)
                    files = os.listdir(rawpath)
                    if '.DS_Store' in files: files.remove('.DS_Store')
                    for f in files:
                        if "icon" not in f.lower():
                            filepath = os.path.join(id, ctx, f)
                            iden_split.loc[len(iden_split)] = [phase, filepath, id_int, ctx]
        iden_split.to_csv(os.path.join(DATAPATH, 'phase_map.csv'))
    return


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


def dataset_to_wav(readpath, outpath, filenames=None):
    '''

    Scans through all subdirectories of ./dataset/raw/, and recreates them in ./dataset/processed/ (writing new
    directories if needed), and converting the m4a files into wav format
    '''

    if filenames is None:
        ids = os.listdir(readpath)
    else:
        ids = filenames

    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(readpath, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:
            if "icon" not in ctx.lower():
                rawpath = os.path.join(readpath, id, ctx)
                procpath = os.path.join(outpath, id, ctx)
                mkdir_if_not_exists(procpath)

                files = os.listdir(rawpath)
                if '.DS_Store' in files: files.remove('.DS_Store')
                for f in files:
                    try:
                        m4a_to_wav(os.path.join(rawpath, f),
                                   os.path.join(procpath, f[:-3] + 'wav'))
                    except:
                        pass
    return


def dataset_to_pt(readpath, outpath, filenames = None):
    '''

    Scans through all subdirectories of ./dataset/processed/, and recreates them in ./dataset/spectrogram/ (writing new
    directories if needed), and converting the wav files into pt files with spectrograms

    added in filenames to isolate the filenames we want to perform spectrograms on
    '''

    if filenames is None:
        ids = os.listdir(readpath)
    else:
        ids = filenames

    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(readpath, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(readpath, id, ctx)
            procpath = os.path.join(outpath, id, ctx)
            mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                wav_to_spectrogram(os.path.join(rawpath, f),
                                   os.path.join(procpath, f[:-3] + 'pt'))

    return

def dataset_to_png(readpath, outpath, filenames = None):
    '''

    Scans through all subdirectories of ./dataset/processed/, and recreates them in ./dataset/spectrogram/ (writing new
    directories if needed), and converting the wav files into pt files with spectrograms

    added in filenames to isolate the filenames we want to perform spectrograms on
    '''

    if filenames is None:
        ids = os.listdir(readpath)
    else:
        ids = filenames

    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(readpath, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(readpath, id, ctx)
            procpath = os.path.join(outpath, id, ctx)
            mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                m4a_to_specs(os.path.join(rawpath, f),
                                   os.path.join(procpath, f[:-3] + 'png'))

    return

def noise_datasets(readpath, filenames = None):
    '''

    Scans through all subdirectories of ./dataset/processed/, and recreates them in ./dataset/spectrogram/ (writing new
    directories if needed), and converting the wav files into pt files with spectrograms

    added in filenames to isolate the filenames we want to perform spectrograms on
    '''

    if filenames is None:
        ids = os.listdir(readpath)
    else:
        ids = filenames
    durations = []
    if '.DS_Store' in ids: ids.remove('.DS_Store')
    for id in tqdm(ids):  # run a proc bar just to keep track

        contexts = os.listdir(os.path.join(readpath, id))
        if '.DS_Store' in contexts: contexts.remove('.DS_Store')
        for ctx in contexts:

            rawpath = os.path.join(readpath, id, ctx)
            # procpath = os.path.join(outpath, id, ctx)
            # mkdir_if_not_exists(procpath)

            files = os.listdir(rawpath)
            if '.DS_Store' in files: files.remove('.DS_Store')
            for f in files:
                durations.append(noise_tests(os.path.join(rawpath, f)))
    print(durations)
    print(max(durations))
    return

if __name__ == '__main__':
    print(os.getcwd())
#     print(os.path.dirname(os.path.realpath('/Users/devyanigauri/Documents/GitHub/DL_Project/dataset'
# )))
    #m4apath = '/Users/devyanigauri/Documents/GitHub/DL_Project/dataset/raw'
    #wavpath = '/Users/devyanigauri/Documents/GitHub/DL_Project/dataset/wav'
    #sptpath = '/Users/devyanigauri/Documents/GitHub/DL_Project/dataset/spectrograms'
    m4apath = os.path.normpath(os.getcwd()+os.sep+os.pardir+os.sep+os.pardir)+'/minidata/raw'
    sptpath = os.path.normpath(os.getcwd()+os.sep+os.pardir+os.sep+os.pardir)+'/minidata/spectrograms'
    # m4apath = '../dataset/raw'
    # sptpath = '../dataset/spectrograms'
    # dataset_to_wav(m4apath, wavpath)
    # dataset_to_png(m4apath, sptpath)
    dataset_to_pt(m4apath, sptpath)
    # noise_datasets(m4apath)
    gen_phases(m4apath, train_split=0.7, valid_split=0.15, test_split=0.15)
