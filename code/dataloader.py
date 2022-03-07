import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import librosa
import librosa.display
import warnings
from scipy import signal
import cv2

warnings.filterwarnings('ignore') # surpress warnings


def cmvnw(vec, win_size=301, variance_normalization=False):
    # Implementation from https://github.com/astorfi/speechpy
    """ This function is aimed to perform local mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        win_size (int): The size of sliding window for local normalization.
            Default=301 which is around 3s if 100 Hz rate is
            considered(== 10ms frame stide)
        variance_normalization (bool): If the variance normilization should
            be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    eps = 2 ** -30
    rows, cols = vec.shape

    # Windows size must be odd.
    assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), 'symmetric')
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i:i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:

        # Initial definitions.
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(
            mean_subtracted, ((pad_size, pad_size), (0, 0)), 'symmetric')

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i:i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] \
                = mean_subtracted[i, :] / (window_variance + eps)
        output = variance_normalized
    else:
        output = mean_subtracted

    return output


class VoxDataset(Dataset):

    def __init__(self, rootpath, phase='train', phase_map_file='phase_map.csv'):
        super(Dataset, self).__init__()
        '''

        Args: rootpath: path to the parent folder of data. typically ./dataset/processed/
            train: (str): either 'train', 'validation', 'test'

        '''
        self.rootpath = rootpath
        self.phase = phase

        map_data = pd.read_csv(os.path.join(rootpath, phase_map_file))[['phase', 'path', 'id', 'context']]

        phase_int = {'train': 1, 'validation': 2, 'test': 3}[phase]  # map phase onto the representative int

        mask = (map_data['phase'] == phase_int)
        self.dataset = map_data[['path', 'id', 'context']][mask].reset_index(drop=True)

        self.dataset['id_int'] = self.dataset.apply(lambda x: list(self.dataset['id'].unique()).index(x['id']), axis=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''

        getitem written to specifically pull idx from disk into memory to avoid needing the full dataset
        stored in memory

        Args:
            idx: integer

        Returns: label (id string) and spectogram (in tensor format)

        '''

        sample_meta = self.dataset.iloc[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id_int']
        full_path = os.path.join(self.rootpath, sample_path)
        spec = torch.load(full_path)

        return label, spec.transpose(dim0=1, dim1=2)


class VoxDatasetFly(VoxDataset):

    def __init__(self, rootpath, phase='train', phase_map_file='phase_map.csv', fftmethod='librosa.sfft'):
        super(VoxDatasetFly, self).__init__(rootpath, phase=phase, phase_map_file=phase_map_file)
        self._fftmethod = fftmethod.lower()

    def __getitem__(self, idx):
        if self._fftmethod == 'librosa.stft':
            return self._librosaSTFT(idx)
        elif self._fftmethod == 'librosa.mfcc':
            return self._librosaMFCC(idx)
        elif self._fftmethod == 'librosa.mel':
            return self._librosaMEL(idx)
        elif self._fftmethod == 'signal.stft':
            return self._signalSTFT(idx)
        else:
            raise AssertionError("VoxDatsetFly __getitem__ not impemeneted: {}".format(self._fftmethod))

    def _librosaSTFT(self, idx):
        '''
        __getitem__ specialisation for using librosa's sfft
        '''

        sample_meta = self.dataset.iloc[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id_int']

        full_path = os.path.join(self.rootpath, sample_path)

        sr = 16e3
        Ts = 10
        Tw = 25

        Ns = int(float(Ts) / 1000 * sr)  # need to specify the size of the fft windows
        Nw = int(float(Tw) / 1000 * sr)

        x, sr = librosa.load(full_path, sr=sr, mono=True, duration=3)
        spec = librosa.stft(x, hop_length=Ns, win_length=Nw, n_fft=1024)  # FFT in complex numbers
        #spec_abs = librosa.amplitude_to_db(spec)
        spec_abs = np.log(np.abs(spec))

        #spec_log = np.log(spec_abs)
        spec_norm = (spec_abs - spec_abs.mean()) / spec_abs.std()
        spec_tens = torch.tensor(spec_norm).unsqueeze(0) # tensorize into the correct dimension
        return label, spec_tens.transpose(dim0=1, dim1=2)


    def _librosaMFCC(self, idx):
        '''
        __getitem__ specialisation for using librosa's mfcc
        '''

        sample_meta = self.dataset.iloc[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id_int']

        full_path = os.path.join(self.rootpath, sample_path)

        sr = 16e3
        Ts = 10
        Tw = 25

        Ns = int(float(Ts) / 1000 * sr)  # need to specify the size of the fft windows
        Nw = int(float(Tw) / 1000 * sr)

        x, sr = librosa.load(full_path, sr=sr, mono=True, duration=3)
        spec = librosa.feature.mfcc(x, hop_length=Ns, win_length=Nw, n_mfcc=1024)  # FFT in complex numbers
        #spec_abs = librosa.amplitude_to_db(spec)
        spec_abs = np.log(np.abs(spec))
        spec_norm = (spec_abs - spec_abs.mean()) / spec_abs.std()
        spec_resize = cv2.resize(spec_norm, dsize=(301, 513), interpolation=cv2.INTER_CUBIC)

        spec_tens = torch.tensor(spec_resize).unsqueeze(0)  # tensorize into the correct dimension
        return label, spec_tens.transpose(dim0=1, dim1=2)


    def _librosaMEL(self, idx):
        '''
        __getitem__ specialisation for using librosa's melspectrogram
        '''

        sample_meta = self.dataset.iloc[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id_int']

        full_path = os.path.join(self.rootpath, sample_path)

        sr = 16e3
        Ts = 10
        Tw = 25

        Ns = int(float(Ts) / 1000 * sr)  # need to specify the size of the fft windows
        Nw = int(float(Tw) / 1000 * sr)

        x, sr = librosa.load(full_path, sr=sr, mono=True, duration=3)
        spec = librosa.feature.melspectrogram(x, hop_length=Ns, win_length=Nw, n_fft=1024)  # FFT in complex numbers
        spec_abs = np.log(np.abs(spec))
        #spec_abs = librosa.amplitude_to_db(spec)
        #spec_abs = spec # ???
        spec_norm = (spec_abs - spec_abs.mean()) / spec_abs.std()
        spec_resize = cv2.resize(spec_norm, dsize=(301, 513), interpolation=cv2.INTER_CUBIC)

        spec_tens = torch.tensor(spec_resize).unsqueeze(0)  # tensorize into the correct dimension
        return label, spec_tens.transpose(dim0=1, dim1=2)


    def _signalSTFT(self, idx):
        '''
        TODO: this doesn't seem to work when run on a network... runs when debugging though
        __getitem__ specialisation for using scipy's sfft
        '''

        sample_meta = self.dataset.iloc[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id_int']

        full_path = os.path.join(self.rootpath, sample_path)

        window = 'hamming'
        Tw = 25
        Ts = 10
        sr = 16000

        #rate, samples = wavfile.read(full_path)
        samples, rate = librosa.load(full_path, sr=sr, mono=True, duration=3)

        Nw = int(rate * Tw * 1e-3)
        Ns = int(rate * (Tw - Ts) * 1e-3)

        nfft = 2 ** (Nw - 1).bit_length()
        pre_emphasis = 0.97

        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

        samples = signal.lfilter([1, -1], [1, -0.99], samples)
        dither = np.random.uniform(-1, 1, samples.shape)
        spow = np.std(samples)
        samples = samples + 1e-6 * spow * dither # add in some random noise

        if self.phase==1: # if training
            segment_len = 3
            upper_bound = len(samples) - segment_len * rate
            start = np.random.randint(0, upper_bound)
            end = start + segment_len * rate
            samples = samples[start:end]

        _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft,
                                        mode='magnitude', return_onesided=False)
        spec = cmvnw(spec, win_size=3 * rate + 1)
        spec *= rate / 10

        # now pad to match librosa size (512, 301)
        spec = np.pad(spec, 3, mode='constant', constant_values=0)[3:-2, 3:]

        spec_tens = torch.tensor(spec).unsqueeze(0)  # tensorize into the correct dimension
        return label, spec_tens.transpose(dim0=1, dim1=2)



class VoxDataloader(pl.LightningDataModule):

 def __init__(self, path, num_workers=2, batch_size=32, phase_map_file='phase_map.csv', fftmethod='librosa.stft'):
     super(VoxDataloader, self).__init__()
     self.num_workers = num_workers
     self.batch_size = batch_size

     self.train = VoxDatasetFly(path, phase='train', phase_map_file=phase_map_file, fftmethod=fftmethod)
     self.val = VoxDatasetFly(path, phase='validation', phase_map_file=phase_map_file, fftmethod=fftmethod)
     self.test = VoxDatasetFly(path, phase='test', phase_map_file=phase_map_file, fftmethod=fftmethod)

 def train_dataloader(self):
     return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

 def val_dataloader(self):
     return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

 def test_dataloader(self):
     return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

 def num_classes(self):
     return len(self.train.dataset['id_int'].unique())


if __name__ == '__main__':
 _root = '/Users/jameswilkinson/Downloads/dev/wav3/'
 dataloader = VoxDataloader(_root, phase_map_file='phase_map_small.csv', batch_size=32)
