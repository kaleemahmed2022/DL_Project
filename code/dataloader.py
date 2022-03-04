import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore') # surpress warnings


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

    def __init__(self, rootpath, phase='train', phase_map_file='phase_map.csv'):
        super(VoxDatasetFly, self).__init__(rootpath, phase=phase, phase_map_file=phase_map_file)

    def __getitem__(self, idx):

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
        spec_abs = np.abs(spec)
        spec_log = np.log(spec_abs)
        spec_norm = (spec_log - spec_log.mean()) / spec_log.std()
        spec_tens = torch.tensor(spec_norm).unsqueeze(0) # tensorize into the correct dimension
        return label, spec_tens.transpose(dim0=1, dim1=2)



class VoxDataloader(pl.LightningDataModule):

 def __init__(self, path, num_workers=2, batch_size=32, phase_map_file='phase_map.csv'):
     super(VoxDataloader, self).__init__()
     self.num_workers = num_workers
     self.batch_size = batch_size

     self.train = VoxDatasetFly(path, phase='train', phase_map_file=phase_map_file)
     self.val = VoxDatasetFly(path, phase='validation', phase_map_file=phase_map_file)
     self.test = VoxDatasetFly(path, phase='test', phase_map_file=phase_map_file)

 def train_dataloader(self):
     return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

 def val_dataloader(self):
     return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

 def test_dataloader(self):
     return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

 def num_classes(self):
     return len(self.train.dataset['id_int'].unique())


if __name__ == '__main__':
 _root = './dataset/spectrograms/'
 train = VoxDataset(_root, 'train')
 valid = VoxDataset(_root, 'validation')
 test = VoxDataset(_root, 'test')

 dataloader = VoxDataloader(train, valid, test)
