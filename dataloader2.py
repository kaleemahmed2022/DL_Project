import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class VoxDataset(Dataset):

    def __init__(self, rootpath, train):
        '''

        Args: rootpath: path to the parent folder of data. typically ./dataset/processed/
            train: bool. True if we're training
        '''
        self.rootpath = rootpath
        self.train = train

        map_data = pd.read_csv(os.path.join(rootpath, 'phase_map.csv'))[['phase', 'path', 'id', 'context']]

        phases = [1, 2] if train else [3]

        mask = map_data['phase'].isin(phases)
        self.dataset = map_data[['path', 'id', 'context']][mask].reset_index(drop=True)

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

        sample_meta = self.dataset[idx]
        sample_path = sample_meta['path']
        label = sample_meta['id']
        full_path = os.path.join(self.rootpath, sample_path)
        spec = torch.load(full_path)
        return label, spec

x = VoxDataset('./dataset/spectrograms/', True)
