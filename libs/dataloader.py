import os
import numpy as np
from torch.utils.data import Dataset
from libs.helper import get_mfcc_features

class LibriSpeechData(Dataset):
    def __init__(self, metadata, info, data_dir='./LibriSpeech', max_len=None, norm_path='./data/mfcc_scaler.npy'):
        self.info = info[info['speaker_id'].isin(metadata)]
        self.data_dir = data_dir
        self.max_len = max_len
        self.norm_path = norm_path

        if not os.path.exists(self.norm_path):
            print('Normalization file not found, please run dataloader.ipynb to generate it')
            raise ValueError('Normalization file not found')

        self.sex_dict = {'M': 0, 'F': 1}

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        sample = self.info.iloc[idx]
        
        speaker_id = sample['speaker_id']
        transcript = sample['transcript']
        sex = sample['sex']

        audio_path = f'{self.data_dir}/{sample["audio_path"]}'
        mfcc_features = get_mfcc_features(audio_path)
        
        if self.max_len is not None:
            if mfcc_features.shape[-1] < self.max_len:
                mfcc_features = np.pad(mfcc_features, ((0, 0), (0, self.max_len - mfcc_features.shape[-1])), 'constant')
            else:
                mfcc_features = mfcc_features[:, :self.max_len]

        label = self.sex_dict[sex]
        return mfcc_features, label, (speaker_id, transcript)