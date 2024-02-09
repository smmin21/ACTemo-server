import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class IemoCapDataset(torch.utils.data.Dataset):
    def __init__(self, path=''):
        self.df = pd.read_csv(path)
        self.feature = [np.array(self.df.iloc[i, :-1].values, dtype=np.float32) for i in range(len(self.df))]
        self.emotion = [self.df.iloc[i, -1] for i in range(len(self.df))]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature[idx], dtype=torch.float32)
        emotion = torch.tensor(self.emotion[idx], dtype=torch.long)
        return feature, emotion


class RavdessDataset(Dataset):
    def __init__(self, path=''):
        self.df = pd.read_csv(path)
        self.feature = [np.array(self.df.iloc[i, :-1].values, dtype=np.float32) for i in range(len(self.df))]
        self.emotion = [self.df.iloc[i, -1] for i in range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature[idx], dtype=torch.float32)
        emotion = torch.tensor(self.emotion[idx], dtype=torch.long)
        return feature, emotion


class ActemoDataset(Dataset):
    def __init__(self, path=''):
        self.df = pd.read_csv(path)
        self.feature = [np.array(self.df.iloc[i, :-1].values, dtype=np.float32) for i in range(len(self.df))]
        self.emotion = [self.df.iloc[i, -1] for i in range(len(self.df))]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature[idx], dtype=torch.float32)
        emotion = torch.tensor(self.emotion[idx], dtype=torch.long)
        return feature, emotion
    

# if __name__ == '__main__':
#     dataset = RavdessDataset('SER_model/features_dataset.csv')
#     print(dataset)