# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class EnvDataset(data.Dataset):

    def __init__(self, mode='train',root='./data/subset.npz', type = 'openset'):
        dataset = np.load(root)
        x = dataset['x']
        y = dataset['y']
        if type == 'openset':
            dataset = np.load('./data/open_set_train.npz')
            X_train = dataset['x']
            y_train = dataset['y']
            dataset = np.load('./data/open_set_test.npz')
            X_test = dataset['x']
            y_test = dataset['y']
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

        if mode == 'train':
            self.x = X_train
            self.y = y_train
        else:
            self.x = X_test
            self.y = y_test

    def __getitem__(self, idx):
        return np.array(self.x[idx],dtype=np.float32), self.y[idx]

    def __len__(self):
        return len(self.y)
    

# dataset = EnvDataset()
# for xx,yy in dataset:
#     print(xx.shape,yy)
