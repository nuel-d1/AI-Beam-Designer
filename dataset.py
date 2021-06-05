# imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class feature_dataset(Dataset):

    def __init__(self, filename):
        data = pd.read_csv(filename)
        x = data.iloc[0:, 0:9].values
        y = data.iloc[0:, 9:].values

        sc = MinMaxScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
