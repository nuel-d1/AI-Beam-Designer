# imports
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class feature_dataset(Dataset):

    def __init__(self, data):
        x = data.iloc[0:, 0:9].values
        y = data.iloc[0:, 9:].values

        sc = MinMaxScaler()
        x_values = sc.fit_transform(x)
        y_values = sc.fit_transform(y)

        self.x_values = torch.tensor(x_values, dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)

    def __len__(self):
        return len(self.y_values)

    def __getitem__(self, index):
        return self.x_values[index], self.y_values[index]
