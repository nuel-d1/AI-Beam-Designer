# imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib

class feature_dataset(Dataset):

    def __init__(self, data):
        sc_x = MinMaxScaler()
        sc_y = MinMaxScaler()
        
        data[['Span (m)', 'Ultimate load (kN/m)']] = sc_x.fit_transform(data[['Span (m)', 'Ultimate load (kN/m)']])
        x = data.iloc[0:, 0:9].values
        y = data.iloc[0:, 9:].values

        
        x_values = x
        y_values = sc_y.fit_transform(y)

        joblib.dump(sc_x, 'scaler_x')
        joblib.dump(sc_y, 'scaler_y')
        
        self.x_values = torch.tensor(x_values, dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y_values)

    def __getitem__(self, index):
        return self.x_values[index], self.y_values[index]

