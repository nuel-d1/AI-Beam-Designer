# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from model import Net
from dataset import feature_dataset

feature_set = feature_dataset('data/data.csv')
train_loader = torch.utils.data.DataLoader(feature_set, shuffle=True)

model = Net()
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def training():
    epochs = 100

    for epoch in range(epochs):
        train_loss = []
        for features, labels in train_loader:
            output = model(features)
            loss = criterion(output, labels)
            train_loss.append(loss.item())

            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

        print('Training loss: {}'.format(sum(train_loss)))
