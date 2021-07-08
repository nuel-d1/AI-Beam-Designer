# imports
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from model import Net
from dataset import feature_dataset

# read entire data from csv file
train_dataset = pd.read_csv('data/data.csv')

train = feature_dataset(train_dataset)
train_loader = torch.utils.data.DataLoader(train, shuffle=True)

model = Net()
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#criterion = nn.BCELoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def training():
    epochs = 500

    for epoch in range(epochs):
        train_loss = []

        for features, labels in train_loader:
            output = model(features)
            loss = criterion(output, labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(epoch % 100 == 0):
            print('Epoch {}/{} | Training loss: {:.4f}'.format(epoch, epochs, sum(train_loss)))

# begin training
training()

# Save the model state
torch.save(model.state_dict(), 'designer_state.pth')
