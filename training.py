# imports
import pandas as pd
import torch
from torch import optim, nn
from model import Net
from dataset import feature_dataset

# read entire data from csv file
dataset = pd.read_csv('data/data.csv')

# split dataset into training and testing sets &
# load into custom dataset class
train = feature_dataset(dataset[:90])
test = feature_dataset(dataset[90:])
train_loader = torch.utils.data.DataLoader(train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test)

model = Net()
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def training():
    epochs = 100

    for epoch in range(epochs):
        train_loss = []
        for features, labels in train_loader:
            output = model(features)
            loss = criterion(output, labels)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}/{} | Training loss: {:.4f}'.format(epoch +
              1, epochs, sum(train_loss)))


training()
