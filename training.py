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
test_dataset = pd.read_csv('data/test.csv')

#
train = feature_dataset(train_dataset)
train_loader = torch.utils.data.DataLoader(train, shuffle=True)
test = feature_dataset(test_dataset)
test_loader = torch.utils.data.DataLoader(test, shuffle=True)

model = Net()
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def training():
    '''method for training the algorithm'''

    # number of epochs
    epochs = 1000

    for epoch in range(epochs):
        train_loss = []

        for features, labels in train_loader:

            # forward pass
            output = model(features)
            loss = criterion(output, labels)
            train_loss.append(loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(epoch % 100 == 0):
            with torch.no_grad():
                # model testing
                test_loss = []
                for t_features, t_labels in test_loader:
                    t_output = model(t_features)
                    t_loss = criterion(t_output, t_labels)
                    test_loss.append(t_loss.item())

            print('Epoch {}/{} | Training loss: {:.4f}| Test Loss: {}'.format(epoch,
                  epochs, sum(train_loss), sum(test_loss)/len(test_loader)))
    mod_plot(train_loss, test_loss)


def mod_plot(train_loss, test_loss):
    '''method for making train-validation loss plots'''

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(test_loss, label="test")
    plt.plot(train_loss, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# begin training
training()

# Save the model state
torch.save(model.state_dict(), 'designer_state.pth')
