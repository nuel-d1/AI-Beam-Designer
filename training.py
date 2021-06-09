# imports
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from model import Net
from dataset import feature_dataset
from torch.utils.tensorboard import SummaryWriter

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
    writer = SummaryWriter()
    epochs = 500
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        train_loss = []

        for features, labels in train_loader:
            output = model(features)
            loss = criterion(output, labels)
            train_loss.append(loss.item())
            train_loss_list.append(loss.item())
            tr_loss = np.array(train_loss)

            for i, val in enumerate(tr_loss):
                writer.add_scalar(
                    'Loss/train', scalar_value=val, global_step=i)
            writer.close()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(epoch % 100 == 0):
            # Testing
            with torch.no_grad():
                test_loss = []
                
                for features, labels in test_loader:
                    output = model(features)
                    loss = criterion(output, labels)
                    test_loss.append(loss.item())
                    test_loss_list.append(loss.item())
                    te_loss = np.array(test_loss)
                    
                    for i, val in enumerate(te_loss):
                        writer.add_scalar(
                            'Loss/test', scalar_value=val, global_step=i)
                    writer.close()

            print('Epoch {}/{} | Training loss: {:.4f} | Test loss: {:.4f}'.format(
                epoch, epochs, sum(train_loss), sum(test_loss)))


def make_plot(train_loss, test_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(test_loss)
    plt.plot(train_loss)
    plt.legend()
    plt.show()


training()
