import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from symptoms_classifier.tests import quick_embedding_ex

matplotlib.use('Qt5Agg')
np.random.seed(42)

# testing dataset
x_emb, y = quick_embedding_ex(
    data_file="https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv",
    text_col='text',
    class_col='airline_sentiment')


def train_NN(x_emb, y, test_size=0.2):
    ####################################################################################################################
    # 1. split dataset in train/test
    ####################################################################################################################
    x_train, x_test, y_train, y_test = train_test_split(x_emb, y, test_size=test_size)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)


    ####################################################################################################################
    # 2. build network
    ####################################################################################################################
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(300, 100)
            self.fc2 = nn.Linear(100, 3)  # 3 classes = 3 neurons
            self.d1 = nn.Dropout(0.5)  # do we want dropout?

        def forward(self, x):
            x = self.d1(torch.relu(self.fc1(x)))
            x = torch.sigmoid(self.fc2(x))
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()
    # to fix the imbalance: add weights to the cross entropy (here we tell the model we have 10 times more negatives)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.99)  # TODO: move to Adam?

    ####################################################################################################################
    # 3. train
    ####################################################################################################################
    net.train()
    for epoch in range(5000):
        optimizer.zero_grad()
        outputs = net(x_train)  # forward
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        if epoch % 500 == 0:
            net.eval()
            outputs_idx = torch.max(outputs, dim=1).indices.numpy()  # Use max to get the index of max for each row
            outputs_dev = net(x_test)
            outputs_dev_idx = torch.max(outputs_dev, dim=1).indices.numpy()  # Get the index of max per row

            # performance metrics
            acc = sklearn.metrics.accuracy_score(outputs_idx, y_train.numpy())
            acc_dev = sklearn.metrics.accuracy_score(outputs_dev_idx, y_test.numpy())
            f1_dev = sklearn.metrics.f1_score(outputs_dev_idx, y_test.numpy())  # Use f1 from sklearn
            p_dev = sklearn.metrics.precision_score(outputs_dev_idx, y_test.numpy())  # Use precision from sklearn
            r_dev = sklearn.metrics.recall_score(outputs_dev_idx, y_test.numpy())  # Use recall from sklearn

            print("Epoch: {:4} Loss: {:.5f} Acc: {:.3f} Acc Dev: {:.3f} F1 Dev: {:.3f} p Dev: {:.3f} r Dev: {:.3f}"
                  .format(epoch, loss.item(), acc, acc_dev, f1_dev, p_dev, r_dev))
            net.train()

    print('Finished Training')

    print(outputs_idx)
