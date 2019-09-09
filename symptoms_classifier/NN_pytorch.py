import torch
import torch.optim as optim
from torch import nn
from symptoms_classifier.classifiers_utils import nn_classification_report, nn_print_perf
from sklearn.metrics import accuracy_score
from code_utils.plot_utils import plot_multi_lists
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)


class Net(nn.Module):
    def __init__(self, first_layer_neurons, nb_classes, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(first_layer_neurons, 100)
        self.fc2 = nn.Linear(100, nb_classes)  # 3 classes = 3 neurons
        if dropout is not None:
            print('using dropout')
            self.d1 = nn.Dropout(dropout)  # do we want dropout?

    def forward(self, x):
        if hasattr(self, 'd1'):
            x = self.d1(torch.relu(self.fc1(x)))
        else:
            x = torch.relu(self.fc1(x))  # torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
# net = Net(first_layer_neurons=100, nb_classes=1, dropout=0.5)


def train_nn_simple(x_emb, y, test_size=0.2, random_state=0, class_weight='balanced', dropout=0.5, n_epochs=5000, debug_mode=True):

    x_train, x_test, y_train, y_test = train_test_split(x_emb, y, test_size=test_size, random_state=random_state)

    """#Convert the inputs to PyTorch"""
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    first_layer_neurons = x_train.shape[1]

    """# Initialize the NN, create the criterion (loss function) and the optimizer """
    # net = nn_create(nb_classes=1, first_layer_neurons=first_layer_neurons, dropout=dropout)
    net = Net(first_layer_neurons=first_layer_neurons, nb_classes=1, dropout=dropout)
    criterion = nn.BCELoss()  # loss function (binary cross entropy loss or log loss)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)  # sigmoid gradient descent optimizer to update the weights

    """# Train the NN"""
    losses, accs, ws, bs = [[], [], [], []]
    for epoch in range(n_epochs):
        net.train()
        optimizer.zero_grad()  # zero the gradients
        train_preds = net(x_train)  # Forward
        loss = criterion(train_preds, y_train_torch)  # Calculate error
        loss.backward()  # Backward
        optimizer.step()  # Optimize/Update parameters

        # Track the changes - This is normally done using tensorboard or similar
        if debug_mode:
            losses.append(loss.item())
            accs.append(accuracy_score([1 if x > 0.5 else 0 for x in train_preds.detach().numpy()], y_train))
            ws.append(net.fc1.weight.detach().numpy()[0][0])
            bs.append(net.fc1.bias.detach().numpy()[0])

        # print statistics
        if epoch % 500 == 0:
            net.eval()
            test_preds = net(x_test)
            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            nn_print_perf(train_preds=train_preds.detach(), y_train=y_train,
                          test_preds=test_preds.detach(), y_test=y_test, multi_class=False)
    print('Finished Training')

    if debug_mode:
        plot_multi_lists({'Bias': bs, 'Weight': ws, 'Loss': losses, 'Accuracy': accs})

    preds, df_test, df_train = nn_classification_report(y, train_preds.detach(), y_train, test_preds.detach(), y_test, multi_class=False)
    # net(torch.tensor([22], dtype=torch.float32))  # 22: input temperature

    return [net, preds, df_test, df_train]


def train_nn(x_emb, y, test_size=0.2, random_state=0, class_weight='balanced', dropout=0.5, n_epochs=5000, debug_mode=True):
    ####################################################################################################################
    # 1. split dataset in train/test
    ####################################################################################################################
    x_train, x_test, y_train, y_test = train_test_split(x_emb, y, test_size=test_size, random_state=random_state)

    weights = list(y_train.value_counts()/y_train.count())
    nb_classes = y_train.nunique()
    first_layer_neurons = x_train.shape[1]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    ####################################################################################################################
    # 2. build network
    ####################################################################################################################
    # net = cutils.nn_create(nb_classes=nb_classes, first_layer_neurons=first_layer_neurons, dropout=dropout)
    net = Net(first_layer_neurons=first_layer_neurons, nb_classes=nb_classes, dropout=dropout)
    weight = torch.tensor(weights) if class_weight == 'balanced' else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.99)  # TODO: move to Adam?

    ####################################################################################################################
    # 3. train
    ####################################################################################################################
    for epoch in range(n_epochs):
        net.train()
        optimizer.zero_grad()
        train_preds = net(x_train)  # forward
        loss = criterion(train_preds, y_train_torch)
        loss.backward()
        optimizer.step()

        # print statistics
        if epoch % 500 == 0:
            net.eval()
            test_preds = net(x_test)

            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            nn_print_perf(train_preds=train_preds.detach(), y_train=y_train,
                          test_preds=test_preds.detach(), y_test=y_test, multi_class=True)

    print('Finished Training')

    preds, df_test, df_train = nn_classification_report(y, train_preds.detach(), y_train, test_preds.detach(), y_test, multi_class=True)

    return [net, preds, df_test, df_train]