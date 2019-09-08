import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import symptoms_classifier.classifiers_utils as cutils
np.random.seed(42)


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
    net = cutils.create_nn(nb_classes=nb_classes, first_layer_neurons=first_layer_neurons, dropout=dropout)
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
            # train_preds_idx = train_preds.argmax(1).numpy()  # torch.max(outputs, dim=1).indices.numpy()
            test_preds = net(x_test)

            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            cutils.print_nn_perf(train_preds=train_preds.detach(), y_train=y_train,
                                 test_preds=test_preds.detach(), y_test=y_test, multi_class=True)

    print('Finished Training')

    preds, df_test, df_train = cutils.evaluate_nn(y, train_preds.detach(), y_train, test_preds.detach(), y_test, multi_class=True)

    return [preds, df_test, df_train]
