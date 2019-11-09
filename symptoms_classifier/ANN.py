import torch
import torch.optim as optim
from torch import nn
from symptoms_classifier.classifiers_utils import nn_classification_report, nn_print_perf, nn_graph_perf
from code_utils.plot_utils import plot_multi_lists
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self, first_layer_neurons, final_layer_neurons, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(first_layer_neurons, 100)
        self.fc2 = nn.Linear(100, final_layer_neurons)  # 3 categories/classes = 3 final neurons
        if dropout is not None:
            self.d1 = nn.Dropout(dropout)  # do we want dropout?
        print('NN created with:\n', first_layer_neurons, 'neurons on 1st layer\n', final_layer_neurons, 'final neurons\n'
              , dropout, 'dropout')

    def forward(self, x):
        if hasattr(self, 'd1'):
            x = self.d1(torch.relu(self.fc1(x)))
        else:
            x = torch.relu(self.fc1(x))  # torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
# net = Net(first_layer_neurons=100, final_layer_neurons=1, dropout=0.5)


def train_nn(x_emb, y, idx_train=None, idx_test=None, test_size=0.2, random_state=0, class_weight='balanced', dropout=0.5, n_epochs=5000,
             multi_class=True, debug_mode=True):
    ####################################################################################################################
    # 1. split dataset in train/test
    ####################################################################################################################
    torch.manual_seed(random_state)

    if idx_train is not None and idx_test is not None:
        x_train, x_test, y_train, y_test = [x_emb[idx_train], x_emb[idx_test], y[idx_train], y[idx_test]]
        print('x train:', x_train.shape, 'x test:', x_test.shape, 'y train:', y_train.shape, 'y test:', y_test.shape)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_emb, y, test_size=test_size, random_state=random_state)
        print('x train:', x_train.shape, 'x test:', x_test.shape, 'y train:', y_train.shape, 'y test:', y_test.shape)

    final_layer_neurons = y_train.nunique()
    if not multi_class and final_layer_neurons <= 2:  # binary mode
        final_layer_neurons = 1
    else:
        multi_class = True
    print('using mode', ('multiclass' if multi_class else 'binary'))
    first_layer_neurons = x_train.shape[1]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    if multi_class:
        y_train_torch = torch.tensor(y_train.values, dtype=torch.long)
    else:
        y_train_torch = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    ####################################################################################################################
    # 2. build network
    ####################################################################################################################
    net = Net(first_layer_neurons=first_layer_neurons, final_layer_neurons=final_layer_neurons, dropout=dropout)
    if multi_class:
        if class_weight == 'balanced':
            weight = torch.tensor(list(y_train.value_counts() / y_train.count()))
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.BCELoss()  # loss function (binary cross entropy loss or log loss)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.99)  # TODO: move to Adam?
    parameters = filter(lambda p: p.requires_grad, net.parameters())  # We don't want parameters that don't require a grad in the optimizer
    optimizer = optim.Adam(parameters, lr=0.001)
    ####################################################################################################################
    # 3. train
    ####################################################################################################################
    losses, accs, ws, bs = [[], [], [], []]
    for epoch in range(n_epochs):
        net.train()
        optimizer.zero_grad()  # zero the gradients
        train_preds = net(x_train)  # forward
        loss = criterion(train_preds, y_train_torch)  # calculate error
        loss.backward()  # backward
        optimizer.step()  # optimize/Update parameters

        if debug_mode:  # Track the changes - TODO: check tensorboard or similar
            losses, accs, ws, bs = nn_graph_perf(train_preds, y_train, net, loss,
                                                 losses=losses, accs=accs, ws=ws, bs=bs, multi_class=multi_class)

        # print statistics
        if epoch % 500 == 0 or epoch >= n_epochs-1:
            net.eval()
            test_preds = net(x_test)

            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            nn_print_perf(train_preds=train_preds.detach(), y_train=y_train,
                          test_preds=test_preds.detach(), y_test=y_test, multi_class=multi_class)

    print('Finished Training')
    # net.eval()
    if debug_mode:
        plot_multi_lists({'Bias': bs, 'Weight': ws, 'Loss': losses, 'Accuracy': accs})

    preds, df_test, df_train = nn_classification_report(y, train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    return [net, preds, df_test, df_train]
