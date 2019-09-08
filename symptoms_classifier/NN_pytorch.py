import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import symptoms_classifier.classifiers_utils as cutils
from code_utils.plot_utils import plot_multi_lists
np.random.seed(42)


def train_nn_simple(x_emb, y, test_size=0.2, random_state=0, class_weight='balanced', dropout=0.5, n_epochs=5000, debug_mode=True):

    x_train, x_test, y_train, y_test = train_test_split(x_emb, y, test_size=test_size, random_state=random_state)

    """#Convert the inputs to PyTorch"""
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    first_layer_neurons = x_train.shape[1]

    """# Initialize the NN, create the criterion (loss function) and the optimizer """
    net = cutils.nn_create(nb_classes=1, first_layer_neurons=first_layer_neurons, dropout=dropout)
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
            cutils.nn_print_perf(train_preds=train_preds.detach(), y_train=y_train,
                                 test_preds=test_preds.detach(), y_test=y_test, multi_class=False)
    print('Finished Training')

    if debug_mode:
        plot_multi_lists({'Bias': bs, 'Weight': ws, 'Loss': losses, 'Accuracy': accs})

    preds, df_test, df_train = cutils.nn_classification_report(y, train_preds.detach(), y_train, test_preds.detach(), y_test, multi_class=False)

    #net(torch.tensor([22], dtype=torch.float32))  # 22: input temperature

    return [net, preds, df_test, df_train]
