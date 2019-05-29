import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# Get torch stuff
import torch
from torch import nn
import torch.optim as optim
matplotlib.use('Qt5Agg')
np.random.seed(42)


# **Activation functions**
def step(z):
    return np.array(z > 0, dtype=np.int32)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


# DATASET:  1000 integers with values from -30 to 40 (hot if >20)
x = np.random.randint(-30, 40, 1000)
y = np.array([1 if v > 20 else 0 for v in x])

# Generate a train/test/dev dataset
inds = np.random.permutation(len(x))
inds_train = inds[0:int(0.8*len(x))]
inds_test = inds[int(0.8*len(x)):int(0.9*len(x))]
inds_dev = inds[int(0.9*len(x)):]
# 80% of the dataset
x_train = x[inds_train]
y_train = y[inds_train]
# 10% of the dataset
x_test = x[inds_test]
y_test = y[inds_test]
# 10% of the dataset
x_dev = x[inds_dev]
y_dev = y[inds_dev]

"""#Convert the inputs to PyTorch"""
x_train = torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

x_dev = torch.tensor(x_dev.reshape(-1, 1), dtype=torch.float32)
y_dev = torch.tensor(y_dev.reshape(-1, 1), dtype=torch.float32)

x_test = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

"""# Build the Neural Network"""
#L1 - 4 Neurons
#L2 - 3 Neurons
#L3 - 1 Neuron
device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 4)  # 1 layer with 1 input (1 variable: temperature), 4 outputs (4 neurons)
        self.fc2 = nn.Linear(4, 3)  # 1 layer with 4 inputs, 3 outputs
        self.fc3 = nn.Linear(3, 1)  # 3 inputs, 1 output (only 1 prediction)

    def forward(self, x):
        x1 = torch.sigmoid(self.fc1(x))  # linear layer receives as input x
        x2 = torch.sigmoid(self.fc2(x1))
        x3 = torch.sigmoid(self.fc3(x2))
        return x3


"""# Initialize the NN, create the criterion (loss function) and the optimizer """
net = Net()  # instantiate the class
criterion = nn.BCELoss()  # loss function (binary cross entropy loss or log loss)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)  # to update the weights
# here optimizer = sigmoid gradient descent with learning rate or 0.0001

"""# Train the NN"""
net.to(device)  # move the network to device (CPU)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_dev = x_dev.to(device)
y_dev = y_dev.to(device)

net.train()
losses = []
accs = []
ws = []
bs = []
for epoch in range(10000):  # do 10,000 epoch
    optimizer.zero_grad()  # zero the gradients
    outputs = net(x_train)  # Forward
    loss = criterion(outputs, y_train)  # Calculate error
    loss.backward()  # Backward
    optimizer.step()  # Optimize/Update parameters

    # Track the changes - This is normally done using tensorboard or similar
    losses.append(loss.item())
    accs.append(sklearn.metrics.accuracy_score([1 if x > 0.5 else 0 for x in outputs.cpu().detach().numpy()],y_train.cpu().numpy()))
    ws.append(net.fc1.weight.cpu().detach().numpy()[0][0])
    bs.append(net.fc1.bias.cpu().detach().numpy()[0])

    # print statistics
    if epoch % 500 == 0:
        acc = sklearn.metrics.accuracy_score([1 if x > 0.5 else 0 for x in outputs.cpu().detach().numpy()],y_train.cpu().numpy())
        print("Epoch: {:4} Loss: {:.5} Acc: {:.3}".format(epoch, loss.item(), acc))

print('Finished Training')

"""# Plot Everything"""
fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.6)
fig.set_size_inches(10, 10)
plt.subplot(2, 2, 1)
sns.lineplot(np.arange(0, len(bs)), bs).set_title("Bias")
plt.subplot(2, 2, 2)
sns.lineplot(np.arange(0, len(ws)), ws).set_title("Weight")
plt.subplot(2, 2, 3)
sns.lineplot(np.arange(0, len(losses)), losses).set_title("loss")
plt.subplot(2, 2, 4)
sns.lineplot(np.arange(0, len(accs)), accs).set_title("accuracy")
fig.show()

net.eval()  # switch network to evaluation mode
net(torch.tensor([22], dtype=torch.float32))  # 22: input temperature
net(torch.tensor([100], dtype=torch.float32))  # how well the network generalizes with never seen temperatures