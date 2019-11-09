import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from symptoms_classifier.classifiers_utils import nn_classification_report, nn_print_perf, nn_graph_perf, prep_nn_dataset
from code_utils.plot_utils import plot_multi_lists
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_cnn(w2v, sentences, y, idx_train=None, idx_test=None,
              tokenization_type=None, MAX_SEQ_LEN=40, test_size=0.2, random_state=0, dropout=0.5, n_epochs=200, debug_mode=False):

    embeddings, word2id, x_train, y_train, y_train_torch, l_train, mask_train, x_test, y_test, y_test_torch, l_test, mask_test = \
        prep_nn_dataset(w2v=w2v, sentences=sentences, y=y, idx_train=idx_train, idx_test=idx_test, test_size=test_size,
                        tokenization_type=tokenization_type, MAX_SEQ_LEN=MAX_SEQ_LEN, random_state=random_state)

    class CNN(nn.Module):
        def __init__(self, embeddings):
            super(CNN, self).__init__()
            vocab_size = embeddings.shape[0]
            embedding_size = embeddings.shape[1]
            # Initialize embeddings
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.embeddings.load_state_dict({'weight': embeddings})
            self.embeddings.weight.requires_grad = False  # disable training for the embeddings
            n_filters = 128  # Set the number of filters
            #  3 different kernel sizes (to have patterns of 3, 4 and 2 words)
            k1 = (3, embedding_size)
            k2 = (4, embedding_size)
            k3 = (2, embedding_size)
            # convolutional layers
            self.conv1 = nn.Conv2d(1, n_filters, k1)  # nb of channels always 1 in text
            self.conv2 = nn.Conv2d(1, n_filters, k2)
            self.conv3 = nn.Conv2d(1, n_filters, k3)
            # fully connected network: concatenate the 3 conv layers and put through linear layer (size = 3*n_filters)
            self.fc1 = nn.Linear(3 * n_filters, 2)
            self.d1 = nn.Dropout(dropout)

        def conv_block(self, input, conv):
            out = conv(input)  # conv function
            out = F.relu(out.squeeze(3))  # activation function
            out = F.max_pool1d(out, out.size()[2]).squeeze(2)  # max pooling
            return out

        def forward(self, x, lns=0):
            x = self.embeddings(x)  # x.shape = batch_size x sequence_length x emb_size
            x = x.unsqueeze(1)  # Because the expected shape = batch_size x channels x sequence_length x emb_size
            x1 = self.conv_block(x, self.conv1)  # Get the output from conv layer 1
            x2 = self.conv_block(x, self.conv2)  # Get the output from conv layer 2
            x3 = self.conv_block(x, self.conv3)  # Get the output from conv layer 3
            x_all = torch.cat((x1, x2, x3), 1)  # concatenate 3 outputs for the 3 conv blocks
            x_all = self.d1(x_all)  # dropout
            logits = self.fc1(x_all)  # run through fc1
            return logits

    cnn = CNN(embeddings)
    parameters = filter(lambda p: p.requires_grad, cnn.parameters())  # We don't want parameters that don't require a grad in the optimizer
    optimizer = optim.Adam(parameters, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses, accs, ws, bs = [[], [], [], []]
    for epoch in range(n_epochs):
        cnn.train()
        optimizer.zero_grad()
        train_preds = cnn(x_train, l_train)
        loss = criterion(train_preds, y_train_torch)
        loss.backward()
        optimizer.step()

        if debug_mode:  # Track the changes
            losses, accs, ws, bs = nn_graph_perf(train_preds, y_train, cnn, loss,
                                                 losses=losses, accs=accs, ws=ws, bs=bs)

        if epoch % 10 == 0 or epoch >= n_epochs-1:
            cnn.eval()
            outputs_train = torch.max(train_preds, 1)[1]
            outputs_test = torch.max(cnn(x_test, l_test), 1)[1]
            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            nn_print_perf(train_preds=outputs_train.detach(), y_train=y_train,
                          test_preds=outputs_test.detach(), y_test=y_test)

    print('Finished Training')
    cnn.eval()
    # logits_test = F.softmax(cnn(x_test), dim=1)
    outputs_test = torch.max(cnn(x_test, l_test), 1)[1]
    # logits_train = F.softmax(cnn(x_train), dim=1)
    outputs_train = torch.max(cnn(x_train, l_train), 1)[1]

    if debug_mode:
        plot_multi_lists({'Bias': bs, 'Weight': ws, 'Loss': losses, 'Accuracy': accs})
    preds, df_test, df_train = nn_classification_report(y, outputs_train, y_train, outputs_test, y_test)

    return [cnn, preds, df_test, df_train]

