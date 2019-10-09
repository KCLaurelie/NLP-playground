from symptoms_classifier.NLP_embedding import *
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sklearn.metrics
from symptoms_classifier.classifiers_utils import nn_classification_report, nn_print_perf, nn_graph_perf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def test():
    from symptoms_classifier.symptoms_classifier import *
    MAX_SEQ_LEN, tokenization_type, test_size, random_state, dropout, n_epochs = [40, None, 0.2, 0, 0.5, 5000]
    tweets = TextsToClassify(filepath=r'C:\Users\K1774755\Downloads\phd\Tweets.csv',
                             class_col='airline_sentiment', text_col='text', binary_main_class='positive')
    df = tweets.load_data()
    sentences, y = [df.text, df.airline_sentiment.replace({'neutral':0, 'positive':0, 'negative':1})]
    w2v = load_embedding_model(r'C:\Users\K1774755\PycharmProjects\toy-models\embeddings\w2v_wiki.model', model_type='w2v')


def train_cnn(w2v, sentences, y, tokenization_type=None, MAX_SEQ_LEN=40, test_size=0.2, random_state=0, dropout=0.5, n_epochs=5000):
    embeddings_res = embedding2torch(w2v, SEED=0)
    embeddings = embeddings_res['embeddings']
    word2id = embeddings_res['word2id']
    x_ind, prim_len = words2integers(raw_text=sentences, word2id=word2id, tokenization_type=tokenization_type, MAX_SEQ_LEN=MAX_SEQ_LEN)
    # emb_weights, x_ind, c_ind = embed_text_with_padding(sentences=sentences, w2v=w2v, tokenization_type=tokenization_type, keywords=keywords, ln=ln)

    x_train, x_test, y_train, y_test, l_train, l_test = train_test_split(x_ind, y, prim_len,
                                                                         test_size=test_size, random_state=random_state)

    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    l_train = torch.tensor(l_train, dtype=torch.float32).reshape(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    l_test = torch.tensor(l_test, dtype=torch.float32).reshape(-1, 1)

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

    for epoch in range(n_epochs):
        cnn.train()
        optimizer.zero_grad()
        outputs = cnn(x_train, l_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            cnn.eval()
            outputs_train = torch.max(cnn(x_train, l_train), 1)[1]
            outputs_test = torch.max(cnn(x_test, l_test), 1)[1]
            al = sklearn.metrics.classification_report(outputs_test.detach().numpy(), y_test.detach().numpy())
            f1 = sklearn.metrics.f1_score(outputs_test.detach().numpy(), y_test.detach().numpy())
            print(al)
            print(f1)

    cnn.eval()
    logits_test = F.softmax(cnn(x_test), dim=1)
    outputs_test = torch.max(cnn(x_test), 1)[1]
    logits_train = F.softmax(cnn(x_train), dim=1)
    outputs_train = torch.max(cnn(x_train), 1)[1]
    preds, df_test, df_train = nn_classification_report(y, outputs_train, y_train, outputs_test, y_test)

    return [cnn, preds, df_test, df_train]


def test_CNN(sentence, w2v, tokenization_type, cnn):
    embeddings_res = embedding2torch(w2v, SEED=0)
    word2id = embeddings_res['word2id']
    x_ind = words2integers(raw_text=sentence, word2id=word2id, tokenization_type=tokenization_type)
    x_ind = torch.tensor([x_ind], dtype=torch.long)
    cnn = cnn.eval()
    res_tmp = cnn(x_ind)  # output of the netweork
    res = torch.softmax(res_tmp, dim=1)
    return res
