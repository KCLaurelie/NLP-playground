import logging
import torch
import torch.nn as nn
import torch.optim as optim
from symptoms_classifier.classifiers_utils import nn_classification_report, nn_print_perf, nn_graph_perf, prep_nn_dataset
from code_utils.plot_utils import plot_multi_lists

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def test():
    from symptoms_classifier.symptoms_classifier import TextsToClassify
    from symptoms_classifier.NLP_embedding import load_embedding_model
    MAX_SEQ_LEN, tokenization_type, test_size, random_state, dropout, n_epochs, debug_mode, rnn_type, lr, bid, simulate_attn, num_layers, hidden_size = \
        [40, None, 0.2, 0, 0.5, 1, False, 'LSTM', 0.001, False, False, 2, 300]
    tweets = TextsToClassify(filepath=r'C:\Users\K1774755\Downloads\phd\Tweets.csv', already_split=False,
                             class_col='airline_sentiment', text_col='text', binary_main_class='positive')
    df = tweets.load_data()
    idx_train, idx_test = tweets.get_train_test_split()
    sentences, y = [df.text, df.airline_sentiment.replace({'neutral': 0, 'positive': 0, 'negative': 1})]
    w2v = load_embedding_model(r'C:\Users\K1774755\PycharmProjects\toy-models\embeddings\w2v_wiki.model', model_type='w2v')


def train_rnn(w2v, sentences, y,
              idx_train=None, idx_test=None, test_size=0.2,
              tokenization_type=None, MAX_SEQ_LEN=40, random_state=0, dropout=0.5,
              num_layers=2, hidden_size=300, n_epochs=200, lr=0.001, debug_mode=False, rnn_type='RNN',
              bid=False, simulate_attn=False):
    """
    :param w2v: pre-trained word2vec embedding model
    :param sentences: pandas Series of sentences to classify (1 row = 1 sentence)
    :param y: corresponding classes
    :param idx_test: indices of testing set in case already split
    :param idx_train: indices of training set in case already split
    :param tokenization_type: (clean, lem, lem_stop, None) preprocessing to apply to text
    :param MAX_SEQ_LEN: max number of words to use
    :param test_size: (float between 0 and 1) proportion of sentences to use for testing
    :param random_state: initialize random state
    :param dropout: dropout rate
    :param num_layers: nb of layers in the neural net
    :param hidden_size: size of hidden layer
    :param n_epochs: number of epochs for training
    :param lr: learning rate for Adam optimizer
    :param debug_mode: (True, False): set to True to display intermediate performance stats and graphs
    :param rnn_type: (RNN, LSTM, GRU): type of network to use
    :param bid: (True, False) bidirectional network (only applies to LSTM)
    :param simulate_attn: simulates attention by stopping bidirectional LSTM halfway
    :return:
    """
    torch.manual_seed(random_state)

    embeddings, word2id, x_train, y_train, y_train_torch, l_train, mask_train, x_test, y_test, y_test_torch, l_test, mask_test = \
        prep_nn_dataset(w2v=w2v, sentences=sentences, y=y, idx_train=idx_train, idx_test=idx_test, test_size=test_size,
                        tokenization_type=tokenization_type, MAX_SEQ_LEN=MAX_SEQ_LEN, random_state=random_state)

    class RNN(nn.Module):
        def __init__(self, embeddings, padding_idx):
            super(RNN, self).__init__()
            vocab_size = len(embeddings)
            embedding_size = len(embeddings[0])

            # Initialize embeddings
            self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Create embeddings
            self.embeddings.load_state_dict({'weight': embeddings})  # load existing weights
            self.embeddings.weight.requires_grad = False  # Disable training for the embeddings - IMPORTANT

            # Create the RNN cell
            if rnn_type.lower() == 'gru':
                self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            elif rnn_type.lower() == 'lstm':
                print('running bidirectional LSTM') if bid else print('running standard LSTM')
                self.rnn = nn.LSTM(input_size=embedding_size,
                                   hidden_size=hidden_size // (2 if bid else 1),
                                   num_layers=num_layers, dropout=dropout, bidirectional=bid)
            else:
                self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            self.fc1 = nn.Linear(hidden_size, 2)
            self.d1 = nn.Dropout(0.5)

        def forward(self, x, lns, mask):
            if debug_mode: print('start forward function - x: ', x.shape)
            x = self.embeddings(x)  # x.shape = batch_size x sequence_length x emb_size
            if debug_mode: print('after embedding - x: ', x.shape)
            # Tell RNN to ignore padding and set the batch_first to True (sequence length 1st for historical reasons)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True, enforce_sorted=False)
            x, hidden = self.rnn(x)  # run 'x' through the RNN
            x, hidden = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # Add the padding again

            # select the value at the length of that sentence (we are only interested in last output) or middle if bidirectional
            row_indices = torch.arange(0, x.size(0)).long()
            print('type and shape of x before output selection', type(x), x.shape)
            if bid:
                if simulate_attn:
                    x = x[row_indices, lns / 2, :]
                else:
                    x = x[row_indices, lns - 1, :] # torch.cat((x[row_indices, lns - 1, :], x[row_indices, 0, :]), 0)
            else:
                x = x[row_indices, lns - 1, :]
            print('type and shape of x', type(x), x.shape)
            x = self.d1(x) # dropout
            x = self.fc1(x)
            return x

    rnn = RNN(embeddings, padding_idx=word2id['<PAD>'])
    parameters = filter(lambda p: p.requires_grad, rnn.parameters())  # We don't want parameters that don't require a grad in the optimizer
    optimizer = optim.Adam(parameters, lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses, accs, ws, bs, best_f1, best_epoch, best_train_preds, best_test_preds = [[], [], [], [], 0, 0, None, None]
    for epoch in range(n_epochs):
        rnn.train()
        optimizer.zero_grad()
        train_preds = rnn(x_train, l_train, mask_train)
        loss = criterion(train_preds, y_train_torch)
        loss.backward()
        optimizer.step()

        if debug_mode:  # Track the changes
            losses, accs, ws, bs = nn_graph_perf(train_preds, y_train, rnn, loss,
                                                 losses=losses, accs=accs, ws=ws, bs=bs)

        if epoch % 1 == 0 or epoch >= n_epochs-1:
            rnn.eval()
            outputs_train = torch.max(train_preds, 1)[1]
            outputs_test = torch.max(rnn(x_test, l_test, mask_test), 1)[1]
            print("Epoch: {:4} Loss: {:.5f} ".format(epoch, loss.item()))
            f1 = nn_print_perf(train_preds=outputs_train.detach(), y_train=y_train,
                               test_preds=outputs_test.detach(), y_test=y_test)
            if f1 > best_f1:
                best_f1, best_epoch, best_train_preds, best_test_preds = f1, epoch, outputs_train, outputs_test

    print('Finished Training, best F1 obtained on test set:', best_f1, 'at epoch', best_epoch)
    if debug_mode:
        plot_multi_lists({'Bias': bs, 'Weight': ws, 'Loss': losses, 'Accuracy': accs})
    preds, df_test, df_train = nn_classification_report(y, best_train_preds, y_train, best_test_preds, y_test)

    return [rnn, preds, df_test, df_train]
