from code_utils.global_variables import *
from symptoms_classifier.NLP_embedding import clean_and_embed_text
import logging
import torch
import torch.nn as nn
import spacy
from sklearn.model_selection import train_test_split
import torch.optim as optim
import sklearn.metrics
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load(spacy_en_path, disable=['ner', 'parser'])
BATCH_SIZE = 100
DEVICE = torch.device("cpu")


def train_lstm(w2v, sentences, y, tokenization_type, keywords, ln=15, test_size=0.2, random_state=0, class_weight='balanced', dropout=0.5, n_epochs=5000):
    EMB_SIZE = w2v.wv.vector_size
    emb_weights, x_ind, c_ind = clean_and_embed_text(sentences=sentences, w2v=w2v, tokenization_type=tokenization_type, keywords=keywords, ln=ln)

    x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(x_ind, y, c_ind, test_size=test_size, random_state=random_state)
    x_test = torch.tensor(x_test).to(DEVICE)
    y_test = torch.tensor(y_test).to(DEVICE)

    class LSTM(nn.Module):
        def __init__(self, emb_weights, vocab_size, emb_size, kernel_sizes=[4, 3, 3]):
            super(LSTM, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, emb_size)
            self.embeddings.load_state_dict({'weight': torch.tensor(emb_weights, dtype=torch.float32)})
            self.embeddings.weight.requires_grad = False
            self.hidden_size = 64
            self.num_layers = 2

            self.lstm = nn.LSTM(input_size=emb_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=0.3,
                                bidirectional=True)
            self.fc = nn.Linear(self.hidden_size * self.num_layers, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, inds):
            x = self.embeddings(x)
            x = x.permute(1, 0, 2)

            output, (h_n, c_n) = self.lstm(x)
            output = self.dropout(output)
            tmp = [output[j, i, :] for i, j in enumerate(inds)]
            fin = torch.stack(tmp)
            # Do the thing
            logits = self.fc(fin)
            return logits

    lstm = LSTM(emb_weights, len(emb_weights), EMB_SIZE)
    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]))

    n_batches = len(x_train) // BATCH_SIZE
    for epoch in range(50):
        lstm.eval()
        print("TEST")
        outputs = torch.max(lstm(x_test, ind_test), 1)[1]
        al = sklearn.metrics.classification_report(outputs.detach().numpy(), y_test.detach().numpy())
        f1 = sklearn.metrics.f1_score(outputs.detach().numpy(), y_test.detach().numpy())
        print(al)
        print(f1)

        print("TRAIN")
        outputs = torch.max(lstm(torch.tensor(x_train), ind_train), 1)[1]
        al = sklearn.metrics.classification_report(outputs.detach().numpy(), y_train)
        f1 = sklearn.metrics.f1_score(outputs.detach().numpy(), y_train)
        print(al)
        print(f1)
        loss = 0
        for b_ind in range(n_batches):
            lstm.train()
            start = b_ind * BATCH_SIZE
            end = (b_ind + 1) * BATCH_SIZE
            x_batch = torch.tensor(x_train[start:end]).to(DEVICE)
            y_batch = torch.tensor(y_train[start:end]).to(DEVICE)
            ind_batch = torch.tensor(ind_train[start:end]).to(DEVICE)

            optimizer.zero_grad()
            outputs = lstm(x_batch, ind_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            loss += loss.item()
            optimizer.step()
        print(loss / n_batches)
