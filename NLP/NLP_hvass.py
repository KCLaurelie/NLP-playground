# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorlayer as tl
# import NLP.NLP_Utils
# from scipy.spatial.distance import cdist

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import HVASS.imdb as imdb
data_dir = r'C:\Users\K1774755\PycharmProjects\data'

# just do this step once (loading movie comments from imdb database)
imdb.maybe_download_and_extract(download_dir=r'C:\Users\K1774755\PycharmProjects\data')

# tl.files.maybe_download_and_extract(filename='aclImdb_v1.tar.gz',
#                                     url_source='http://ai.stanford.edu/~amaas/data/sentiment/',
#                                     working_directory='data/',
#                                     extract=True)
x_train_text, y_train = imdb.load_data(train=True, download_dir=data_dir)
x_test_text, y_test = imdb.load_data(train=False, download_dir=data_dir)
data_text = x_train_text + x_test_text

text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
text9 = "really bad movie"
text10 = "this sucks but i still liked it"
text11 = "had a great time"
text12 = "do not go watch this movie"
text13 = "eeek"
text14 = "already got tix to watch it again"
text15 = "really awful"
text16 = "not bad"
x_train_text = [text1, text2, text3, text4, text5, text6, text7, text8]
y_train = [1, 1, 0, 0, 0, 0, 0, 0]
x_test_text = [text8, text9, text10, text11, text12, text13, text14, text15, text16]
y_test = [0, 0, 1, 1, 0, 0, 1, 0, 1]
texts = x_train_text + x_test_text
num_words = 100  # use only 100 most popular words from the dataset
max_tokens = 10  # truncate individual texts to only 10 words

# We first convert these texts to arrays of integer-tokens because that is needed by the model
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train_text)
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)
tokenizer.word_index  # inspect vocabulary
np.array(x_train_tokens[1])  # how text 2 has been tokenized

# To input texts with different lengths into the model, we also need to pad and truncate them
num_tokens = np.array([len(i) for i in x_train_tokens])
np.sum(num_tokens < max_tokens) / len(num_tokens)  # check text covered after truncating
pad = 'pre'  # better to add 0 at beginning of sequence
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=pad, truncating='pre')
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating='pre')
x_train_pad.shape

"""
CREATE RECURRENT NEURAL NETWORK
"""
model = Sequential()
"""
The embedding-layer is trained as a part of the RNN and will learn to map words with similar semantic meanings to similar embedding-vectors.
First we define the size of the embedding-vector for each integer-token. 
In this case we have set it to 8, so that each integer-token will be converted to a vector of length 8. 
The values of the embedding-vector will generally fall roughly between -1.0 and 1.0, although they may exceed these
"""
embedding_size = 8
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
# add different layers to network: Gated Recurrent Unit (GRU) to the network.
model.add(GRU(units=16,
              return_sequences=True))  # 1st layer uses 16 output. we need to return sequences of data because the next GRU expects sequences as its input.
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,
                activation='sigmoid'))  # fully-connected / dense layer which computes a value between 0.0 and 1.0 that will be used as the classification output
optimizer = Adam(lr=1e-3)  # learning rate
model.compile(loss='binary_crossentropy',  # compile keras model
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()  # check model
model.fit(x_train_pad, y_train,  # train model
          validation_split=0.05, epochs=3, batch_size=64)
# We can now use the trained model to predict the sentiment for these texts.

result_train = model.evaluate(x_train_pad, y_train)  # test model on train set
result_test = model.evaluate(x_test_pad, y_test)  # test model on test set
print("Accuracy: {0:.2%}".format(result_test[1]))
model.predict(x_test_pad)  # predict new texts
# A value close to 0.0 means a negative sentiment and a value close to 1.0 means a positive sentiment. These numbers will vary every time you train the model

"""
The model cannot work on integer-tokens directly, because they are integer values that may range between 0 and the number of words in our vocabulary, e.g. 10000. 
So we need to convert the integer-tokens into vectors of values that are roughly between -1.0 and 1.0 which can be used as input to a neural network.
This mapping from integer-tokens to real-valued vectors is also called an "embedding". 
It is essentially just a matrix where each row contains the vector-mapping of a single token. 
This means we can quickly lookup the mapping of each integer-token by simply using the token as an index into the matrix. 
The embeddings are learned along with the rest of the model during training.
Ideally the embedding would learn a mapping where words that are similar in meaning also have similar embedding-values. Let us investigate if that has happened here.
First we need to get the embedding-layer from the model
"""
layer_embedding = model.get_layer('layer_embedding')
# We can then get the weights used for the mapping done by the embedding-layer.
weights_embedding = layer_embedding.get_weights()[0]
# Note that the weights are actually just a matrix with the number of words in the vocabulary times the vector length for each embedding.
# That's because it is basically just a lookup-matrix.
token_bad = tokenizer.word_index['bad']
weights_embedding[token_bad]
