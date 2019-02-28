#This notebook uses a popular neural network API, Keras, to build a simple CNN classifer,
# and runs it over movie reviews from IMDb - the Internet Movie Database.
# These reviews are available as a pre-prepared dataset that can be downloaded by the Keras distribution.
#The dataset is constructed from very polarised reviews, and has been used in text classification evaluations for several years.

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# For displaying
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt

# For processing example texts into one-hot vectors
import nltk
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing import text

#####################################
# 1.PARAMETERS
#####################################
# let's set up some parameters, such as number of features, embedding dimensions, batch size, epochs etc.

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

#####################################
# 2.DATA
#####################################
# Let's load the data, and pad it out so all are the same length.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#####################################
# 3.BUILDING THE MODEL
#####################################
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# let's take a look at the model
print(model.summary())
SVG(model_to_dot(model).create(prog='dot', format='svg'))

#####################################
# 4. TRAIN THE MODEL
#####################################
# Now let's train it. Keras will validate against our test data, showing us loss and accuracy as it goes.
# We will save our metrics so we can display them afterwards.

history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#####################################
# 5. VISUALIZE TRAINING PROCESS
#####################################
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#####################################
# 6. USE THE MODEL
#####################################
example = 'A truly awful movie. I never want to see rubbish like this again.'


# Download NLTK stopwords
nltk.download('stopwords')

# Prepare the stopwords
stopwords_nltk = set(stopwords.words('english'))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))

# Remove the stop words from input text
example = ' '.join([word for word in example.split() if word not in stopwords_filtered])

# One-hot the input text
example = text.one_hot(example, max_features)
example = np.array(example)

# Pad the sequences
example = sequence.pad_sequences([example], maxlen=maxlen)

# Make the prediction
pred_prob = model.predict(example)[0][0]
pred_class = model.predict_classes(example)[0][0]

print("Probability: ", pred_prob)
print("Class: ", pred_class)