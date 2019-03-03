from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews
import os
data_folder = r"""C:\Users\K1774755\King's College London\Cognitive Impairment in Schizophrenia - Documents\Courses\CUSMUMH\week 7 - NLP with nltk & spacy"""
pycharm_folder = r'C:\Users\K1774755\PycharmProjects\toy-models\NLP'

# Let's generate word vectors over the Brown corpus text.
# We will have 20 dimensions, using a window of five for the context words in the skip-grams
# (e.g. c1, c2, w, c3, c4).
# This might be a little slow (maybe 1-2 minutes).

# for the Brown corpus
b = Word2Vec(brown.sents(), size=400, window=10, min_count=5)
# for the movie review corpus
mr = Word2Vec(movie_reviews.sents(), size=20, window=5, min_count=3)

#Now we have the vectors, we can see how good they are by measuring which words are similar to each other.
b.wv.most_similar('company', topn=5)
mr.wv.most_similar('love', topn=5)
#Try altering the window and the dimension size, to see if you get better results.

#We can also do some arithmetic with the words. Let's try that classical result, king - man + woman.

b.wv.most_similar(positive=['biggest', 'small'], negative=['big'], topn=5)

#We can then load these in using Gensim; they might take a minute to load.
from gensim.models.keyedvectors import KeyedVectors
glove = KeyedVectors.load_word2vec_format(os.path.join(pycharm_folder,'glove.twitter.27B.25d.txt.bz2'), binary=False)
print("Done loading")

#Can you find any cool word combinations? What differences are there in the datasets?
glove.most_similar('company', topn=5)
glove.most_similar(positive=['biggest', 'small'], negative=['big'], topn=5) #hoping to get smallest
glove.most_similar(positive=['woman', 'king'], negative=['man'])
glove.similarity('car', 'bike')
glove.doesnt_match("breakfast cereal dinner lunch".split())