from collections import Counter
import pickle  # to save models
from nltk import tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
from collections import defaultdict
from symptoms_classifier.NLP_text_cleaning import clean_string, text2sentences, preprocess_text


"""
FOR ALL THE FUNCTIONS BELOW:
my_text is a pd.Series of sentences (1 row = 1 sentence)
"""


def test():
    data = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
    data_clean = data[['airline_sentiment', 'text']].rename(columns={'airline_sentiment': 'class'})
    txt = data_clean['text'][0:10]
    raw_text = "hi my name is link...I like to fight, And i'm in love with princess zelda.bim. bam.Boum. Bom"
    raw_text = clean_string(raw_text)
    raw_text = text2sentences(raw_text)

    txt = 'C:\\temp\\bla.txt'
    text2sentences(txt)

    clean_text = preprocess_text(txt)
    preprocess_text(txt, remove_stopwords=True, stemmer='snowball', lemmatizer=None)
    #vocab = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    w2v = fit_text2vec(clean_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100)
    list(w2v.vocabulary_.keys())[:10]
    processed_features = transform_text2vec(clean_text, w2v, algo='tfidf')

    w2v = fit_text2vec(clean_text, min_df=0.00125, max_df=0.7, algo='word2vec', _size=100)
    list(w2v.wv.vocab)
    return 0


def top_features(vectorizer, top_n=2):
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    _top_features = [features[i] for i in indices[:top_n]]
    return _top_features

# TODO check http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XPFMnohKiUk

def transform_text2vec(my_text, w2v_model, algo='tfidf', _size=100, load_vectorizer=False):
    """

    :param my_text: pd.Series of texts (1 row = 1 sentence)
    :param w2v_model: pre-trained embedding model to use
    :param algo: embedding model type (can be either tfidf, counter or word2vec)
    :param _size: size of vector desired
    :param load_vectorizer: option to load pre-trained model from file
    :return: vectorized text
    """
    algo = str(algo).lower()
    if any(substring in algo for substring in ('idf', 'count')):
        if load_vectorizer: w2v_model = pickle.load(open("vectorizer.pickle"), "rb")
        vectors = w2v_model.transform(my_text).toarray()
    elif 'word2vec' in algo:
        if load_vectorizer: w2v_model = Word2Vec.load("word2vec.model")
        sentences = my_text.to_list()
        vectors = np.zeros((len(sentences), _size))

        # TODO
        # shall i do that?
        for idx, snt in enumerate(sentences):
            vectors[idx] = [w2v_model.wv.get_vector(x) for x in tokenize.word_tokenize(snt)]
        vectors = np.zeros((len(sentences), _size))

        # or that???
        """# Convert each sentence into the average sum of its tokens"""
        # Loop over sentences
        for i_snt, snt in enumerate(sentences):
            cnt = 0
            for i_word, word in enumerate(snt):  # Loop over the words of a sentence
                if word in w2v_model.wv:
                    vectors[i_snt] += w2v_model.wv.get_vector(word)
                    cnt += 1
            if cnt > 0:
                vectors[i_snt] = vectors[i_snt] / cnt
            i_snt += 1
    else:
        return 'unknown algo'
    return vectors


def fit_text2vec(my_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100, save_vectorizer=False):
    """

    :param my_text: pd.Series of texts (1 row = 1 sentence)
    :param min_df: ignore words below that frequency (used for tfidf and couter)
    :param max_df: ignore words above that frequency (used for tfidf and couter)
    :param algo: embedding model type (can be either tfidf, counter or word2vec)
    :param _size: desired vector size
    :param save_vectorizer: option to save trained model
    :return: word embeddings
    """

    algo = str(algo).lower()
    stop_words = stopwords.words('english')
    if 'idf' in algo.lower():
        _vectorizer = text.TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            use_idf=True,
            analyzer='word',
            ngram_range=(1, 5),
            stop_words=stop_words
        )
        w2v = _vectorizer.fit(my_text)
    elif 'count' in algo:
        _vectorizer = CountVectorizer(max_features=2500, min_df=min_df, max_df=max_df, stop_words=stop_words)
        w2v = _vectorizer.fit(my_text)
        if save_vectorizer: pickle.dump(w2v, open("vectorizer.pickle", "wb"))
    elif 'word2vec' in algo:
        if isinstance(my_text, str):
            sentences = tokenize.sent_tokenize(my_text)
        else:
            sentences = my_text.to_list()
        vocab = []
        for snt in sentences:
            vocab.append(tokenize.word_tokenize(snt))
        w2v = Word2Vec(vocab, min_count=1, size=_size)  #, window=6, min_count=5, workers=4)
        if save_vectorizer: w2v.save("word2vec.model")
    else:
        return 'unknown algo'
    return w2v


