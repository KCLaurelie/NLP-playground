from collections import Counter
from nltk import tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import re
import contractions
import unicodedata


"""
FOR ALL THE FUNCTIONS BELOW:
my_text is a pd.Series of sentences (1 row = 1 sentence)
"""


def test():
    data = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
    data_clean = data[['airline_sentiment', 'text']].rename(columns={'airline_sentiment': 'class'})
    my_text = data_clean['text'][0:10]
    raw_text = "hi my name is link...I like to fight, And i'm in love with princess zelda.bim. bam.Boum. Bom"
    raw_text = clean_string(raw_text)
    raw_text = text2sentences(raw_text)

    my_text = preprocess_text(my_text)
    #vocab = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    w2v = fit_text2vec(my_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100)
    w2v = fit_text2vec(my_text, min_df=0.00125, max_df=0.7, algo='word2vec', _size=100)
    list(w2v.wv.vocab)
    return 0


def text2sentences(raw_text):
    """

    :param raw_text: string or file of the path containing the text
    :return: dataframe of sentences (1 row = 1 sentence)
    """
    if raw_text.endswith('.csv'):
        raw_text = pd.read_csv(raw_text)

    # some pre-cleaning before using punkt sentence tokenizer
    raw_text = clean_string(raw_text, remove_punctuation=False)

    sentences = tokenize.sent_tokenize(raw_text)
    return pd.Series(sentences)


def stem_text(my_text, stemmer='snowball'):
    stemmer = stemmer.lower()
    if stemmer == 'porter':
        st = PorterStemmer()
    elif stemmer == 'snowball':
        st = SnowballStemmer('english')
    elif stemmer == 'lancaster':
        st = LancasterStemmer()
    else:
        return 'unknow stemmer'
    my_text = my_text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return my_text


def lemmatize_verbs(my_text, lemmatizer='wordnet'):
    lemmatizer = lemmatizer.lower()
    if lemmatizer == 'wordnet':
        lm = WordNetLemmatizer()
    else:
        return 'unknow lemmatizer'
    my_text = my_text.apply(lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
    return my_text


def clean_string(my_string, remove_punctuation=False):
    my_string = my_string.strip()  # remove leading/trailing characters
    my_string = my_string.lower()  # to lower case
    my_string = my_string.replace('e.g.', 'exempli gratia')  # replace e.g. (otherwise punkt tokenizer breaks)
    my_string = my_string.replace('i.e.', 'id est')
    my_string = re.sub(r'\.+', ".", my_string)  # replace multiple dots by single dot
    my_string = my_string.replace('.', '. ').replace('.  ', '. ')  # ensure dots are followed by space
    my_string = contractions.fix(my_string)  # expand english contractions (didn't -> did not)
    my_string = unicodedata.normalize('NFKD', my_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove non ascii characters
    # remove special characters
    if remove_punctuation:
        my_string = re.sub(r'[^\w\s]', '', my_string)
    else:
        re.sub(r'[^a-zA-Z0-9.,-?!\s]+', ' ', my_string)
    my_string = ' '.join(my_string.split())  # substitute multiple spaces with single space
    return my_string


def preprocess_text(my_text, clean_strings=True, remove_stopwords=False, stemmer=None, lemmatizer=None):  # text = series of texts
    if clean_strings:
        my_text = my_text.apply(lambda x: clean_string(x, remove_punctuation=True))
    # remove stop words
    if remove_stopwords:
        stop = stopwords.words('english')
        my_text = my_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # stemming
    if stemmer is not None:
        my_text = stem_text(my_text, stemmer)
    # lemming with respect to verbs
    if lemmatizer is not None:
        my_text = lemmatize_verbs(my_text, stemmer)
    return my_text


def transform_text2vec(my_text, w2v_model, algo='tfidf', _size=100):
    if algo in ('tfidf', 'counter'):
        vectors = w2v_model.transform(my_text).toarray()
    elif algo == 'word2vec':
        sentences = my_text.to_list()
        vectors = np.zeros((len(sentences), _size))

        # shall i do that?
        for idx, snt in enumerate(sentences):
            vectors[idx] = [w2v_model.wv.get_vector(x) for x in tokenize.word_tokenize(snt)]
        vectors = np.zeros((len(sentences), _size))

        # or that???
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


def fit_text2vec(my_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100):
    stop_words = stopwords.words('english')
    if algo == 'tfidf':
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
    elif algo == 'counter':
        _vectorizer = CountVectorizer(max_features=2500, min_df=min_df, max_df=max_df, stop_words=stop_words)
        w2v = _vectorizer.fit(my_text)
    elif algo == 'word2vec':
        sentences = my_text.to_list()  # tokenize.sent_tokenize(texts.str.cat(sep='. '))
        vocab = []
        for snt in sentences:
            vocab.append(tokenize.word_tokenize(snt))
        w2v = Word2Vec(vocab, min_count=1, size=_size)  #, window=6, min_count=5, workers=4)
    else:
        return 'unknown algo'
    return w2v


def perf_metrics(data_labels, data_preds):
    data_labels = pd.Series(data_labels)
    data_preds = pd.Series(data_preds)
    labels = list(data_labels.unique())
    #labels = [1, -1]
    acc_score = accuracy_score(data_labels, data_preds)
    precision = precision_score(data_labels, data_preds, average=None, labels=labels)
    recall = recall_score(data_labels, data_preds, average=None, labels=labels)
    f1score = f1_score(data_labels, data_preds, average=None, labels=labels)
    return acc_score, precision, recall, f1score


def all_words(raw_text):
    return re.findall('\\w+', raw_text.lower())


def prob(word, word_dic_file):
    # word_dic_file = 'constants/big.txt'
    """Probability of `word`."""
    word_dic = Counter(all_words(open(word_dic_file).read()))
    N = sum(word_dic.values())
    return word_dic[word] / N


def correction(word):
    """Most probable spelling correction for word."""
    return max(candidates(word), key=P)


def candidates(word):
    """Generate possible spelling corrections for word."""
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words, word_dic):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set((w for w in words if w in word_dic))


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [ (word[:i], word[i:]) for i in range(len(word) + 1) ]
    deletes = [ L + R[1:] for L, R in splits if R ]
    transposes = [ L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1 ]
    replaces = [ L + c + R[1:] for L, R in splits if R for c in letters ]
    inserts = [ L + c + R for L, R in splits for c in letters ]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
