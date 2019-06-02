from nltk import tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import contractions
import unicodedata



"""
FOR ALL THE FUNCTIONS BELOW:
my_text is either a string or a pd.Series of sentences (1 row = 1 sentence)
"""


def text2sentences(raw_text):
    """

    :param raw_text: string or file of the path containing the text
    :return: dataframe of sentences (1 row = 1 sentence)
    """
    if raw_text.endswith('.txt'):
        with open(raw_text, encoding='utf8') as f:
            raw_text = f.read().strip().replace('\n', '. ')
    # some pre-cleaning before using punkt tokenizer
    raw_text = clean_string(raw_text, remove_punctuation=False)  # we keep punctuation for tokenizing
    sentences = tokenize.sent_tokenize(raw_text)
    sentences = [re.sub(r'[^\w\s]', '', stn) for stn in sentences]  # now cleanup punctuation
    return pd.Series(sentences)


def keywords_filter(my_string, keywords):
    if any(s in my_string for s in list(keywords)):
        return my_string
    else:
        return np.nan


def preprocess_text(my_text, remove_stopwords=False, stemmer=None, lemmatizer=None, keywords=None):
    """

    :param my_text: raw text, can be either string, .txt file containing text or pd.Series of sentences
    :param remove_stopwords: option to remove or keep stopwords (nltk function)
    :param stemmer: can be either porter, snowball, lancaster or None
    :param lemmatizer: for verb lemmatization. can be either wordnet or None
    :param keywords: list of keywords to select sentences of interest
    :return: pd.Series of cleaned sentences
    """
    if isinstance(my_text, str):  # convert text to sentences if not already done
        my_text = text2sentences(my_text)
    else:  # otherwise just clean the text
        my_text = my_text.apply(lambda x: clean_string(x, remove_punctuation=True))
    # keep only sentences with relevant keywords
    if keywords is not None:
        keywords = [x.lower() for x in keywords]
        my_text = my_text.apply(lambda x: keywords_filter(x, keywords=keywords)).dropna()
    # removing stop words
    if remove_stopwords:
        stop = stopwords.words('english')
        my_text = my_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # stemming
    if stemmer is not None:
        my_text = stem_text(my_text, stemmer)
    # lemming with respect to verbs
    if lemmatizer is not None:
        my_text = lemmatize_verbs(my_text, lemmatizer)
    return my_text.replace('', np.nan).dropna()


def clean_string(my_string, remove_punctuation=False):
    my_string = my_string.strip()  # remove leading/trailing characters
    my_string = unicodedata.normalize('NFKD', my_string).\
        encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove non ascii characters
    my_string = my_string.replace('i', 'I')  # to allow expansion of i'm, i've...
    my_string = contractions.fix(my_string)  # expand english contractions (didn't -> did not)
    my_string = my_string.lower()  # to lower case
    my_string = my_string.replace('e.g.', 'exempli gratia')  # replace e.g. (otherwise punkt tokenizer breaks)
    my_string = my_string.replace('i.e.', 'id est')  # replace i.e. (otherwise punkt tokenizer breaks)
    my_string = re.sub(r'\.+', ".", my_string)  # replace multiple dots by single dot
    my_string = my_string.replace('.', '. ').replace('.  ', '. ')  # ensure dots are followed by space for tokenization

    # remove special characters
    if remove_punctuation:
        my_string = re.sub(r'[^\w\s]', ' ', my_string)
    else:  # keep bare minimum punctuation
        my_string = re.sub(r'[^a-zA-Z0-9.,-?!()\s]+', ' ', my_string)
    my_string = ' '.join(my_string.split())  # substitute multiple spaces with single space

    return my_string


def stem_text(my_text, stemmer='snowball'):
    """

    :param my_text: pd.Series of texts
    :param stemmer: type of stemmer to use, can be either porter, snowball or lancaster
    :return: stemmed pd.Series of texts
    """
    stemmer = stemmer.lower()
    if 'porter' in stemmer:
        st = PorterStemmer()
    elif 'snowball' in stemmer:
        st = SnowballStemmer('english')
    elif 'lancaster' in stemmer:
        st = LancasterStemmer()
    else:
        return 'unknow stemmer'
    my_text = my_text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return my_text


def lemmatize_verbs(my_text, lemmatizer='wordnet'):
    lemmatizer = lemmatizer.lower()
    if 'wordnet' in lemmatizer:
        lm = WordNetLemmatizer()
    else:
        return 'unknow lemmatizer'
    my_text = my_text.apply(lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
    return my_text

