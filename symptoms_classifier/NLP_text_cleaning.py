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
my_text can either be a string or a pd.Series of sentences (1 row = 1 sentence)
"""


def text2sentences(raw_text):
    """
    converts text into a series of sentences
    :param raw_text: string or file of the path containing the text
    :return: pd.Series of sentences (1 row = 1 sentence)
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
    """
    checks if a string contains any keywords from a list, and returns the string if so
    :param my_string: string to check
    :param keywords: list of keywords
    :return: original string if it contains any keywords from the string, NaN if not
    """
    if any(s in my_string for s in list(keywords)):
        return my_string
    else:
        return np.nan


def find_with_context(raw_text, keywords, context_length=20, context_type='portion', keyword_search='together'):
    """
    extracts portion of text containing specific keyword
    :param raw_text: string or file of the path containing the text
    :param keywords: the list of keywords that has to be contained in the text portion
    :param context_length: the size of the portion of text (each side) in number of words to extract
                            e.g. context_length = 3 means we want to retrieve 3 words surrounding the keyword
    :param context_type: option to return either the portion of text surrounding the keyword ('portion')
                        or extract the sentences ('sentence')
    :param keyword_search: ways to search for keywords
                            together: if 2 keywords are in the same text portion, will only return the joint portion
                            separate: if 2 keywords are in the same text portion, will return 2 portions
    :return: series of portions of text that contain the keyword
    """
    # convert keywords input to list
    keywords = [keywords.lower()] if isinstance(keywords, str) else [x.lower() for x in keywords]

    # read text from file if applicable
    if raw_text.endswith('.txt'):
        with open(raw_text, encoding='utf8') as f:
            raw_text = f.read().strip().replace('\n', '. ')

    # now extract portions/sentences of text containing the keywords
    if 'sentence' in context_type.lower():  # extract sentences containing the keywords
        # res = re.findall(r"([^.]*?{}[^.]*\.)".format(keyword), raw_text.lower())
        if keyword_search == 'together':
            res = [s + '.' for s in raw_text.split('.') if any(key in s.lower() for key in keywords)]  # faster than regex
        else:
            res = []
            for key in keywords:
                res.extend([s + '.' for s in raw_text.split('.') if key in s.lower()])  # this seems faster than regex
    else:  # extract portions of text containing the keywords
        if keyword_search == 'together':
            key = '|'.join(list(keywords))
            regex = r'\w*\W*' * context_length + key + r'\w*\W*' * context_length
            res = re.findall(regex, raw_text, re.I)
        else:
            res = []
            for key in keywords:
                regex = r'\w*\W*'*context_length + key + r'\w*\W*'*context_length
                res.extend(re.findall(regex, raw_text, re.I))
    return pd.Series(res)


def preprocess_text(my_text, remove_stopwords=False, stemmer=None, lemmatizer=None, keywords=None):
    """
    cleans text and outputs series of pre-processed sentences
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
    """

    :param my_string: string to clean
    :param remove_punctuation: select True to remove all punctuation/special characters from the text.
    otherwise only special characters will be removed.
    :return: clean string, ready to be tokenized
    """
    my_string = my_string.strip()  # remove leading/trailing characters
    my_string = unicodedata.normalize('NFKD', my_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove non ascii characters
    my_string = my_string.replace('i', 'I')  # to allow expansion of i'm, i've...
    my_string = contractions.fix(my_string)  # expand english contractions (didn't -> did not)
    my_string = my_string.lower()  # to lower case
    my_string = my_string.replace('e.g.', 'exempli gratia')  # replace e.g. (otherwise punkt tokenizer breaks)
    my_string = my_string.replace('i.e.', 'id est')  # replace i.e. (otherwise punkt tokenizer breaks)
    my_string = my_string.replace('. ', '.')  # remove spaces after dot to get rid of escapes
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
    elif 'snowball' in stemmer or 'default' in stemmer:
        st = SnowballStemmer('english')
    elif 'lancaster' in stemmer:
        st = LancasterStemmer()
    else:
        print('unknown stemmer, skipping stemming')
        return my_text
    my_text = my_text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return my_text


def lemmatize_verbs(my_text, lemmatizer='wordnet'):
    """

    :param my_text: pd.Series of texts
    :param lemmatizer: type of lemmatizer to use
    :return: lemmatized pd.Series of texts
    """
    lemmatizer = lemmatizer.lower()
    if 'wordnet' in lemmatizer or 'default' in lemmatizer:
        lm = WordNetLemmatizer()
    else:
        print('unknown lemmatizer, skipping lemmatizing')
        return my_text
    my_text = my_text.apply(lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
    return my_text
