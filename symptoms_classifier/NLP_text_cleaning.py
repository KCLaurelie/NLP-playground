from nltk import tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import unicodedata
import itertools
import contractions

"""
FOR ALL THE FUNCTIONS BELOW:
my_text can either be a string or a pd.Series of sentences (1 row = 1 sentence)
"""


def parse_text(raw_text, convert_to_series=False, remove_punctuation=False):
    """

    :param raw_text: string or path of the text file to parse
    :param convert_to_series: to convert text in a series of sentences
    :param remove_punctuation: remove punctuation once text converted to sentences
    :return:
    """
    if isinstance(raw_text, str):
        if raw_text.endswith('.txt'):
            with open(raw_text, encoding='utf8') as f:
                raw_text = f.read().strip().replace('\n', '. ')
        if convert_to_series:  # convert text to sentences if not already done
            raw_text = text2sentences(raw_text, remove_punctuation=remove_punctuation)
    return raw_text


def text2sentences(raw_text, remove_punctuation=False):
    """
    converts text into a series of sentences
    :param raw_text: string or file of the path containing the text
    :param remove_punctuation: clean punctuation from sentences
    :return: pd.Series of sentences (1 row = 1 sentence)
    """
    # some pre-cleaning before using punkt tokenizer
    raw_text = clean_string(raw_text, remove_punctuation=False)  # we keep punctuation for tokenizing
    sentences = tokenize.sent_tokenize(raw_text)
    if remove_punctuation:
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


def near_words_filter(raw_text, word1, word2, max_distance=float('inf'), either_side=True):
    """
    finds 2 words near each other in a string
    :param raw_text: string to look into
    :param word1: first word
    :param word2: second word
    :param max_distance: max number of words separating word1 and word2
    :param either_side: looking for word1...word2 only (False) or word2...word1 as well (True)
    :return: 1 if a match is found, 0 otherwise
    """
    word1 = word1.lower()
    word2 = word2.lower()
    raw_text = raw_text.lower()
    if max_distance == float('inf'):  # no need to do regex if max distance is infinity
        res = 1 if word1 in raw_text and word2 in raw_text else 0
    else:
        max_distance = str(max_distance)
        side1 = '?:' + word1 + r'\W+(?:\w+\W+){0,' + max_distance + r'}?' + word2
        side2 = word2 + r'\W+(?:\w+\W+){0,' + max_distance + r'}?' + word1
        regex = r'(' + side1 + '|' + side2 + r')' if either_side else side1
        match = re.search(regex, raw_text, re.I)
        res = 0 if match is None else 1
    return res


def distance_between_words(raw_text, word1, word2):
    """
    determines proximity of 2 words in a sentence
    :param raw_text: string to look into
    :param word1: first word
    :param word2: second word
    :return: minimum distance between the occurences of word1 and word2 in the string
    """
    words = raw_text.split()
    if word1 in words and word2 in words:
        w1_indexes = [index for index, value in enumerate(words) if value == word1]
        w2_indexes = [index for index, value in enumerate(words) if value == word2]
        distances = [abs(item[0] - item[1]) for item in itertools.product(w1_indexes, w2_indexes)]
        res = min(distances)
    else:
        res = np.nan
    return res


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


def preprocess_text(raw_text, remove_stopwords=False, stemmer=None, lemmatizer=None, keywords=None, remove_punctuation=False):
    """
    cleans text and outputs series of pre-processed sentences
    :param raw_text: raw text, can be either string, .txt file containing text or pd.Series of sentences
    :param remove_stopwords: option to remove or keep stopwords (nltk function)
    :param stemmer: can be either porter, snowball, lancaster or None
    :param lemmatizer: for verb lemmatization. can be either wordnet or None
    :param keywords: list of keywords to select sentences of interest
    :param remove_punctuation: remove or keep punctuation symbols
    :return: pd.Series of cleaned sentences
    """
    raw_text = parse_text(raw_text, convert_to_series=True, remove_punctuation=remove_punctuation)
    # clean the text
    raw_text = raw_text.apply(lambda x: clean_string(x, remove_punctuation=remove_punctuation))
    # keep only sentences with relevant keywords
    if keywords is not None:
        keywords = [x.lower() for x in keywords]
        raw_text = raw_text.apply(lambda x: keywords_filter(x, keywords=keywords)).dropna()
    # removing stop words
    if remove_stopwords: # we want to keep negations
        stop = [x for x in stopwords.words('english') if 'no' not in x]
        raw_text = raw_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # stemming
    if stemmer is not None:
        raw_text = raw_text.apply(lambda x: stem_text(x, stemmer))
    # lemming with respect to verbs
    if lemmatizer is not None:
        raw_text = raw_text.apply(lambda x: lemmatize_verbs(x, lemmatizer))

    return raw_text.replace('', np.nan).dropna()


def quick_clean_txt(text, remove_contractions=True):
    text = str(text)
    if remove_contractions: text = contractions.fix(text.replace('i', 'I')) # expand contractions (you're -> you are)
    text = re.sub(r'[^a-z0-9-\'\s]', ' ', text.lower())  # remove non alphanumeric character
    text = re.sub(r'([^0-9]{1})\1{2,}', r'\1\1', text)  # more than 3 consecutive letters -> 2 (hellooooooooo -> helloo)
    return text


def clean_string(raw_text, remove_punctuation=False):
    """

    :param raw_text: string to clean
    :param remove_punctuation: select True to remove all punctuation/special characters from the text.
    otherwise only special characters will be removed.
    :return: clean string, ready to be tokenized
    """
    raw_text = raw_text.strip()  # remove leading/trailing characters
    raw_text = unicodedata.normalize('NFKD', raw_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove non ascii characters
    raw_text = raw_text.replace('i', 'I')  # to allow expansion of i'm, i've...
    raw_text = contractions.fix(raw_text)  # expand english contractions (didn't -> did not)
    raw_text = raw_text.lower()  # to lower case
    raw_text = raw_text.replace('e.g.', 'exempli gratia')  # replace e.g. (otherwise punkt tokenizer breaks)
    raw_text = raw_text.replace('i.e.', 'id est')  # replace i.e. (otherwise punkt tokenizer breaks)
    raw_text = raw_text.replace('. ', '.')  # remove spaces after dot to get rid of escapes
    raw_text = re.sub(r'\.+', ".", raw_text)  # replace multiple dots by single dot
    raw_text = raw_text.replace('.', '. ').replace('.  ', '. ')  # ensure dots are followed by space for tokenization

    # remove special characters
    if remove_punctuation:
        raw_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_text)
    else:  # keep bare minimum punctuation
        raw_text = re.sub(r'[^a-zA-Z0-9.,\'-?!()\s]+', ' ', raw_text)
    raw_text = ' '.join(raw_text.split())  # substitute multiple spaces with single space

    return raw_text


def stem_text(raw_text, stemmer='snowball'):
    """

    :param raw_text: string to stem
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
        return raw_text
    # raw_text = raw_text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    raw_text = " ".join([st.stem(word) for word in raw_text.split()])
    return raw_text


def lemmatize_verbs(raw_text, lemmatizer='wordnet'):
    """

    :param raw_text: string to lemmatize
    :param lemmatizer: type of lemmatizer to use
    :return: lemmatized pd.Series of texts
    """
    lemmatizer = lemmatizer.lower()
    if 'wordnet' in lemmatizer or 'default' in lemmatizer:
        lm = WordNetLemmatizer()
    else:
        print('unknown lemmatizer, skipping lemmatizing')
        return raw_text
    # raw_text = raw_text.apply(lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
    raw_text = " ".join([lm.lemmatize(word, pos='v') for word in raw_text.split()])
    return raw_text
