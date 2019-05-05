# from https://github.com/sachinbiradar9/Sentiment-Classification/blob/master/utils.py
from collections import Counter
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction import text
import pandas as pd
import re


def preprocess_text(text):  # text = series of texts
    # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))
    # to lower case
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Removing punctuation
    text = text.str.replace('[^\w\s]', '')
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Stop word removal
    stop = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Stemming
    st = PorterStemmer() #SnowballStemmer('english')
    text = text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    res = " ".join(words)
    return res


def vectorizer(texts):
    _vectorizer = text.TfidfVectorizer(
        min_df=0.00125,
        max_df=0.7,
        sublinear_tf=True,
        use_idf=True,
        analyzer='word',
        ngram_range=(1, 5)
    )
    vectors = _vectorizer.fit_transform(texts)
    return vectors.toarray()


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

def all_words(text):
    return re.findall('\\w+', text.lower())


def P(word, word_dic_file):
    #word_dic_file = 'constants/big.txt'
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


# word dictionary______________________________________________________________________________________________________
def create_word_dictionary(word_dic):
    word_dictionary = list(set(word_dic.words()))
    for alphabet in 'bcdefghjklmnopqrstuvwxyz':
        word_dictionary.remove(alphabet)

    useless_two_letter_words = pandas.read_csv('constants/useless_two_letter_words.csv')
    for word in useless_two_letter_words:
        word_dictionary.remove(word)

    useful_words = pandas.read_csv('constants/useful_words.csv')
    for word in useful_words:
        word_dictionary.append(word)

    contractions = pandas.read_csv('constants/contractions.csv')
    for key in contractions:
        word_dictionary.append(key)

    return word_dictionary
