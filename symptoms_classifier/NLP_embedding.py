import pickle  # to save models
import spacy
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import numpy as np
nltk.download('wordnet')

"""
FOR ALL THE FUNCTIONS BELOW:
my_text is a pd.Series of sentences (1 row = 1 sentence)
"""


def top_features(vectorizer, top_n=2):
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    _top_features = [features[i] for i in indices[:top_n]]
    return _top_features

# TODO check http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XPFMnohKiUk


def convert_stn2avgtoken(sentences, w2v_model):
    """
    convert sentences to embedded sentences using pre-trained Word2Vec model
    :param sentences: sentences to vectorize
    :param w2v_model: Word2Vec pre-trained model (either model object or filepath to savec model)
    :return: embedded sentences
    """
    if isinstance(w2v_model, str):  # load model if saved in file
        w2v_model = Word2Vec.load(w2v_model)
    size = w2v_model.wv.vector_size  # size of embeeding vectors

    sentences_emb = np.zeros((len(sentences), size))  # to store embedded sentences

    # tokenize the text and further cleans it (needed otherwise it would take 1 token = 1 letter)
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    tok_snts = []
    for snt in sentences:
        tkns = [tkn.lemma_.lower() for tkn in nlp.tokenizer(snt) if not tkn.is_punct]
        tok_snts.append(tkns)
    sentences = tok_snts  # save back

    # convert each sentence into the average sum of the vector representations of its tokens
    for i_snt, snt in enumerate(sentences):  # Loop over sentences
        cnt = 0
        for i_word, word in enumerate(snt):  # Loop over the words of a sentence
            print(word)
            if word in w2v_model.wv:
                print('is in model')
                sentences_emb[i_snt] += w2v_model.wv.get_vector(word)
                cnt += 1
        if cnt > 0:
            sentences_emb[i_snt] = sentences_emb[i_snt] / cnt

    return sentences_emb


def transform_text2vec(my_text, emb_model, algo='tfidf', word2vec_option='sentence'):
    """

    :param my_text: pd.Series of texts (1 row = 1 sentence)
    :param emb_model: pre-trained embedding model to use (model object or filepath to model)
    :param algo: embedding model type (can be either tfidf, counter or word2vec)
    :param word2vec_option: represent each word separately ('word') or do an average sum by sentence (default)
    :return: vectorized text
    """
    algo = str(algo).lower()

    if any(substring in algo for substring in ('idf', 'count')):
        if isinstance(emb_model, str):
            emb_model = pickle.load(open(emb_model), "rb")
        vectors = emb_model.transform(my_text).toarray()
    elif 'word2vec' in algo:
        if isinstance(emb_model, str):
            emb_model = Word2Vec.load(emb_model)
        sentences = my_text.to_list()

        if word2vec_option == 'word':  # TODO ????
            vectors = np.zeros((len(sentences), emb_model.wv.vector_size))
            for idx, snt in enumerate(sentences):
                vectors[idx] = [emb_model.wv.get_vector(x) for x in tokenize.word_tokenize(snt)]
        else:  # Convert each sentence into the average sum of its tokens
            vectors = convert_stn2avgtoken(sentences, w2v_model=emb_model)
    else:
        return 'unknown algo'
    return vectors


def fit_text2vec(my_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100, save_vectorizer=False):
    """

    :param my_text: pd.Series of texts (1 row = 1 sentence)
    :param min_df: ignore words below that frequency (used for tfidf and counter)
    :param max_df: ignore words above that frequency (used for tfidf and counter)
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
        if save_vectorizer:
            pickle.dump(w2v, open("vectorizer.pickle", "wb"))
    elif 'word2vec' in algo:
        if isinstance(my_text, str):
            sentences = tokenize.sent_tokenize(my_text)
        else:
            sentences = my_text.to_list()
        vocab = []
        for snt in sentences:
            vocab.append(tokenize.word_tokenize(snt))
        w2v = Word2Vec(vocab, min_count=1, size=_size)  #, window=6, min_count=5, workers=4)
        if save_vectorizer:
            w2v.save("word2vec.model")
    else:
        return 'unknown algo'
    return w2v


