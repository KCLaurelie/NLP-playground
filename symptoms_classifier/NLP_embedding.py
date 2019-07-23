import os
import sys
import pickle  # to save models
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from symptoms_classifier.NLP_text_cleaning import clean_string
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')

# spacy stuff
spacy_lib = r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\envs\spacy\Lib\site-packages'
sys.path.append(spacy_lib)
spacy_en_path = os.path.join(spacy_lib, r'en_core_web_sm\en_core_web_sm-2.1.0')
import spacy
nlp = spacy.load(spacy_en_path, disable=['ner', 'parser'])


def top_features_idf(vectorizer, top_n=2):
    # TODO check http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XPFMnohKiUk
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    _top_features = [features[i] for i in indices[:top_n]]
    return _top_features


def tokenize_sentences(sentences, manually_clean_text=True):
    """
    tokenize text using spacy
    :param sentences: pd.Series of sentences
    :param manually_clean_text: set to True to expand contractions etc
    :return: list of tokenized sentences
    """
    tok_snts = []
    for snt in sentences:
        if manually_clean_text:
            snt = clean_string(snt, remove_punctuation=True)
        # else:
        #     snt = re.sub(r'[^a-zA-Z0-9\'\s]', ' ', snt)
        tkns = [tkn.lemma_.lower() for tkn in nlp.tokenizer(snt.replace('…','. '))
                if (not tkn.is_punct) and (not tkn.is_space) and tkn.is_ascii]
        if len(tkns) > 0: tok_snts.append(tkns)
    return tok_snts


def convert_snt2avgtoken(sentences, w2v_model, clean_text=True):
    """
    convert sentences to embedded sentences using pre-trained Word2Vec model
    :param sentences: pd.Series of sentences to vectorize
    :param w2v_model: Word2Vec pre-trained model (either model object or filepath to savec model)
    :param clean_text: set to true to remove punctuations and special characters (only keeps alphanumerics)
    :return: embedded sentences
    """
    if isinstance(w2v_model, str):  # load model if saved in file
        w2v_model = Word2Vec.load(w2v_model)
    size = w2v_model.wv.vector_size  # size of embeeding vectors

    sentences_emb = np.zeros((len(sentences), size))  # to store embedded sentences

    # tokenize the text and further cleans it (needed otherwise it would take 1 token = 1 letter)
    sentences = tokenize_sentences(sentences, manually_clean_text=clean_text)

    # convert each sentence into the average sum of the vector representations of its tokens
    not_in_model = []
    for i_snt, snt in enumerate(sentences):  # Loop over sentences
        cnt = 0
        for i_word, word in enumerate(snt):  # Loop over the words of a sentence
            if word in w2v_model.wv:
                sentences_emb[i_snt] += w2v_model.wv.get_vector(word)
                cnt += 1
            else:
                not_in_model.append(word)
        if cnt > 0:
            sentences_emb[i_snt] = sentences_emb[i_snt] / cnt

    print('words not in model:', list(dict.fromkeys(not_in_model)))
    return sentences_emb


def transform_text2vec(sentences, emb_model, algo='tfidf', word2vec_option='sentence', clean_text=True):
    """
    embed text using pre-trained embedding model
    :param sentences: pd.Series of texts (1 row = 1 sentence)
    :param emb_model: pre-trained embedding model to use (model object or filepath to model)
    :param algo: embedding model type (can be either tfidf, counter or word2vec)
    :param word2vec_option: represent each word separately ('word') or do an average sum by sentence (default)
    :param clean_text: set to true to remove punctuations and special characters (only keeps alphanumerics)
    :return: vectorized text
    """
    algo = str(algo).lower()

    if any(substring in algo for substring in ('idf', 'count')):
        if isinstance(emb_model, str):
            emb_model = pickle.load(open(emb_model), "rb")
        vectors = emb_model.transform(sentences).toarray()
    elif 'word2vec' in algo or 'w2v' in algo:
        if isinstance(emb_model, str):
            emb_model = Word2Vec.load(emb_model)
        sentences = sentences.to_list()

        if word2vec_option == 'word':  # TODO ????
            vectors = np.zeros((len(sentences), emb_model.wv.vector_size))
            for idx, snt in enumerate(sentences):
                vectors[idx] = [emb_model.wv.get_vector(x) for x in tokenize.word_tokenize(snt)]
        else:  # Convert each sentence into the average sum of its tokens
            vectors = convert_snt2avgtoken(sentences, w2v_model=emb_model, clean_text=clean_text)
    else:
        return 'unknown algo'
    return vectors


def fit_text2vec(sentences, embedding_algo='w2v', save_model=False,
                 size=100, window=5, min_count=4, workers=4, min_df=0.00125, max_df=0.7):
    """
    train embedding model using series of texts
    :param sentences: pd.Series of texts (1 row = 1 sentence)
    :param min_df: ignore words below that frequency (used for tfidf and counter)
    :param max_df: ignore words above that frequency (used for tfidf and counter)
    :param min_count: w2v parameter
    :param window: w2v parameter
    :param workers: w2v parameter
    :param embedding_algo: embedding model type (can be either tfidf, counter or word2vec)
    :param size: desired vector size
    :param save_model: option to save trained model
    :return: word embeddings
    """

    embedding_algo = str(embedding_algo).lower()
    stop_words = stopwords.words('english')
    if 'idf' in embedding_algo:
        _vectorizer = text.TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            use_idf=True,
            analyzer='word',
            ngram_range=(1, 5),
            stop_words=stop_words
        )
        w2v = _vectorizer.fit(sentences)
    elif 'count' in embedding_algo:
        _vectorizer = CountVectorizer(max_features=2500, min_df=min_df, max_df=max_df, stop_words=stop_words)
        w2v = _vectorizer.fit(sentences)
        if save_model:
            pickle.dump(w2v, open("vectorizer.pickle", "wb"))
    elif 'word2vec' in embedding_algo or 'w2v' in embedding_algo:
        tok_snts = tokenize_sentences(sentences, manually_clean_text=True)  # tokenize sentences
        w2v = Word2Vec(tok_snts, size=size, window=window, min_count=min_count, workers=workers)  # train word2vec
        if save_model:
            w2v.save("word2vec.model")
    else:
        return 'unknown embedding_algo'
    return w2v


