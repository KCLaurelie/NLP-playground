import pickle  # to save models
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from symptoms_classifier.NLP_text_cleaning import *
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
import code_utils.general_utils as gutils

# spacy stuff
from code_utils.global_variables import *
import spacy
nlp = spacy.load(spacy_en_path, disable=['ner', 'parser'])
nltk.download('wordnet')


def lemmatize_words(words_list):
    words_list = ' '.join(gutils.to_list(words_list))
    res = [tkn.lemma_.lower() for tkn in nlp(words_list)]
    return res


def top_features_idf(vectorizer, top_n=2):
    # TODO check http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XPFMnohKiUk
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    _top_features = [features[i] for i in indices[:top_n]]
    return _top_features


def tokenize_text_series(text_series, manually_clean_text=True):
    """
    tokenizes series of texts using spacy (first splits in sentences then tokens)
    :param text_series: pd.Series of texts
    :param manually_clean_text: set to True to expand contractions etc
    :return:
    """
    snt = pd.Series()
    for rawtext in text_series:
        snt_tmp = text2sentences(rawtext, remove_punctuation=False)
        snt = snt.append(snt_tmp, ignore_index=True)
    res = tokenize_sentences(snt, manually_clean_text=manually_clean_text)
    return res


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
        tkns = [tkn.lemma_.lower() for tkn in nlp.tokenizer(snt.replace('…', '. '))
                if (not tkn.is_punct) and (not tkn.is_space) and tkn.is_ascii]
        if len(tkns) > 0: tok_snts.append(tkns)
    return tok_snts


def convert_snt2avgtoken(sentences, w2v_model, clean_text=True, use_weights=False, keywords=None, context=10):
    """
    convert sentences to embedded sentences using pre-trained Word2Vec model
    :param sentences: pd.Series of sentences to vectorize
    :param w2v_model: Word2Vec pre-trained model (either model object or filepath to savec model)
    :param clean_text: set to true to remove punctuations and special characters (only keeps alphanumerics)
    :param use_weights: use weight average instead f simple average, based on distance from specific keyword(s)
    :param keywords: string or list of strings to compute distance from for weighted average option
    :param context: number of tokens to use around the keywords for weighted average option
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
        if use_weights:
            weights = gutils.get_wa(sentence=snt, keywords=keywords, context=context)
            if i_snt < 20: print(snt, weights)  # print first 20 sentences to check
        for i_word, word in enumerate(snt):  # Loop over the words of a sentence
            if word in w2v_model.wv:
                word_emb = w2v_model.wv.get_vector(word) * (weights[i_word] if use_weights else 1)
                sentences_emb[i_snt] += word_emb  # w2v_model.wv.get_vector(word)
                cnt += 1
            else:
                not_in_model.append(word)
        if cnt > 0 and not use_weights:
            sentences_emb[i_snt] = sentences_emb[i_snt] / cnt

    excluded_words = list(dict.fromkeys(not_in_model))
    print(len(excluded_words), 'words not in model:', excluded_words)
    return sentences_emb


def embed_sentences(sentences, emb_model, algo='w2v', emb_option='sentence', clean_text=True,
                    use_weights=False, keywords=None, context=10):
    """
    embed text using pre-trained embedding model
    :param sentences: pd.Series of texts (1 row = 1 sentence)
    :param emb_model: pre-trained embedding model to use (model object or filepath to model)
    :param algo: embedding model type (can be either tfidf or word2vec)
    :param emb_option: represent each word separately ('word') or do an average sum by sentence (default) (only for w2v)
    :param clean_text: set to true to remove punctuations and special characters (only keeps alphanumerics)
    :param use_weights: see convert_snt2avgtoken documentation
    :param keywords: see convert_snt2avgtoken documentation
    :param context: see convert_snt2avgtoken documentation
    :return: embedded sentences
    """
    algo = str(algo).lower()

    if 'idf' in algo:
        if isinstance(emb_model, str):  # load model if stored in file
            emb_model = pickle.load(open(emb_model), "rb")
        snt_emb = emb_model.transform(sentences).toarray()
    elif 'word2vec' in algo or 'w2v' in algo:
        if isinstance(emb_model, str):  # load model if stored in file
            emb_model = Word2Vec.load(emb_model)
        sentences = sentences.to_list()

        if emb_option == 'word':  # TODO: IS THIS WORKING ????
            snt_emb = np.zeros((len(sentences), emb_model.wv.vector_size))
            for idx, snt in enumerate(sentences):
                snt_emb[idx] = [emb_model.wv.get_vector(x) for x in tokenize.word_tokenize(snt)]
        else:  # Convert each sentence into the average sum of its tokens
            snt_emb = convert_snt2avgtoken(sentences, w2v_model=emb_model, clean_text=clean_text,
                                           use_weights=use_weights, keywords=keywords, context=context)
    else:
        return 'unknown algo'
    return snt_emb


def fit_embedding_model(sentences, embedding_algo='w2v', saved_model_path=None, stop_words=None, sublinear_tf=True,
                        ngram_range=(1, 5), size=100, window=5, min_count=4, workers=4, min_df=0.00125, max_df=0.7):
    """
    train embedding model using series of texts (at the moment only allows tfidf and word2vec)
    :param sentences: pd.Series of texts (1 row = 1 sentence)
    :param min_df: ignore words below that frequency (tfidf parameter)
    :param max_df: ignore words above that frequency (tfidf parameter)
    :param sublinear_tf: replace tf with 1 + log(tf) (tfidf parameter)
    :param ngram_range: lower and upper boundary of the range of n-values for n-grams to be extracted (tfidf parameter)
    :param min_count: Ignores all words with total frequency lower than this (Word2Vec parameter)
    :param window: Maximum distance between the current and predicted word within a sentence (Word2Vec parameter)
    :param workers: for faster training with multicore machines (Word2Vec parameter)
    :param embedding_algo: embedding model type (can be either tfidf or word2vec)
    :param stop_words: tokens that will be removed from the resulting tokens. If 'english' will load nltk list
    :param size: desired size for each embedding vector
    :param saved_model_path: option to save trained model (if None will not save, otherwise will use value as file_path)
    :return: word embeddings
    """

    embedding_algo = str(embedding_algo).lower()
    tok_snts = tokenize_sentences(sentences, manually_clean_text=True)  # tokenize sentences
    if stop_words == 'english':
        stop_words = stopwords.words('english')
        stop_words = [x for x in stop_words if ('no' not in x) and ('n\'t' not in x)]  # we want to keep negations
    if 'idf' in embedding_algo:
        _vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words=stop_words,
                                      sublinear_tf=sublinear_tf, use_idf=True, preprocessor=' '.join)

        w2v = _vectorizer.fit(tok_snts)
        if saved_model_path is not None:
            with open(saved_model_path + '.pickle', 'wb') as fin:
                pickle.dump(w2v, fin)
    elif 'word2vec' in embedding_algo or 'w2v' in embedding_algo:
        w2v = Word2Vec(tok_snts, size=size, window=window, min_count=min_count, workers=workers)  # train word2vec
        if saved_model_path is not None:
            w2v.save(saved_model_path + '.model')
    else:
        return 'unknown embedding_algo'
    return w2v
