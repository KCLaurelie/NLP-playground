from code_utils.global_variables import *
from symptoms_classifier.NLP_text_cleaning import *
import code_utils.general_utils as gutils
import numpy as np
import pickle  # to save models
from gensim.models import Word2Vec, TfidfModel, KeyedVectors
from gensim.matutils import corpus2csc
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load(spacy_en_path, disable=['ner', 'parser'])
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
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


def tokenize_text_series(text_series, **kwargs):
    """
    tokenizes series of texts using spacy (first splits in sentences then tokens)
    :param text_series: pd.Series of texts
    :param tokenization_type: 'lem' (lemmatized), 'lem_stop' (lemmatized and stopwords removed), 'wo_space' (simple tokenization)
    :param output_file_path: to save tokenized sentences in a file
    :return:
    """
    snt = pd.Series()
    for rawtext in text_series:
        snt_tmp = text2sentences(rawtext, remove_punctuation=False)
        snt = snt.append(snt_tmp, ignore_index=True)
    res = tokenize_sentences(snt, **kwargs)
    return res


def tokenize_sentences(sentences, tokenization_type=None, output_file_path=None, remove_contractions=True):
    if tokenization_type not in ('lem', 'wo_space', 'lem_stop'):
        tokenization_type = detect_tokenization_type(tokenization_type)
    print('tokenizing text using', tokenization_type)
    tok_snts = []
    for snt in sentences:
        tkns = nlp.tokenizer(contractions.fix(snt.replace('i', 'I'))) if remove_contractions else nlp.tokenizer(snt)
        if tokenization_type == 'wo_space':
            _tkns = [str(x.text).lower() for x in tkns if not x.is_space]
        elif tokenization_type == 'lem':
            _tkns = [str(x.lemma_).lower() for x in tkns if not x.is_space and not x.is_punct]
        else:
            _tkns = [str(x.lemma_).lower() for x in tkns if not x.is_space and not x.is_punct
                     and not (x.is_stop and 'no' not in str(x) and 'n\'t' not in str(x))]
        tok_snts.append(_tkns)
        if output_file_path is not None: output_file_path.write("{}\n".format("\t".join(_tkns)))

    if output_file_path is not None: output_file_path.close()
    return tok_snts


def detect_tokenization_type(emb_model_file):
    if emb_model_file is None:
        print('no embedding file or tokenization type provided, using default tokenization method (simple without space)')
        return 'wo_space'
    emb_model_file = emb_model_file.lower()
    if 'lem' not in emb_model_file and 'stop' not in emb_model_file:
        res = 'wo_space'
    elif 'lem' in emb_model_file and 'stop' not in emb_model_file:
        res = 'lem'
    else:
        res = 'lem_stop'
    print('tokenization method detected:', res)
    return res


def detect_embedding_model(emb_model_file):
    emb_model_file = emb_model_file.lower()
    if 'w2v' in emb_model_file or 'word2' in emb_model_file:
        res = 'w2v'
    elif 'idf' in emb_model_file and 'gensim' in emb_model_file:
        res = 'gensim_tfidf'
    elif 'idf' in emb_model_file:
        res = 'tfidf'
    else:
        raise NotImplementedError("Model cannot be inferred from file name")
    print('embedding model detected:', res)
    return res


def load_embedding_model(filename, model_type=None):
    if model_type is None:
        model_type = detect_embedding_model(filename.lower())
    else:
        model_type = model_type.lower()
    if 'w2v' in model_type or 'word2' in model_type:
        try:
            return Word2Vec.load(filename)
        except:
            return KeyedVectors.load(filename)
    elif 'idf' in model_type and 'gensim' in model_type:
        return TfidfModel.load(filename)
    elif 'idf' in model_type and 'sklearn' in model_type:  # TFIDF with sklearn
        return pickle.load(open(filename), "rb")
    else:
        raise NotImplementedError("Model not currently supported")


def read_tokens_list(filename, sep="\t"):
    class SentenceIterator:
        def __init__(self, filepath):
            self.filepath = filepath

        def __iter__(self):
            for line in open(self.filepath):
                yield line.split(sep)

    return SentenceIterator(filename)


def snt_2_w2vemb(sentences, w2v_model, tokenization_type='lem',
                 do_avg=True, use_weights=False, keywords=None, context=10):
    """
    convert sentences to embedded sentences using pre-trained Word2Vec model
    :param sentences: pd.Series of sentences to vectorize
    :param w2v_model: Word2Vec pre-trained model (either model object or filepath to savec model)
    :param tokenization_type: 'lem' (lemmatized), 'lem_stop' (lemmatized and stopwords removed), 'wo_space' (simple tokenization)
    :param do_avg: (True or False) if set to True, compute average of embedding vectors (instead of storing them for each word)
    :param use_weights: use weight average instead f simple average, based on distance from specific keyword(s)
    :param keywords: string or list of strings to compute distance from for weighted average option
    :param context: number of tokens to use around the keywords for weighted average option
    :return: embedded sentences
    """
    print('embedding using Word2Vec model (using average sum of tokens) with:', '\nuse of weights:', use_weights,
          '\nkeywords:', keywords, '\ncontext:', context)

    if isinstance(w2v_model, str):  # load model if saved in file
        w2v_model = Word2Vec.load(w2v_model)

    # tokenize the text and further cleans it (needed otherwise it would take 1 token = 1 letter)
    if not isinstance(sentences[0], list):
        if tokenization_type is None:  # detect tokenization from model file name if not precised
            tokenization_type = detect_tokenization_type(w2v_model)
        sentences = tokenize_sentences(sentences, tokenization_type=tokenization_type)

    # initialize array to store embedded sentences
    emb_size = w2v_model.wv.vector_size
    size = emb_size if do_avg else max([len(snt) for snt in sentences]) * emb_size
    sentences_emb = np.zeros((len(sentences), size))

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
                if do_avg:
                    sentences_emb[i_snt] += word_emb
                    cnt += 1
                else:
                    sentences_emb[i_snt, (i_word*emb_size):((i_word+1)*emb_size)] = word_emb
            else:
                not_in_model.append(word)
        if cnt > 0 and not use_weights:
            sentences_emb[i_snt] = sentences_emb[i_snt] / cnt

    excluded_words = list(dict.fromkeys(not_in_model))
    print(len(excluded_words), 'words not in model:', excluded_words)
    return sentences_emb


def embed_sentences(tkn_sentences, embedding_model, embedding_algo='w2v', **kwargs):
    """
    embed text using pre-trained embedding model
    :param tkn_sentences: pd.Series of tokenized sentences (1 row = 1 list of tokens)
    :param embedding_model: pre-trained embedding model to use (model object or filepath to model)
    :param embedding_algo: embedding model type (can be either tfidf or word2vec)
    :return: embedded sentences
    """

    if embedding_algo is None:
        embedding_algo = detect_embedding_model(str(embedding_model))

    embedding_algo = str(embedding_algo).lower()

    if isinstance(embedding_model, str):
        embedding_model = load_embedding_model(embedding_model, model_type=embedding_algo)

    if 'idf' in embedding_algo and ('sklearn' in embedding_algo or 'gensim' not in embedding_algo): # TFIDF with sklearn
        print('embedding using sklearn TfIdf model')
        snt_emb = embedding_model.transform(tkn_sentences).toarray()
    elif 'idf' in embedding_algo and 'gensim' in embedding_algo:  # TFIDF with gensim
        print('embedding using gensim TfIdf model')
        tkn_sentences_dict = corpora.Dictionary(tkn_sentences)
        tkn_sentences_corpus = [tkn_sentences_dict.doc2bow(stn) for stn in tkn_sentences]
        snt_emb = corpus2csc(embedding_model[tkn_sentences_corpus]).T.toarray()
    elif 'word2vec' in embedding_algo or 'w2v' in embedding_algo:
        snt_emb = snt_2_w2vemb(tkn_sentences, w2v_model=embedding_model, **kwargs)

    else:
        return 'unknown embedding_algo'
    return snt_emb


def fit_embedding_model(sentences, embedding_algo='w2v', tokenization_type='lem', save_model_path=None, stop_words=None, sublinear_tf=True,
                        ngram_range=(1, 5), size=300, window=5, min_count=1, workers=4, min_df=0.00125, max_df=0.7, max_features=None):
    """
    train embedding model using series of texts (at the moment only allows tfidf and word2vec)
    :param sentences: pd.Series of texts or tokens (1 row = 1 sentence or 1 list of tokens)
    :param min_df: ignore words below that frequency (tfidf parameter)
    :param max_df: ignore words above that frequency (tfidf parameter)
    :param tokenization_type: 'lem' (lemmatized), 'lem_stop' (lemmatized and stopwords removed), 'wo_space' (simple tokenization)
    :param sublinear_tf: replace tf with 1 + log(tf) (tfidf parameter)
    :param max_features: only consider the top max_features ordered by term frequency across the corpus (tfidf parameter)
    :param ngram_range: lower and upper boundary of the range of n-values for n-grams to be extracted (tfidf parameter)
    :param min_count: Ignores all words with total frequency lower than this (Word2Vec parameter)
    :param window: Maximum distance between the current and predicted word within a sentence (Word2Vec parameter)
    :param workers: for faster training with multicore machines (Word2Vec parameter)
    :param embedding_algo: embedding model type (can be either tfidf or word2vec)
    :param stop_words: tokens that will be removed from the resulting tokens. If 'english' will load nltk list
    :param size: desired size for each embedding vector
    :param save_model_path: option to save trained model (if None will not save, otherwise will use value as file_path)
    :return: word embeddings
    """

    embedding_algo = str(embedding_algo).lower()
    if not isinstance(sentences[0], list):  # sentences have not been tokenized
        sentences = tokenize_sentences(sentences, tokenization_type=tokenization_type)  # tokenize sentences
    if stop_words == 'english':
        stop_words = stopwords.words('english')
        stop_words = [x for x in stop_words if ('no' not in x) and ('n\'t' not in x)]  # we want to keep negations
    if 'idf' in embedding_algo and ('sklearn' in embedding_algo or 'gensim' not in embedding_algo):
        print('training sklearn TfIdf vectorizer with:', '\nmax features:', max_features, '\nmindf:', min_df,
              '\nmaxdf', max_df, '\nngram', ngram_range)
        _vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words=stop_words,
                                      sublinear_tf=sublinear_tf, use_idf=True, preprocessor=' '.join, max_features=max_features)

        w2v = _vectorizer.fit(sentences)
        if save_model_path is not None:
            with open(save_model_path + '_mindf_' + str(min_df) + '_maxdf_' + str(max_df) + '_ngram_' + str(ngram_range) + '.pickle', 'wb') as fin:
                pickle.dump(w2v, fin)
    elif 'idf' in embedding_algo and 'gensim' in embedding_algo:
        print('training gensim TfIdf model')
        tkn_sentences_dict = corpora.Dictionary(sentences)
        tkn_sentences_corpus = [tkn_sentences_dict.doc2bow(stn) for stn in sentences]
        w2v = TfidfModel(tkn_sentences_corpus)
        if save_model_path is not None:
            w2v.save(save_model_path + embedding_algo + '.model')
    elif 'word2vec' in embedding_algo or 'w2v' in embedding_algo:
        print('training Word2Vec model with:', '\nsize:', size, '\nwindow:', window, '\nmincount:', min_count)
        w2v = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)  # train word2vec
        if save_model_path is not None:
            w2v.save(save_model_path + '_size_' + str(size) + '_win_' + str(window) + '_mincount_' + str(min_count) + '.model')
    else:
        return 'unknown embedding_algo'
    return w2v


def add_padding(snt_emb, max_len=None):
    if max_len is None:
        max_len = max([len(snt) for snt in snt_emb])
    return 0


def embed_text_with_padding(sentences, w2v, tokenization_type, keywords, ln=40):
    EMB_SIZE = w2v.wv.vector_size
    # tokenize text
    x = tokenize_sentences(sentences, tokenization_type=tokenization_type)

    # Get embedding weights
    emb_weights = []
    tkn_ind = {}
    for word in w2v.wv.vocab.keys():
        tkn_ind[word] = len(emb_weights)
        emb_weights.append(w2v.wv[word])

    # Add the special tokens
    tkn_ind["<pad>"] = len(emb_weights)
    emb_weights.append(np.random.rand(EMB_SIZE))

    tkn_ind["<unk>"] = len(emb_weights)
    emb_weights.append(np.zeros(EMB_SIZE))

    # Convert to numpy
    emb_weights = np.array(emb_weights)

    ind_tkn = {}
    for key in tkn_ind.keys():
        ind_tkn[tkn_ind[key]] = key

    # Remove above 7 from each side
    new_x = [None] * len(x)
    for ind, row in enumerate(x):
        n_row = row[0:ln * 2]
        for i, word in enumerate(row):
            if any([x in word for x in keywords]):
                n_row = x[ind][max(0, i - ln):(i + ln)]
        new_x[ind] = n_row
    x = new_x

    c_ind = [-1] * len(x)
    for ind, row in enumerate(x):
        for i, word in enumerate(row):
            if any([x in word for x in keywords]):
                c_ind[ind] = i

    # Index 'x'
    x_ind = [[tkn_ind[tkn] if tkn in tkn_ind else tkn_ind['<unk>'] for tkn in tkns] for tkns in x]

    # Pad 'x'
    MAX_LENGTH = max([len(doc) for doc in x_ind])
    for i in range(len(x_ind)):
        while len(x_ind[i]) != MAX_LENGTH:
            x_ind[i].append(tkn_ind['<pad>'])

    return [emb_weights, x_ind, c_ind]
