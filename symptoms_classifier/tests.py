import pandas as pd
from symptoms_classifier.NLP_text_cleaning import clean_string, text2sentences, preprocess_text, parse_text
from symptoms_classifier.NLP_embedding import fit_text2vec, transform_text2vec, convert_stn2avgtoken
from gensim.models import Word2Vec
import spacy


def quick_embedding_ex(
        data_file="https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv",
        text_col='text',
        class_col='airline_sentiment'):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    data = pd.read_csv(data_file)[[class_col, text_col]]
    sentences = data[text_col]
    y = data[class_col]
    tok_snts = []
    for snt in sentences:
        tkns = [tkn.lemma_.lower() for tkn in nlp.tokenizer(snt) if not tkn.is_punct]
        tok_snts.append(tkns)
    sentences = tok_snts
    w2v_model = Word2Vec(sentences, size=300, window=6, min_count=4, workers=4)
    x_emb = convert_stn2avgtoken(sentences, w2v_model)
    return [x_emb, y]


def test0():
    # w2v_model = Word2Vec.load(r'C:\Users\K1774755\Downloads\phd\discharge_summaries_unigram_size100_window5_mincount5')
    w2v_model = Word2Vec.load(
        r'C:\Users\K1774755\Downloads\phd\early_intervention_services_unigram_size100_window5_mincount5')
    w2v_model.wv['attention']
    w2v_model.wv.similar_by_vector(w2v_model.wv['attention'], topn=10)
    w2v_model.wv.similarity('attention', 'concentration')
    vectors = [w2v_model[x] for x in "the patient shows poor concentration".split(' ')]

    sentences = parse_text('C:\\temp\\bla.txt', convert_to_series=True, remove_punctuation=True)
    emb_snt = convert_stn2avgtoken(sentences, w2v_model)


def test():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
    data_clean = data[['airline_sentiment', 'text']].rename(columns={'airline_sentiment': 'class'})
    txt = data_clean['text'][0:10]
    raw_text = "hi my name is link...I like to fight, And i'm in love with princess zelda.bim. bam.Boum. Bom"
    raw_text = clean_string(raw_text)
    raw_text = text2sentences(raw_text)

    txt = 'C:\\temp\\bla.txt'
    text2sentences(txt)

    clean_text = preprocess_text(txt)
    preprocess_text(txt, remove_stopwords=True, stemmer='snowball', lemmatizer=None)
    # vocab = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    w2v = fit_text2vec(clean_text, min_df=0.00125, max_df=0.7, algo='tfidf', _size=100)
    list(w2v.vocabulary_.keys())[:10]
    processed_features = transform_text2vec(clean_text, w2v, algo='tfidf')

    w2v = fit_text2vec(clean_text, min_df=0.00125, max_df=0.7, algo='word2vec', _size=100)
    list(w2v.wv.vocab)
    return 0
