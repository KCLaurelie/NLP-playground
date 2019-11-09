from code_utils.general_utils import list_to_excel
from symptoms_classifier.symptoms_classifier import *
output_path = r'C:\Users\K1774755\PycharmProjects\toy-models\symptoms_classifier\files'


def test_final():
    tweets = TextsToClassify(
        filepath=r'C:\Users\K1774755\Downloads\phd\Tweets.csv',
        class_col='airline_sentiment', text_col='text',
        binary_main_class='positive', already_split=True)
    df = tweets.load_data()
    # tkns = tweets.tokenize_text(tokenization_type='lem', update_obj=True)
    w2v = load_embedding_model(r'C:\Users\K1774755\PycharmProjects\toy-models\embeddings\w2v_wiki.model', model_type='w2v')
    tweets.embedding_model = w2v
    tweets.make_binary_class(binary_main_class='negative')

    emb = tweets.embed_text(update_obj=True, embedding_model=w2v, tokenization_type='lem', embedding_algo='w2v', context=20, use_weights=True, keywords=['virgin', 'awesome'])

    res1 = train_nn(x_emb=tweets.embedded_text, y=tweets.dataset.class_numeric, random_state=0, n_epochs=20, multi_class=False, debug_mode=True)

    res = tweets.run_neural_net(nn_type='RNN', rnn_type='LSTM', binary=True, binary_main_class='negative',
                                tokenization_type='clean',  dropout=0.5, n_epochs=1, debug_mode=True, random_state=42)
    res = tweets.run_neural_net(nn_type='ANN', binary=True, binary_main_class='negative', dropout=0.5, n_epochs=2, debug_mode=True)

    res = tweets.run_classifier(test_size=0.2, binary=True, binary_main_class='negative', output_errors=False)

    model = cutils.load_classifier_from_file(output_path + '\\test_nn_simple.pt', classifier_type='nn')
    cutils.test_classifier('i love virgin america, they are awesome', classifier=model,
                    embedding_model_path=output_path + '\\test_w2v.dat', classifier_type='nn', tokenization_type='lem')

    res = []
    for model in cutils.classifiers.keys():
        tmp_res = tweets.run_classifier(classifier_model=model, test_size=0.2, binary=True, binary_main_class='negative',save_model=False)
        res += tmp_res['report']
    list_to_excel(res, 'testnew.xlsx', sheet_name=str(tweets.embedding_algo), startrow=0, startcol=0)
    #w2v.wv['you']

def trainw2v(
        file_path='https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv',
        text_col='text'):
    data = pd.read_csv(file_path, low_memory=False, header=0, encoding='utf8', engine='c', error_bad_lines=False)
    tok_snts = tokenize_sentences(data[text_col], tokenization_type='lem')  # tokenize sentences
    print('text tokenized')
    w2v_model = Word2Vec(tok_snts, size=100, window=5, min_count=4, workers=4)  # train word2vec
    w2v_model.save(file_path.replace('.csv', '_w2v.model'))
    print('w2v model saved')
    # x_emb = sentences2embedding_w2v(data[text_col], w2v_model, clean_text=True)  # embed sentences
    return 0


def classifier_test():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
    data_clean = data[['airline_sentiment', 'text']].rename(columns={'airline_sentiment': 'class'})
    data_clean['text'] = preprocess_text(data_clean['text'])
    emb_algo = 'tfidf'  # 'word2vec'
    vectorizer = train_embedding_model(data_clean['text'], embedding_algo=emb_algo, size=100)
    processed_features = sentences2embedding(data_clean['text'], vectorizer, embedding_algo=emb_algo)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(processed_features, data_clean['class'], test_size=0.8,
                                                        random_state=0)

    classifier = cutils.classifiers['SVM']
    classifier.fit(x_train, y_train)
    preds = classifier.predict(x_train)
    # test_metrics = cutils.perf_metrics(y_train, preds)

    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    print(confusion_matrix(y_train, preds))
    print(classification_report(y_train, preds))
    print(accuracy_score(y_train, preds))
    return 0


def w2v_test():
    file = r'C:\Users\K1774755\PycharmProjects\toy-models\embeddings\f20_all_docs(15M)_tkns_wo_space_word2vec.dat'
    w2v_model = Word2Vec.load(file)
    w2v_model = load_embedding_model(file, model_type='w2v')
    w2v_model.wv['attention']
    w2v_model.wv.similar_by_vector(w2v_model.wv['attention'], topn=10)
    w2v_model.wv.similarity('attention', 'concentration')
    vectors = [w2v_model[x] for x in "the patient shows poor concentration".split(' ')]

    sentences = parse_text('C:\\temp\\bla.txt', convert_to_series=True, remove_punctuation=True)
    sentences = ['hello my name is zelda', 'cool', 'cool cool cool', 'really cool']
    emb_snt = sentences2embedding_w2v(sentences, w2v_model, do_avg=True, use_weights=True, keywords='cool')


def test0():
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
    w2v = train_embedding_model(clean_text, min_df=0.00125, max_df=0.7, embedding_algo='tfidf', size=100)
    list(w2v.vocabulary_.keys())[:10]
    processed_features = sentences2embedding(clean_text, w2v, embedding_algo='tfidf')

    w2v = train_embedding_model(clean_text, min_df=0.00125, max_df=0.7, embedding_algo='word2vec', size=100)
    list(w2v.wv.vocab)
    return 0


def run_classifier_test(classifier_model, train_data, test_data):
    from sklearn.model_selection import cross_val_predict
    train_text = preprocess_text(train_data['text'], remove_stopwords=True, stemmer=None, lemmatizer=None)
    test_text = test_data[['text']]
    train_class = train_data[['class']]
    test_class = test_data[['class']]

    # vectorize text data
    vectorizer = train_embedding_model(train_text)
    train_data_features = sentences2embedding(train_text, vectorizer)
    test_data_features = sentences2embedding(test_text, vectorizer)

    # train classifier
    classifier = cutils.classifiers[classifier_model]
    classifier.fit(train_data_features, train_class)

    # test classifier
    test_preds = classifier.predict(test_data_features)
    train_preds = cross_val_predict(classifier, train_data_features, train_class, cv=10)
    test_metrics = cutils.perf_metrics(test_class, test_preds)
    train_metrics = cutils.perf_metrics(train_class, train_preds)
    return {'test': test_metrics, 'train': train_metrics}
