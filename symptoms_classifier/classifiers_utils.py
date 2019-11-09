import datetime
import time
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
import xgboost as xgb
import catboost
from symptoms_classifier.NLP_embedding import *

classifiers = {
    'LightGBM': 'LightGBM',
    'XGBOOST': xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                max_depth=5, alpha=10, n_estimators=10, random_state=0),
    'CatBoost': catboost.CatBoostRegressor(iterations=1000, learning_rate=0.1, random_state=0),
    'Multinomial NB': naive_bayes.MultinomialNB(),
    'Gaussian NB': naive_bayes.GaussianNB(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance'),
    'Decision Tree': tree.DecisionTreeClassifier(class_weight='balanced', random_state=0),
    'Random Forest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs=10, class_weight='balanced',
                                                     random_state=0),
    'Logistic Reg': linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial'
                                                    , class_weight='balanced', max_iter=1000, random_state=0),
    'Logistic Reg CV': linear_model.LogisticRegressionCV(class_weight='balanced', solver='liblinear', random_state=0),
    'Linear SVM': svm.LinearSVC(multi_class='crammer_singer', class_weight='balanced', random_state=0),
    'SVM': svm.SVC(gamma='scale', class_weight='balanced', random_state=0),
    'SVM linear kernel': svm.SVC(kernel='linear', class_weight='balanced', random_state=0),
    'SVM sigmoid kernel': svm.SVC(kernel='sigmoid', class_weight='balanced', random_state=0),
    'SVM poly kernel': svm.SVC(kernel='poly', class_weight='balanced', random_state=0)
}


def formatted_classification_report(y_test, y_train, test_preds, train_preds):
    preds_test_clean = [round(value) for value in test_preds]
    test_report = classification_report(y_test, preds_test_clean, output_dict=True)
    df_test = pd.DataFrame(test_report).transpose()
    df_test['accuracy'] = accuracy_score(y_test, preds_test_clean)
    df_test.index.names = ['TEST']

    preds_train_clean = [round(value) for value in train_preds]
    train_report = classification_report(y_train, preds_train_clean, output_dict=True)
    df_train = pd.DataFrame(train_report).transpose()
    df_train['accuracy'] = accuracy_score(y_train, preds_train_clean)
    df_train.index.names = ['TRAIN']

    return [df_test, df_train]


def save_classifier_to_file(model, filename, timestamp=False, model_type=None):
    if timestamp:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        file_root, file_extension = os.path.splitext(filename)
        filename = file_root + '_' + str(st) + file_extension
    if 'neural' in model_type.lower() or model_type.lower() == 'nn':
        torch.save(model, filename)
    else:
        joblib.dump(model, filename)
    print('model saved under:', filename)
    return 0


def load_classifier(classifier_model):
    if isinstance(classifier_model, str):  # loading from file / dict
        if classifier_model in classifiers.keys():
            classifier = classifiers[classifier_model]
        else:
            classifier = load_classifier_from_file(classifier_model)
    else:  # loading from variable
        classifier = classifier_model
    return classifier


def load_classifier_from_file(filename, classifier_type=None):  # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    if 'neural' in classifier_type.lower() or classifier_type.lower() == 'nn':
        nn_model=torch.load(filename)
        nn_model.eval()
        return nn_model
    else:
        return joblib.load(filename)


def perf_metrics(data_labels, data_preds):
    data_labels = pd.Series(data_labels)
    data_preds = pd.Series(data_preds)
    labels = list(data_labels.unique())
    acc_score = accuracy_score(data_labels, data_preds)
    precision = precision_score(data_labels, data_preds, average=None, labels=labels)
    recall = recall_score(data_labels, data_preds, average=None, labels=labels)
    f1score = f1_score(data_labels, data_preds, average=None, labels=labels)

    res = {'acc': acc_score, 'precision': precision, 'recall': recall, 'f1': f1score}
    return res


def nn_graph_perf(train_preds, y_train, net, loss, losses=[], accs=[], ws=[], bs=[], multi_class=False):
    train_preds = clean_torch_output(train_preds, multi_class=multi_class)
    y_train = torch2numpy(y_train)
    losses.append(loss.item())
    accs.append(accuracy_score([round(value) for value in train_preds], y_train))
    ws.append(net.fc1.weight.detach().numpy()[0][0])
    bs.append(net.fc1.bias.detach().numpy()[0])
    return [losses, accs, ws, bs]


def nn_print_perf(train_preds, y_train, test_preds, y_test, multi_class=False, average='weighted'):
    train_preds, y_train, test_preds, y_test = clean_torch_vectors(train_preds, y_train, test_preds, y_test, multi_class=multi_class)
    # average = 'weighted' if y_train.nunique() <= 2 else 'binary'
    print("TRAIN -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(accuracy_score(train_preds, y_train),
                  f1_score(train_preds, y_train, average=average),
                  precision_score(train_preds, y_train, average=average),
                  recall_score(train_preds, y_train, average=average)))
    print("TEST -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(accuracy_score(test_preds, y_test),
                  f1_score(test_preds, y_test, average=average),
                  precision_score(test_preds, y_test, average=average),
                  recall_score(test_preds, y_test, average=average)))


def nn_classification_report(y, train_preds, y_train, test_preds, y_test, multi_class=False):
    train_preds, y_train, test_preds, y_test = clean_torch_vectors(train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    preds = pd.DataFrame({'class': y})
    preds.loc[y_train.index, 'split'] = 'train'
    preds.loc[y_test.index, 'split'] = 'test'
    preds.loc[y_train.index, 'preds'] = train_preds
    preds.loc[y_test.index, 'preds'] = test_preds

    df_test, df_train = formatted_classification_report(y_test, y_train, test_preds, train_preds)

    return [preds, df_test, df_train]


def test_classifier(raw_text, classifier, embedding_model_path, classifier_type, embedding_algo=None,
                    tokenization_type=None, use_weights=False, keywords=None):
    from symptoms_classifier.symptoms_classifier import TextsToClassify
    test = TextsToClassify(
        dataset=pd.DataFrame([[raw_text, 0]], columns=['text', 'class']),
        class_col='class', text_col='text')
    # Tokenize and embed text
    if tokenization_type is None:
        tokenization_type = detect_tokenization_type(embedding_model_path)
    if embedding_algo is None:
        embedding_algo = detect_embedding_model(embedding_model_path)
    test.tokenize_text(tokenization_type=tokenization_type, update_obj=True)
    test.embed_text(update_obj=True, embedding_model=embedding_model_path, embedding_algo=embedding_algo, use_weights=use_weights, keywords=keywords)
    if 'nn' in classifier_type.lower() or 'neural' in classifier_type.lower():
        emb = torch.tensor(test.embedded_text, dtype=torch.float32)
        res = classifier(emb)
    else:
        res = classifier.predict(test.embedded_text)
    return res


def clean_torch_vectors(train_preds, y_train, test_preds, y_test, multi_class=False):
    y_train = torch2numpy(y_train)
    y_test = torch2numpy(y_test)
    train_preds = clean_torch_output(train_preds, multi_class=multi_class)
    test_preds = clean_torch_output(test_preds, multi_class=multi_class)

    return [train_preds, y_train, test_preds, y_test]


def clean_torch_output(pred_vec, multi_class=False):
    if multi_class:  # get the index of max for each row
        pred_vec = torch2numpy(pred_vec).argmax(1)
    else:
        try:
            pred_vec = [1 if x > 0.5 else 0 for x in torch2numpy(pred_vec)]
        except:
            pred_vec = torch.max(pred_vec, 1)[1]
            pred_vec = torch2numpy(pred_vec)
    return pred_vec


def torch2numpy(vec):
    if isinstance(vec, torch.Tensor):
        try:
            vec = vec.numpy()
        except:
            vec = vec.detach().numpy()
    return vec


def prep_nn_dataset(w2v, sentences, y, tokenization_type, test_size, MAX_SEQ_LEN, random_state
                    , idx_train=None, idx_test=None):
    embeddings_res = embedding2torch(w2v, SEED=0)
    embeddings = embeddings_res['embeddings']
    word2id = embeddings_res['word2id']
    if idx_train is not None and idx_test is not None:
        print('dataset already split in train/test')
        y_train, y_test = [y[idx_train], y[idx_test]]
        x_train, l_train = words2integers(raw_text=sentences[idx_train], word2id=word2id, tokenization_type=tokenization_type, MAX_SEQ_LEN=MAX_SEQ_LEN)
        x_test, l_test = words2integers(raw_text=sentences[idx_test], word2id=word2id, tokenization_type=tokenization_type, MAX_SEQ_LEN=MAX_SEQ_LEN)
        mask_train = [[1] * min(MAX_SEQ_LEN, doc_len) + [0] * max(0, MAX_SEQ_LEN - doc_len) for doc_len in l_train]
        mask_test = [[1] * min(MAX_SEQ_LEN, doc_len) + [0] * max(0, MAX_SEQ_LEN - doc_len) for doc_len in l_test]
        max_len = max([sum(x) for x in mask_train+mask_test])
    else:
        x_ind, prim_len = words2integers(raw_text=sentences, word2id=word2id, tokenization_type=tokenization_type,
                                         MAX_SEQ_LEN=MAX_SEQ_LEN)
        masks = [[1] * min(MAX_SEQ_LEN, doc_len) + [0] * max(0, MAX_SEQ_LEN - doc_len) for doc_len in prim_len]
        max_len = max([sum(x) for x in masks])
        x_train, x_test, y_train, y_test, l_train, l_test, mask_train, mask_test =\
            train_test_split(x_ind, y, prim_len, masks, test_size=test_size, random_state=random_state)

    print('max len used', max_len)

    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.long)  # need to keep y_train for indices
    l_train = torch.tensor(l_train, dtype=torch.int64)  #.reshape(-1, 1)
    mask_train = torch.tensor(mask_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.long)
    y_test_torch = torch.tensor(y_test.values, dtype=torch.long)
    l_test = torch.tensor(l_test, dtype=torch.int64)  #.reshape(-1, 1)
    mask_test = torch.tensor(mask_test, dtype=torch.float32)

    return [embeddings, word2id,
            x_train[:, 0:max_len], y_train, y_train_torch, l_train, mask_train[:, 0:max_len],
            x_test[:, 0:max_len], y_test, y_test_torch, l_test, mask_test[:, 0:max_len]]


def test_nn(sentence, w2v, tokenization_type, net):
    embeddings_res = embedding2torch(w2v, SEED=0)
    word2id = embeddings_res['word2id']
    x_ind = words2integers(raw_text=sentence, word2id=word2id, tokenization_type=tokenization_type)
    x_ind = torch.tensor([x_ind], dtype=torch.long)
    net = net.eval()
    res_tmp = net(x_ind)  # output of the netweork
    res = torch.softmax(res_tmp, dim=1)
    return res
