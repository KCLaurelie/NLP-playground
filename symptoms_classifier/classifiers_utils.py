import datetime
import time
import torch
from torch import nn
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
from symptoms_classifier.symptoms_classifier import *
import xgboost as xgb
import catboost

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


def save_classifier_to_file(model, filename='finalized_model.sav', timestamp=True, model_type=None):
    if timestamp:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        filename = str(st) + '_' + filename
    # pickle.dump(model, open(filename, 'wb'))
    if 'neural' in model_type.lower() or model_type.lower() == 'nn':
        torch.save(model.state_dict(), filename)
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


def load_classifier_from_file(filename, model_type=None, nn_model=None,
                              nb_classes=1, first_layer_neurons=300, dropout=None):  # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    if 'neural' in model_type.lower() or model_type.lower() == 'nn':
        if nn_model is None:
            nn_model = nn_create(nb_classes=nb_classes, first_layer_neurons=first_layer_neurons, dropout=dropout)
        nn_model.load_state_dict(torch.load(filename))
        nn_model.eval()
        return nn_model
    else:
        return joblib.load(filename)


def save_model_json(model, output_file="model.json", weights_file="model.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(output_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_file)
    return 0


# load json and create model
def load_model_json(json_file, weights_file):
    from keras.models import model_from_json
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model & compile model
    loaded_model.load_weights(weights_file)
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    print("Loaded model from disk")

    # score_saved_model = loaded_model.evaluate(x_test, y_test, verbose=0)
    return loaded_model


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


def nn_create(nb_classes=1, first_layer_neurons=300, dropout=None):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(first_layer_neurons, 100)
            self.fc2 = nn.Linear(100, nb_classes)  # 3 classes = 3 neurons
            if dropout is not None:
                print('using dropout')
                self.d1 = nn.Dropout(dropout)  # do we want dropout?

        def forward(self, x):
            if dropout is not None:
                x = self.d1(torch.relu(self.fc1(x)))
            else:
                x = torch.relu(self.fc1(x))  # torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    return Net()


def nn_print_perf(train_preds, y_train, test_preds, y_test, multi_class=False):
    train_preds, y_train, test_preds, y_test = clean_torch_outputs(train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    print("TRAIN -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(accuracy_score(train_preds, y_train),
                  f1_score(train_preds, y_train),
                  precision_score(train_preds, y_train),
                  recall_score(train_preds, y_train)))
    print("TEST -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(accuracy_score(test_preds, y_test),
                  f1_score(test_preds, y_test),
                  precision_score(test_preds, y_test),
                  recall_score(test_preds, y_test)))


def nn_classification_report(y, train_preds, y_train, test_preds, y_test, multi_class=False):
    train_preds, y_train, test_preds, y_test = clean_torch_outputs(train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    preds = pd.DataFrame({'class': y})
    preds.loc[y_train.index, 'split'] = 'train'
    preds.loc[y_test.index, 'split'] = 'test'
    preds.loc[y_train.index, 'preds'] = train_preds
    preds.loc[y_test.index, 'preds'] = test_preds

    df_test, df_train = formatted_classification_report(y_test, y_train, test_preds, train_preds)

    return [preds, df_test, df_train]


def test_classifier(raw_text, classifier, embedding_model_path, classifier_type, embedding_algo=None,
                    tokenization_type=None, use_weights=False, keywords=None):
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


def clean_torch_outputs(train_preds, y_train, test_preds, y_test, multi_class=False):
    if isinstance(train_preds, torch.Tensor): train_preds = train_preds.numpy()
    if isinstance(y_train, torch.Tensor): y_train = y_train.numpy()
    if isinstance(test_preds, torch.Tensor): test_preds = test_preds.numpy()
    if isinstance(y_test, torch.Tensor): y_test = y_test.numpy()

    if multi_class:  # get the index of max for each row
        train_preds = train_preds.argmax(1)  #.numpy()  # torch.max(train_preds, dim=1).indices.numpy()
        test_preds = test_preds.argmax(1)  #.numpy()
    else:
        test_preds = [1 if x > 0.5 else 0 for x in test_preds]
        train_preds = [1 if x > 0.5 else 0 for x in train_preds]

    return [train_preds, y_train, test_preds, y_test]
