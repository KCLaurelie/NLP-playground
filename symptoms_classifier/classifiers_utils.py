import pandas as pd
import datetime
import time
import torch
from torch import nn
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
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


def save_classifier_to_file(model, filename='finalized_model.sav', timestamp=True):
    if timestamp:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        filename = filename + '_created_' + str(st) + '.sav'
    # pickle.dump(model, open(filename, 'wb'))
    joblib.dump(model, filename)
    print('model saved under:', filename)
    return 0


def load_classifier(classifier_model):
    if isinstance(classifier_model, str):
        if classifier_model in classifiers.keys():
            classifier = classifiers[classifier_model]
        else:
            classifier = load_classifier_from_file(classifier_model)
    else:
        classifier = classifier_model
    return classifier


def load_classifier_from_file(filename):  # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    return loaded_model


# save trained model
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


def create_nn(nb_classes, first_layer_neurons, dropout):
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


def print_nn_perf(train_preds, y_train, test_preds, y_test, multi_class=False):
    train_preds, y_train, test_preds, y_test = clean_torch_outputs(train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    acc = accuracy_score(train_preds, y_train)
    f1 = f1_score(train_preds, y_train)  # Use f1 from sklearn
    p = precision_score(train_preds, y_train)  # Use precision from sklearn
    r = recall_score(train_preds, y_train)  # Use recall from sklearn
    acc_test = accuracy_score(test_preds, y_test)
    f1_test = f1_score(test_preds, y_test)  # Use f1 from sklearn
    p_test = precision_score(test_preds, y_test)  # Use precision from sklearn
    r_test = recall_score(test_preds, y_test)  # Use recall from sklearn

    print("TRAIN -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(acc, f1, p, r))
    print("TEST -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}"
          .format(acc_test, f1_test, p_test, r_test))


def evaluate_nn(y, train_preds, y_train, test_preds, y_test, multi_class=False):
    train_preds, y_train, test_preds, y_test = clean_torch_outputs(train_preds, y_train, test_preds, y_test, multi_class=multi_class)

    preds = pd.DataFrame({'class': y})
    preds.loc[y_train.index, 'split'] = 'train'
    preds.loc[y_test.index, 'split'] = 'test'
    preds.loc[y_train.index, 'preds'] = train_preds
    preds.loc[y_test.index, 'preds'] = test_preds

    df_test, df_train = formatted_classification_report(y_test, y_train, test_preds, train_preds)

    return [preds, df_test, df_train]


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
