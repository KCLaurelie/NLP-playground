import pandas as pd
import datetime
import time
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def save_classifier(model, filename='finalized_model.sav', timestamp=True):
    if timestamp:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        filename = filename + '_created_' + str(st)
    pickle.dump(model, open(filename, 'wb'))
    return 0


def load_classifier(filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    return loaded_model

# save trained model
def save_model_json(model, output_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    return 0


# load json and create model
def load_model_json(json_file):
    from keras.models import model_from_json
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model & compile model
    loaded_model.load_weights("model.h5")
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

