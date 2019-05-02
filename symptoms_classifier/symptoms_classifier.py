import os
from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sys import argv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.chdir(r'C:\Users\K1774755\PycharmProjects\toy-models\symptoms_classifier')

models = {
    'Multinomial NB': naive_bayes.MultinomialNB(),
    'Gaussian NB': naive_bayes.GaussianNB(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': tree.DecisionTreeClassifier(random_state=0),
    'Random Forest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs=10),
    'Logistic Regression': linear_model.LogisticRegression(),
    'SVM': svm.SVC(),
    'SVM with linear kernel': svm.SVC(kernel='linear'),
    'SVM with sigmoid kernel': svm.SVC(kernel='sigmoid'),
    'SVM with poly kernel': svm.SVC(kernel='poly')
}


def vectorizer(train_texts, test_texts=[]):
    _vectorizer = text.TfidfVectorizer(
        min_df=0.00125,
        max_df=0.7,
        sublinear_tf=True,
        use_idf=True,
        analyzer='word',
        ngram_range=(1, 5)
    )
    train_vectors = _vectorizer.fit_transform(train_texts)
    test_vectors = _vectorizer.transform(test_texts) if test_texts else []
    return train_vectors, test_vectors


def feature_generation(train_file, test_file=''):
    with open(train_file, "r") as train_data_text:
        train_data = json.load(train_data_text)
    train_texts = []
    train_class = []
    for doc in train_data:
        train_texts.append(doc['text'])
        train_class.append(doc['label'])

    test_texts = []
    if test_file:
        with open(test_file, "r") as test_data_text:
            test_data = json.load(test_data_text)
        for doc in test_data:
            test_texts.append(doc['text'])

    train_vectors, test_vectors = vectorizer(train_texts, test_texts)
    return train_vectors, train_class, test_vectors


def plot_distribution(class_array, title):
    plt.figure(title)
    pd.DataFrame(class_array, columns=['Class']).Class.value_counts().plot(
        kind='pie',
        autopct='%.2f %%',
    )
    plt.axis('equal')
    plt.title(title)


def over_sample(train_vectors, train_class):
    train_vectors = train_vectors.toarray()
    sm = SMOTE(random_state=42)
    train_vectors, train_class = sm.fit_sample(train_vectors, train_class)

    plot_distribution(train_class, 'After sampling')
    return train_vectors, train_class


def classify(classifier, train_vectors, train_class, test_vectors, test=False):
    if test:
        classifier.fit(train_vectors, train_class)
        preds = classifier.predict(test_vectors)
        return preds
    else:
        preds = cross_val_predict(classifier, train_vectors, train_class, cv=10)
        acc_score = accuracy_score(train_class,preds)
        labels = [1, -1]
        precision = precision_score(train_class, preds, average=None, labels=labels)
        recall = recall_score(train_class, preds, average=None, labels=labels)
        f1score = f1_score(train_class, preds, average=None, labels=labels)
        return acc_score, precision, recall, f1score


def train_classify(train_file, test_file, test=False):
    train_vectors, train_class, test_vectors = feature_generation(train_file, test_file)
    plot_distribution(train_class, train_file + ' Before sampling')
    train_vectors, train_class = over_sample(train_vectors, train_class)

    if test:
        eclf = ensemble.VotingClassifier(estimators=[
            ('nbm', models['Multinomial NB']),
            ('tree', models['Decision Tree']),
            ('rf', models['Random Forest']),
            ('lr', models['Logistic Regression']),
        ], voting='soft')
        preds = classify(eclf, train_vectors, train_class, test_vectors)
        f = open('data/obama_predictions.text', 'w+')
        for index, pred in enumerate(preds):
            f.write(str(index+1)+';;'+str(preds[index])+'\n')
        f.close()
    else:
        metrics = []
        for index, model in enumerate(models):
            print("Classifying using", model)
            acc_score, precision, recall, f1_score = classify(models[model], train_vectors, train_class, test_vectors)
            metrics.append({})
            metrics[index]['Classifier'] = model
            metrics[index]['accuracy'] = acc_score
            metrics[index]['possitive f1score'] = f1_score[0]
            metrics[index]['negative f1score'] = f1_score[1]
        pd.io.json.json_normalize(metrics).plot(kind='bar', x='Classifier')
        plt.title(train_file)
        plt.grid(True, axis='y')
        plt.ylim(ymax=1)
        plt.xticks(rotation=0)
        plt.show()


def testing():
    for candidate in ['obama', 'romney']:
        print('Running for', candidate, '...')
        try: test_file = 'data/' + candidate + '_test_cleaned.json'
        except: test_file = ''
        train_classify('data/' + candidate + '_cleaned.json', test_file)
