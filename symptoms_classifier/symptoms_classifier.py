import os
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from symptoms_classifier.NLP_embedding import fit_text2vec, transform_text2vec
from symptoms_classifier.NLP_text_cleaning import preprocess_text
import symptoms_classifier.general_utils as gutils
os.chdir(r'C:\Users\K1774755\PycharmProjects\toy-models\symptoms_classifier')
matplotlib.use('Qt5Agg')

models = {
    'Multinomial NB': naive_bayes.MultinomialNB(),
    'Gaussian NB': naive_bayes.GaussianNB(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': tree.DecisionTreeClassifier(random_state=0),
    'Random Forest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs=10),
    'Logistic Regression': linear_model.LogisticRegression(),
    'SVM': svm.SVC(gamma='scale'),
    'SVM with linear kernel': svm.SVC(kernel='linear'),
    'SVM with sigmoid kernel': svm.SVC(kernel='sigmoid'),
    'SVM with poly kernel': svm.SVC(kernel='poly')
}

def test():
    data = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
    data_clean = data[['airline_sentiment', 'text']].rename(columns={'airline_sentiment': 'class'})
    data_clean['text'] = preprocess_text(data_clean['text'])
    vectorizer = fit_text2vec(data_clean['text'], algo='word2vec', _size=100)
    processed_features = transform_text2vec(data_clean['text'], vectorizer, algo='word2vec')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(processed_features, data_clean['class'], test_size=0.8, random_state=0)

    classifier = models['SVM']
    classifier.fit(x_train, y_train)
    preds = classifier.predict(x_train)
    #test_metrics = gutils.perf_metrics(y_train, preds)

    print(confusion_matrix(y_train, preds))
    print(classification_report(y_train, preds))
    print(accuracy_score(y_train, preds))
    return 0


def run_model(model, train_data, test_data):
    train_text = preprocess_text(train_data['text'], remove_stopwords=True, stemmer=None, lemmatizer=None)
    test_text = test_data[['text']]
    train_class = train_data[['class']]
    test_class = test_data[['class']]

    # vectorize text data
    vectorizer = fit_text2vec(train_text)
    train_data_features = transform_text2vec(train_text, vectorizer)
    test_data_features = transform_text2vec(test_text, vectorizer)

    # train classifier
    classifier = models[model]
    classifier.fit(train_data_features, train_class)

    # test classifier
    test_preds = classifier.predict(test_data_features)
    train_preds = cross_val_predict(classifier, train_data_features, train_class, cv=10)
    test_metrics = gutils.perf_metrics(test_class, test_preds)
    train_metrics = gutils.perf_metrics(train_class, train_preds)
    return 0


def plot_distribution(class_array, title):
    plt.figure(title)
    pd.DataFrame(class_array, columns=['Class']).Class.value_counts().plot(
        kind='pie',
        autopct='%.2f %%',
    )
    plt.axis('equal')
    plt.title(title)




