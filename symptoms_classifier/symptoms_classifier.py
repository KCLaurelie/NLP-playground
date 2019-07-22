import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import naive_bayes, svm, tree, ensemble, linear_model, neighbors
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
from symptoms_classifier.NLP_embedding import fit_text2vec, transform_text2vec, convert_snt2avgtoken, tokenize_sentences
from symptoms_classifier.NLP_text_cleaning import preprocess_text
from longitudinal_models.general_utils import super_read_csv
import symptoms_classifier.general_utils as gutils

os.chdir(r'C:\Users\K1774755\PycharmProjects\toy-models\symptoms_classifier')
matplotlib.use('Qt5Agg')

classifiers = {
    'Multinomial NB': naive_bayes.MultinomialNB(),
    'Gaussian NB': naive_bayes.GaussianNB(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': tree.DecisionTreeClassifier(random_state=0),
    'Random Forest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs=10),
    'Logistic Regression': linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial'),
    'Linear SVM': svm.LinearSVC(multi_class='crammer_singer'),
    'SVM': svm.SVC(gamma='scale', class_weight='balanced'),
    'SVM with linear kernel': svm.SVC(kernel='linear'),
    'SVM with sigmoid kernel': svm.SVC(kernel='sigmoid'),
    'SVM with poly kernel': svm.SVC(kernel='poly')
}


class TextsToClassify:
    def __init__(self, filepath=None, dataset=pd.DataFrame(), class_col='class', text_col='text',
                 embedding_algo='w2v', embedding_model=None, binary=False,
                 classifier_type='SVM'):
        """

        :param filepath: path to dataset if stored in file
        :param dataset: dataset itself if not stored in file
        :param class_col: column containing annotations
        :param text_col: column containing text
        :param embedding_algo: algo to use for embedding text (at the moment only word2vec supported)
        :param embedding_model: saved pre-trained embedding model
        :param binary: option to convert multiclasses into 1
        :param classifier_type: classifier algo to use
        """
        self.filepath = filepath
        self.dataset = dataset
        self.class_col = class_col
        self.text_col = text_col
        self.embedding_algo = embedding_algo
        self.embedding_model = embedding_model
        self.binary = binary
        self.classifier_type = classifier_type

    def load_data(self):
        if self.filepath is not None:
            if 'csv' in self.filepath:
                try:
                    data = pd.read_csv(self.filepath, header=0)
                except:
                    data = super_read_csv(self.filepath)
            elif self.filepath.endswith('.txt'):
                data = preprocess_text(self.filepath, remove_stopwords=False, stemmer=None, lemmatizer=None,
                                       keywords=None, remove_punctuation=True)
                data = pd.DataFrame(data, columns=self.text_col)
            else:
                return 'unknown file format'
        else:
            data = self.dataset
        cols = [col for col in [self.class_col, self.text_col] if col in data.columns]
        data = data[cols]
        if self.class_col in data.columns:  # convert classes to numbers
            # most common case
            self.dataset['class_orig'] = data[self.class_col].copy()
            data[self.class_col] = data[self.class_col].replace({'positive': 1, 'negative': -1, 'neutral': 0})
            if data[self.class_col].dtypes == 'O':  # any other case
                lb_make = LabelEncoder()
                data[self.class_col] = lb_make.fit_transform(data[self.class_col])

        self.dataset = data
        print('data loaded')
        return data

    def tokenize_text(self, manually_clean_text=True, update_obj=True):
        sentences = self.dataset[self.text_col]
        tokenized_text = tokenize_sentences(sentences, manually_clean_text=manually_clean_text)
        if update_obj:
            self.__setattr__('tokenized_text', tokenized_text)
            print('object updated with tokenized text, to view use self.tokenized_text')
        return tokenized_text

    def train_embedding_model(self, size=100, window=5, min_count=4, workers=4, clean_text=False
                              , save_model=True, update_obj=True):
        if self.tokenized_text is None:
            tok_snts = self.tokenize_text(manually_clean_text=clean_text)
        else:
            tok_snts = self.tokenized_text
        w2v = Word2Vec(tok_snts, size=size, window=window, min_count=min_count, workers=workers)  # train word2vec
        if save_model:
            w2v.save('word2vec.model')
        if update_obj:
            self.__setattr__('embedding_model', w2v)
            print('object updated with embedding model, to view use self.embedding_model')
        # TODO: add other embedding models?
        return w2v

    def embed_text(self, embedding_model=None, update_obj=True):
        if embedding_model is None:
            embedding_model = self.embedding_model
        embedded_text = convert_snt2avgtoken(sentences=self.dataset[self.text_col],
                                             w2v_model=embedding_model,
                                             clean_text=True)
        if update_obj:
            self.__setattr__('embedded_text', embedded_text)
            print('object updated with embedded text, to view use self.embedded_text')
        # TODO: extend to other models?
        return embedded_text

    def run_classifier(self, classifier_type=None, binary=None, test_size=0.2, save_model=True):
        # if object default values not overriden
        if binary is None: binary = self.binary
        if classifier_type is None: classifier_type = self.classifier_type
        embedded_text = self.embedded_text
        text_class = self.dataset[self.class_col]

        # convert annotations to binary class if needed
        if binary:
            if 'class_binary' not in self.dataset.columns:
                most_common_val = text_class.mode()[0]
                replace_val = 0 if most_common_val != 0 else 1
                self.dataset['class_binary'] = np.where(text_class == most_common_val, most_common_val, replace_val)
            text_class = self.dataset['class_binary']

        x_emb_train, x_emb_test, y_train, y_test = train_test_split(embedded_text, text_class, test_size=test_size)

        # train classifier
        classifier = classifiers[classifier_type]
        if save_model:
            gutils.save_classifier(classifiers, filename=str(self.classifier_type), timestamp=True)
            print('model saved')
        classifier.fit(x_emb_train, y_train)

        # test classifier
        test_preds = classifier.predict(x_emb_test)
        train_preds = classifier.predict(x_emb_train)  # cross_val_predict(classifier, x_emb_train, y_train, cv=10)

        self.dataset.loc[y_train.index, 'split'] = 'train'
        self.dataset.loc[y_test.index, 'split'] = 'test'
        self.dataset.loc[y_train.index, 'preds'] = train_preds
        self.dataset.loc[y_test.index, 'preds'] = test_preds

        test_metrics = gutils.perf_metrics(y_test, test_preds)
        test_metrics['type'] = 'test'
        train_metrics = gutils.perf_metrics(y_train, train_preds)
        train_metrics['type'] = 'train'
        return [{'classes': classifier.classes_}, test_metrics, train_metrics]

    def run_all(self):
        self.load_data()
        self.train_embedding_model(update_obj=True)
        self.embed_text(update_obj=True)
        res = self.run_classifier()
        return res


def run_classifier(classifier_type, train_data, test_data):
    train_text = preprocess_text(train_data['text'], remove_stopwords=True, stemmer=None, lemmatizer=None)
    test_text = test_data[['text']]
    train_class = train_data[['class']]
    test_class = test_data[['class']]

    # vectorize text data
    vectorizer = fit_text2vec(train_text)
    train_data_features = transform_text2vec(train_text, vectorizer)
    test_data_features = transform_text2vec(test_text, vectorizer)

    # train classifier
    classifier = classifiers[classifier_type]
    classifier.fit(train_data_features, train_class)

    # test classifier
    test_preds = classifier.predict(test_data_features)
    train_preds = cross_val_predict(classifier, train_data_features, train_class, cv=10)
    test_metrics = gutils.perf_metrics(test_class, test_preds)
    train_metrics = gutils.perf_metrics(train_class, train_preds)
    return [test_metrics, train_metrics]


def plot_distribution(class_array, title):
    plt.figure(title)
    pd.DataFrame(class_array, columns=['Class']).Class.value_counts().plot(
        kind='pie',
        autopct='%.2f %%',
    )
    plt.axis('equal')
    plt.title(title)
