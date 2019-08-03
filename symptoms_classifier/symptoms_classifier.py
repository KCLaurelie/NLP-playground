import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
from symptoms_classifier.NLP_embedding import convert_snt2avgtoken, tokenize_sentences
from symptoms_classifier.NLP_text_cleaning import preprocess_text
from code_utils.general_utils import super_read_csv
import symptoms_classifier.classifiers_utils as cutils

os.chdir(r'C:\Users\K1774755\PycharmProjects\toy-models\symptoms_classifier')
matplotlib.use('Qt5Agg')


class TextsToClassify:
    def __init__(self, filepath=None, dataset=pd.DataFrame(), class_col='class', text_col='text',
                 embedding_algo='w2v', embedding_model=None, binary=False, binary_main_class=1,
                 classifier_model='SVM'):
        """

        :param filepath: path to dataset if stored in file
        :param dataset: dataset itself if not stored in file
        :param class_col: column containing annotations
        :param text_col: column containing text
        :param embedding_algo: algo to use for embedding text (at the moment only word2vec supported)
        :param embedding_model: saved pre-trained embedding model
        :param binary: option to convert multiclasses into 1
        :param binary_main_class: class to keep when converting to binary
        :param classifier_model: classifier algo to use
        """
        self.filepath = filepath
        self.dataset = dataset
        self.class_col = class_col
        self.text_col = text_col
        self.embedding_algo = embedding_algo
        self.embedding_model = embedding_model
        self.binary = binary
        self.binary_main_class = binary_main_class
        self.classifier_model = classifier_model

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

        self.dataset = data
        print('data loaded')
        return data

    def tokenize_text(self, manually_clean_text=True, update_obj=True):
        sentences = self.dataset[self.text_col]
        tokenized_text = tokenize_sentences(sentences, manually_clean_text=manually_clean_text)
        if update_obj:
            self.dataset['tokenized_text'] = tokenized_text
            print('object updated with tokenized text, to view use self.dataset.tokenized_text')
        return tokenized_text

    def train_embedding_model(self, size=100, window=5, min_count=4, workers=4, clean_text=False
                              , save_model=True, update_obj=True):
        if 'tokenized_text' not in self.dataset.columns:
            self.tokenize_text(manually_clean_text=clean_text, update_obj=True)
        # train word2vec on tokenized text
        w2v = Word2Vec(self.dataset['tokenized_text'], size=size, window=window, min_count=min_count, workers=workers)
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

    def make_binary_class(self, binary_main_class=None):
        if binary_main_class is not None:
            self.binary_main_class = binary_main_class
        if self.binary_main_class is None:  # by default take value occuring the most
            self.binary_main_class = self.dataset[self.class_col].mode()[0]

        self.dataset['class_numeric'] = np.where(self.dataset[self.class_col] == self.binary_main_class, 1, 0)
        print(self.binary_main_class, 'used as main value (will be replaced by 1, other values replaced by zero)')

    def make_numeric_class(self):
        if sorted(list(self.dataset[self.class_col].unique())) == ['negative', 'neutral', 'positive']:
            self.dataset['class_numeric'] = self.dataset[self.class_col].replace(
                {'positive': 1, 'negative': -1, 'neutral': 0})
            return 0
        if self.dataset[self.class_col].dtypes == 'O':  # any other case
            lb_make = LabelEncoder()
            self.dataset['class_numeric'] = lb_make.fit_transform(self.dataset[self.class_col])
            return 0

    def run_classifier(self, classifier_model=None, binary=None, binary_main_class=None, test_size=0.2, save_model=True):
        # if object default values not overriden
        if binary is None: binary = self.binary
        if classifier_model is None: classifier_model = self.classifier_model

        # convert annotations to binary class if needed
        if binary:
            self.make_binary_class(binary_main_class=binary_main_class)
        else:
            self.make_numeric_class()
        text_class = self.dataset['class_numeric']

        x_emb_train, x_emb_test, y_train, y_test = train_test_split(self.embedded_text, text_class, test_size=test_size)

        # train classifier
        classifier = cutils.load_classifier(classifier_model)
        classifier.fit(x_emb_train, y_train)
        if save_model:
            cutils.save_classifier_to_file(classifier, filename=str(classifier_model), timestamp=True)

        # test classifier
        test_preds = classifier.predict(x_emb_test)
        train_preds = classifier.predict(x_emb_train)  # cross_val_predict(classifier, x_emb_train, y_train, cv=10)

        self.dataset.loc[y_train.index, 'split'] = 'train'
        self.dataset.loc[y_test.index, 'split'] = 'test'
        self.dataset.loc[y_train.index, 'preds'] = train_preds
        self.dataset.loc[y_test.index, 'preds'] = test_preds

        print(str(classifier), '\n',
              'CLASSES\n',
              self.dataset[['class_numeric', self.class_col]].drop_duplicates(),
              '\n\n',
              'TEST SET\n', 'accuracy:',
              accuracy_score(y_test, test_preds), '\n',
              classification_report(y_test, test_preds), '\n',
              'TRAIN SET\n', 'accuracy:',
              accuracy_score(y_train, train_preds), '\n',
              classification_report(y_train, train_preds)
              )
        return 0

    def run_all(self, manually_clean_text=True, embedding_model=None, classifier_model=None, binary=None,
                binary_main_class=None, test_size=0.2):
        self.load_data()
        self.tokenize_text(manually_clean_text=manually_clean_text, update_obj=True)
        if embedding_model is None:
            embedding_model = self.train_embedding_model(update_obj=True)
        self.embed_text(update_obj=True, embedding_model=embedding_model)
        self.run_classifier(classifier_model=classifier_model, binary=binary, binary_main_class=binary_main_class
                            , test_size=test_size)
        return 0

