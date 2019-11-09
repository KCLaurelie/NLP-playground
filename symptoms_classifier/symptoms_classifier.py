import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from symptoms_classifier.NLP_embedding import *
from symptoms_classifier.ANN import train_nn
from symptoms_classifier.CNN import train_cnn
from symptoms_classifier.RNN import train_rnn
from symptoms_classifier.NLP_text_cleaning import preprocess_text
import symptoms_classifier.classifiers_utils as cutils


class TextsToClassify:
    def __init__(self, filepath=None, dataset=pd.DataFrame(), class_col='class', text_col='text',
                 binary=False, binary_main_class=1,
                 already_split=False, split_col='split', test_val='test', train_val='train',
                 embedding_algo='w2v', embedding_model=None, embedded_text=None,
                 classifier_model='SVM', tokenization_type=None):
        """

        :param filepath: path to dataset if stored in file
        :param dataset: dataset itself if not stored in file
        :param class_col: column containing annotations
        :param text_col: column containing text
        :param split_col: column containing train/dev/test split
        :param embedding_algo: algo to use for embedding text (at the moment only word2vec supported)
        :param embedding_model: saved pre-trained embedding model
        :param binary: option to convert multiclasses into 1
        :param binary_main_class: class to keep when converting to binary
        :param classifier_model: classifier algo to use
        :param tokenization_type: (None, lem, clean, lem_stop)
        """
        self.tokenization_type = tokenization_type
        self.filepath = filepath
        self.dataset = dataset
        self.class_col = class_col
        self.text_col = text_col
        self.binary = binary
        self.binary_main_class = binary_main_class
        self.embedding_algo = embedding_algo
        self.embedding_model = embedding_model
        self.embedded_text = embedded_text
        self.classifier_model = classifier_model
        self.already_split = already_split
        self.train_val = train_val
        self.test_val = test_val
        self.split_col = split_col

    def load_data(self):
        if self.filepath is not None:
            if 'csv' in self.filepath:
                try:
                    data = pd.read_csv(self.filepath, header=0)
                except:
                    data = pd.read_csv(self.filepath, header=0, engine='python', encoding='ISO-8859-1')
            elif 'xls' in self.filepath:
                data = pd.read_excel(self.filepath, header=0)
            elif self.filepath.endswith('.txt'):
                data = preprocess_text(self.filepath, remove_stopwords=False, stemmer=None, lemmatizer=None,
                                       keywords=None, remove_punctuation=True)
                data = pd.DataFrame(data, columns=self.text_col)
            else:
                return 'unknown file format'
        else:
            data = self.dataset
        cols = [col for col in [self.class_col, self.text_col, self.split_col] if col in data.columns]
        data = data[cols]

        self.dataset = data
        self.make_numeric_class()
        print('data loaded')
        return data

    def tokenize_text(self, tokenization_type='lem', update_obj=True, output_file_path=None, remove_contractions=False):
        sentences = self.dataset[self.text_col]
        tokenized_text = tokenize_sentences(sentences, tokenization_type=tokenization_type
                                            , output_file_path=output_file_path, remove_contractions=remove_contractions)
        if update_obj:
            self.dataset['tokenized_text'] = tokenized_text
            self.__setattr__('tokenization_type', tokenization_type)
            print('object updated with tokenized text, to view use self.dataset.tokenized_text')
        return tokenized_text

    def train_embedding_model(self, embedding_algo='w2v', tokenization_type='lem', update_obj=True,
                              remove_contractions=True, **kwargs):
        if 'tokenized_text' not in self.dataset.columns:
            self.tokenize_text(tokenization_type=tokenization_type, update_obj=True, remove_contractions=remove_contractions)
        if embedding_algo is None:
            embedding_algo = self.embedding_algo

        w2v = train_embedding_model(sentences=self.dataset['tokenized_text'], embedding_algo=embedding_algo, **kwargs)

        if update_obj:
            self.__setattr__('embedding_model', w2v)
            self.__setattr__('embedding_algo', embedding_algo)
            print('object updated with embedding model, to view use self.embedding_model')

        return w2v

    def embed_text(self, embedding_algo=None, embedding_model=None, tokenization_type=None, remove_contractions=True
                   , update_obj=True, **kwargs):
        if embedding_model is None:
            embedding_model = self.embedding_model
        if embedding_algo is None:
            embedding_algo = self.embedding_algo
        if tokenization_type is not None:
            print('overriding previous tokenization with:', tokenization_type)
            self.tokenize_text(tokenization_type=tokenization_type, update_obj=True, remove_contractions=remove_contractions)
        elif (tokenization_type is None) and ('tokenized_text' not in self.dataset.columns):
            print('trying to detect tokenization from embedding model:', tokenization_type)
            tokenization_type = detect_tokenization_type(str(embedding_model))
            self.tokenize_text(tokenization_type=tokenization_type, update_obj=True, remove_contractions=remove_contractions)
        else:
            print('using tokenization previously saved:', self.tokenization_type)

        # TODO: extend to other models?
        embedded_text = sentences2embedding(tkn_sentences=self.dataset.tokenized_text,
                                            embedding_algo=embedding_algo, embedding_model=embedding_model, **kwargs)
        if update_obj:
            self.__setattr__('embedded_text', embedded_text)
            print('object updated with embedded text, to view use self.embedded_text')

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
                {'positive': 1, 'negative': 2, 'neutral': 0})
        else:
            self.dataset[self.class_col].replace({-1: 999}, inplace=True)
            lb_make = LabelEncoder()
            self.dataset['class_numeric'] = lb_make.fit_transform(self.dataset[self.class_col])
        return 0

    def get_train_test_split(self):
        x = self.dataset[self.text_col]
        y = self.dataset.class_numeric
        idx_train = self.dataset[self.dataset[self.split_col] == self.train_val].index
        idx_test = self.dataset[self.dataset[self.split_col] == self.test_val].index
        if len(idx_test) <= 1 or len(idx_train) <= 1:
            print('missing train/test values, check split column of dataset', self.train_val, self.test_val)
        return [idx_train, idx_test]

    def convert_class_2_numeric(self, binary=None, binary_main_class=None):
        # convert annotations to binary class if needed
        if binary:
            self.make_binary_class(binary_main_class=binary_main_class)
        else:
            self.make_numeric_class()
        return 0

    def generate_errors_report(self, preds_col, update_obj=True):
        self.dataset['error_pred'] = np.abs(self.dataset[preds_col] - self.dataset.class_numeric)
        errors = self.dataset.loc[self.dataset.error_pred > 0.5,
                                  [self.text_col, 'tokenized_text', 'class_numeric', preds_col, 'error_pred']] \
            .sort_values(by='error_pred')
        if update_obj:
            self.__setattr__('classifier_errors', errors)
        return errors

    def run_neural_net(self, binary=None, binary_main_class=None, multi_class=True, dropout=0.5
                       , output_errors=False, save_model_path=None, timestamp=False, nn_type='ANN', **kwargs):
        title = 'Neural Net' + ('_multiclass' if multi_class else '') + '_dropout=' + str(dropout)
        if self.embedding_model is None:
            print('no embedding model associated to the dataset, please assign one by doing self.embedding_model = ...')
            return 'error'
        if binary is None: binary = self.binary
        self.convert_class_2_numeric(binary=binary, binary_main_class=binary_main_class)
        if self.already_split and self.split_col in self.dataset.columns:
            print('dataset already split in train/test')
            idx_train, idx_test = self.get_train_test_split()
        else:
            idx_train, idx_test = [None, None]
        x_emb = self.embedded_text
        x = self.dataset[self.text_col]
        y = self.dataset.class_numeric

        if nn_type.lower() == 'ann':
            net, preds, df_test, df_train = train_nn(x_emb=x_emb, y=y, idx_train=idx_train, idx_test=idx_test,
                                                     multi_class=multi_class, dropout=dropout, **kwargs)
        elif nn_type.lower() == 'cnn':
            net, preds, df_test, df_train = train_cnn(w2v=self.embedding_model, sentences=x, y=y,
                                                      idx_train=idx_train, idx_test=idx_test, dropout=dropout, **kwargs)
        elif nn_type.lower() == 'rnn':
            net, preds, df_test, df_train = train_rnn(w2v=self.embedding_model, sentences=x, y=y, idx_train=idx_train, idx_test=idx_test,
                                                      dropout=dropout, **kwargs)
        else:
            print('model selected has not been implemented')
            return {'model': nn_type, 'report': ['model not implemented']}
        self.dataset[['split', 'preds']] = preds[['split', 'preds']]
        errors = self.generate_errors_report(preds_col='preds') if output_errors else 'error report not generated'
        classes = self.dataset[['class_numeric', self.class_col]].drop_duplicates()
        self.__setattr__('trained_NN', net)
        if save_model_path is not None:
            cutils.save_classifier_to_file(net, filename=save_model_path, timestamp=timestamp, model_type='nn')
        return {'model': net, 'report': [title, classes, df_test, df_train, errors]}

    def run_classifier(self, classifier_model=None, binary=None, binary_main_class=None, test_size=0.2, random_state=0
                       , save_model_path=None, output_errors=False):
        # if object default values not overriden
        if binary is None: binary = self.binary
        if classifier_model is None: classifier_model = self.classifier_model

        self.convert_class_2_numeric(binary=binary, binary_main_class=binary_main_class)
        text_class = self.dataset['class_numeric']

        if self.already_split:
            print('dataset already split in train/test')
            idx_train, idx_test = self.get_train_test_split()
            x_emb_train = self.embedded_text[idx_train]
            x_emb_test = self.embedded_text[idx_test]
            y_train = text_class[idx_train]
            y_test = text_class[idx_test]
        else:
            x_emb_train, x_emb_test, y_train, y_test = train_test_split(self.embedded_text, text_class, test_size=test_size, random_state=random_state)

        # train classifier
        classifier = cutils.load_classifier(classifier_model)
        title = str(classifier)
        if classifier == 'LightGBM':
            parameters = {
                'application': 'binary',
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': 'true',
                'boosting': 'gbdt',
                'num_leaves': 31,
                'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'bagging_freq': 20,
                'learning_rate': 0.05, 'verbose': 0
            }
            classifier = lgb.train(parameters, lgb.Dataset(x_emb_train, label=y_train), 100)
        else:
            try:
                classifier.fit(x_emb_train, y_train)
            except:
                return {'model': classifier, 'report': ['classification algo failed for:' + title]}

        self.__setattr__('trained_' + str(classifier_model), classifier)
        if save_model_path is not None:
            cutils.save_classifier_to_file(classifier, filename=save_model_path, timestamp=True)

        # test classifier
        test_preds = classifier.predict(x_emb_test)
        train_preds = classifier.predict(x_emb_train)  # cross_val_predict(classifier, x_emb_train, y_train, cv=10)

        self.dataset.loc[y_train.index, 'split'] = 'train'
        self.dataset.loc[y_test.index, 'split'] = 'test'
        self.dataset.loc[y_train.index, 'preds'] = train_preds
        self.dataset.loc[y_test.index, 'preds'] = test_preds

        errors = self.generate_errors_report(preds_col='preds') if output_errors else 'error report not generated'
        classes = self.dataset[['class_numeric', self.class_col]].drop_duplicates()
        df_test, df_train = cutils.formatted_classification_report(y_test, y_train, test_preds, train_preds)
        return {'model': classifier, 'report': [title, classes, df_test, df_train, errors]}

    def run_all(self, tokenization_type='lem', remove_contractions=True, embedding_model=None, classifier_model=None,
                binary=None, binary_main_class=None, test_size=0.2):
        self.load_data()
        self.tokenize_text(tokenization_type=tokenization_type, update_obj=True, remove_contractions=remove_contractions)
        if embedding_model is None:
            embedding_model = self.train_embedding_model(update_obj=True)
        self.embed_text(update_obj=True, embedding_model=embedding_model)
        self.run_classifier(classifier_model=classifier_model, binary=binary, binary_main_class=binary_main_class
                            , test_size=test_size)
        return 0

