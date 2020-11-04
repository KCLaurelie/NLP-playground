########################################################################
# initializing paths and input data parameters
########################################################################
import os
data_path = '/home/ZKraljevic/data/covid_anxiety/'
from NLP_utils import *
from simple_classifiers import *
from BERT import *
from NN import *

"""# Load data"""

df = pd.read_csv(data_path + annotations_file) if 'csv' in annotations_file else pd.read_excel(data_path + annotations_file)
# df = load_and_clean_data(filename, text_col, label_col, strip=True, MAX_LEN=20, clean_labels=True, binary=True, pos_col=None, context=15)
df = df[0:100]
df[text_col] = df[text_col].apply(lambda x: x.strip())
#df[label_col] = df[label_col].fillna('irrelevant').replace({'irrelevant':0, 'affirmed':1, 'negated':0})
#df[label_col] = convert_to_cat(df[label_col], binary=False)['labels']
print('num annotations:', len(df), '\n\n', df[label_col].value_counts(), '\n\n', df[[label_col, text_col]].head())

tokenize_spacy(df=pd.Series(['hello my name is link', 'this is cool']), tokenization_type='clean', outfile=None)

"""# Run BERT"""

BERT_tokenizer = 'bert-base-uncased'
#BERT_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'
#BERT_tokenizer = 'monologg/biobert_v1.0_pubmed_pmc'

n_epochs =1 #5
res = train_BERT(sentences=df[text_col], labels=df[label_col], BERT_tokenizer=BERT_tokenizer, 
                 test_size=0.2, n_epochs=n_epochs, output_dir=data_path+'bert_models', MAX_TKN_LEN=511)
res['stats']

load_and_run_BERT(sentences=['hello my name is link i am in love with princess zelda', 'this is just a test sentence'], trained_bert_model=data_path+'bert_models', BERT_tokenizer='bert-base-uncased')

kf = BERT_KFOLD(sentences=df[text_col], labels=df[label_col], n_splits=10, BERT_tokenizer=BERT_tokenizer, n_epochs=1, random_state=666)
print(kf['stats'], '\n\n', kf['stats_classes'])

"""# Load embedding (For NN or traditional classifiers - not needed for BERT)"""

# load embeddings model
emb_model = KeyedVectors.load(emb_model_file)
emb_model.wv.most_similar('great', topn=5)
emb_model_torch = embedding2torch(emb_model, SEED=0)

"""# Run SVM"""

classifiers = [
               ensemble.RandomForestClassifier(n_estimators=150, max_depth=None, class_weight='balanced'),
               svm.LinearSVC(multi_class='crammer_singer', class_weight='balanced'),
               linear_model.LogisticRegressionCV(class_weight='balanced', solver='liblinear'),
               neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
               ]

"""## train, save and load model with TFIDF"""

# train model on dataset
res = train_classifier(sentences=df[text_col], labels=df[label_col], emb_model='tfidf', 
                       classifier=classifiers[1], test_size=0.2, output_dir=data_path+'ML_models/imdb_rf')
print(res['stats'], '\n',
      res['stats_classes']
      )

res['stats_classes']

res = classifier_KFOLD(sentences=df[text_col], labels=df[label_col], classifier=classifiers[1], emb_model='tfidf', n_splits=2, output_dir=data_path+'ML_models/imdb_rf2')

# load saved model and test on new data
load_and_run_classifier(sentences=df[text_col][0:2], 
                        trained_classifier=data_path+'ML_models/imdb_rf', 
                        emb_model=data_path+'ML_models/imdb_rf_tfidf.pickle')

"""## train, save and load model with GloVE"""

# train model on dataset
res = train_classifier(sentences=df[text_col], labels=df[label_col], emb_model=emb_model, 
                       classifier=classifiers[1], test_size=0.2, 
                       output_dir=None)
res['stats']

# load saved model and test on new data
load_and_run_classifier(sentences=df[text_col][0:2], 
                        trained_classifier=data_path+'ML_models/imdb_rf_glove', 
                        emb_model=emb_model)

"""# Run ANN"""

n_epochs = 2 #1000
nn_model=ANN(embeddings=emb_model_torch['embeddings'], final_layer_neurons=df[label_col].nunique(), debug_mode=False) 
print(nn_model)
res = train_NN(sentences=df[text_col], labels=df[label_col], nn_model=nn_model,  emb_model=emb_model_torch, tokenization_type='clean',
                SEED=0, test_size=0.2, n_epochs=n_epochs, output_dir=data_path+'ML_models/imdb_ann')

pd.DataFrame.from_dict(res['stats'],orient='index', columns=['a'])

res = NN_KFOLD(nn_model=nn_model, sentences=df[text_col], labels=df[label_col], emb_model=emb_model_torch, n_splits=2, output_dir=data_path+'ML_models/imdb_ann')

print(res['stats_classes'])

load_and_run_NN(sentences=df[text_col][0:10], 
                trained_nn_model='gdrive/My Drive/Colab Notebooks/prometheus/datasets/ML_models/imdb_ann', 
                emb_model=emb_model_torch)

"""# Run LSTM"""

n_epochs = 20
rnn_model=RNN(emb_model_torch['embeddings'], padding_idx=emb_model_torch['word2id']['<PAD>']
              , rnn_type='lstm', bid=True, simulate_attn=True, debug_mode=False)
print(rnn_model)
res = train_NN(rnn_model, sentences=df[text_col], labels=df[label_col], emb_model=emb_model_torch, tokenization_type='clean'
                , SEED=0, test_size=0.2, n_epochs = n_epochs, output_dir=None)
print(res['stats'])

"""# Run CNN"""

n_epochs = 100
cnn_model=CNN(embeddings=emb_model_torch['embeddings'])
print(cnn_model)
res = train_NN(cnn_model, sentences=df[text_col], labels=df[label_col], emb_model=emb_model_torch, tokenization_type='clean',
                SEED=0, test_size=0.2, n_epochs = n_epochs, output_dir=None)
print(res['stats'])