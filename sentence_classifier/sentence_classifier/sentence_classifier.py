"""Main module."""
from sentence_classifier.sentence_classifier.simple_classifiers import *
from sentence_classifier.sentence_classifier.BERT import *
from sentence_classifier.sentence_classifier.NN import *

"""# Load data"""

df = pd.read_csv('https://raw.githubusercontent.com/KCLaurelie/NLP-playground/master/prometheus/mimic_status_10folds.csv')
text_col = 'clean_text'
label_col = 'annotation'
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
                 test_size=0.2, n_epochs=n_epochs, output_dir=None, MAX_TKN_LEN=511)
res['stats']

#load_and_run_BERT(sentences=['hello my name is link i am in love with princess zelda', 'this is just a test sentence'], trained_bert_model=data_path+'bert_models', BERT_tokenizer='bert-base-uncased')

kf = BERT_KFOLD(sentences=df[text_col], labels=df[label_col], n_splits=10, BERT_tokenizer=BERT_tokenizer, n_epochs=1, random_state=666)
print(kf['stats'], '\n\n', kf['stats_classes'])


