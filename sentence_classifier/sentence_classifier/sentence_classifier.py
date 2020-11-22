"""Main module."""
from sentence_classifier.sentence_classifier.simple_classifiers import *
from sentence_classifier.sentence_classifier.BERT import *
from sentence_classifier.sentence_classifier.NN import *

"""# Load data"""

df = pd.read_csv('https://raw.githubusercontent.com/KCLaurelie/NLP-playground/master/prometheus/imdb_5k_reviews.csv', header=1)
text_col = 'review'
label_col = 'sentiment'
df = df[0:100]
df[text_col] = df[text_col].apply(lambda x: x.strip().lower())
#df[label_col] = convert_to_cat(df[label_col], binary=False)['labels']
print('num annotations:', len(df), '\n\n', df[label_col].value_counts(), '\n\n', df[[label_col, text_col]].head())


"""# Run BERT"""
BERT_tokenizer = 'bert-base-uncased'

n_epochs =1 #5
res = train_BERT(sentences=df[text_col], labels=df[label_col], BERT_tokenizer=BERT_tokenizer,
                 test_size=0.2, n_epochs=n_epochs, output_dir=None, MAX_TKN_LEN=511)
res['stats']

#load_and_run_BERT(sentences=['hello my name is link i am in love with princess zelda', 'this is just a test sentence'], trained_bert_model=data_path+'bert_models', BERT_tokenizer='bert-base-uncased')

kf = BERT_KFOLD(sentences=df[text_col], labels=df[label_col], n_splits=10, BERT_tokenizer=BERT_tokenizer, n_epochs=1, random_state=666)
print(kf['stats'], '\n\n', kf['stats_classes'])


