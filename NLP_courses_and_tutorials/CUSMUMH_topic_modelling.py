## First we need to import all the necessary packages

import string
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import itertools
import zipfile
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import os
import pandas as pd

import re
import codecs
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
import zipfile

import matplotlib.pyplot as plt
import seaborn as sns

# 1: corpus
# The first step in building a topic model is to read a corpus, or a collection of documents.
# In this example, we are using documents from http://www.mtsamples.com/.
# These are transcribed medical transcriptions sample reports and examples from a variety of clinical disciplines, such as radiology, surgery, discharge summaries. Note that one document can belong to several categories.
# We will save each document, all its words, and which clinical specialty it belongs to, in a dataframe.

data_folder = r"""C:\Users\K1774755\King's College London\Cognitive Impairment in Schizophrenia - Documents\Courses\CUSMUMH\week 7 - NLP_courses_and_tutorials with nltk & spacy"""

## get all documents - the data folder contains a zip file with nested folders and .txt documents
d = os.path.join(data_folder, 'mtsamples_for_topic_model.zip')

## let's create an empty list to store the documents in
all_text = []
all_words = []
specialties = []
filenames = []

## read in the zip directory
zip = zipfile.ZipFile(d, 'r')

print('reading the data')
## loop over all files in the zip and save in the list
for z in zip.namelist():
    to_save = z.split(str('/'))
    f = zip.open(z)
    if f.name.endswith('.txt'):
        specialties.append(to_save[1])
        filenames.append(to_save[2])
        ff = f.read().splitlines()
        txt = ''.join(fff.decode("utf-8") for fff in ff)
        aa = [''.join(c.lower() for c in s if c not in string.punctuation) for s in nltk.word_tokenize(txt)]
        aa = [x for x in aa if x]
        if len(aa) > 0:
            all_words.append(aa)

        all_text.append(txt)
    f.close()
print('done')

# put everything in a dataframe
d = {'Category': specialties, 'Document Name': filenames, 'Document Content': all_text, 'Document words': all_words}
df = pd.DataFrame(d)
df.reset_index(inplace=True)

# How many documents are in the data?
len(df)
# How many clinical specialties are in the data?
df['Category'].value_counts()
df['Category'].value_counts()


# 2 Using gensim and pyLDAVis
# We now need to generate representations for the vocabulary (dictionary) and the text collection (corpus)
# Let's use some functions that we can call later, and that we can modify later if we want.
# (Using all the words in the whole corpus or text collection is not typically what you want, because very common words,
# or very rare words will not generate good topic representations. Why?
# What parameters and configurations could be interesting to change below?)

## functions from http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb
## this function returns a set of stopwords predefined in the nltk package

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))


## this function prepares the data and returns a dictionary and a corpus.
## which parameters do you think would be worth modifying/experimenting with?
def prep_corpus(docs, additional_stopwords=set(), no_below=5, no_above=0.5):
    print('Building dictionary...')
    dictionary = Dictionary(docs)
    stopwords = nltk_stopwords().union(additional_stopwords)
    stopword_ids = map(dictionary.token2id.get, stopwords)
    dictionary.filter_tokens(stopword_ids)
    dictionary.compactify()
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.compactify()

    print('Building corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus

## now, let's use the functions we defined above to get our dictionary and corpus
dictionary, corpus = prep_corpus(df['Document words'])

## Now we have our dictionary and corpus, let's generate an LDA model.
## The LDA model has many parameters that can be set, all available parameters can be found here:
## https://radimrehurek.com/gensim/models/ldamodel.html
## Here, we've set the number of topics to 20.
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=10)

## You can also save the generated model to disk if you want
#lda.save('/Users/sumithra/DSV/MeDESTO/teaching/Farr2017/data/gensim_topic_model_data/mtsamples_20_lda.model')

## you can now look at these topics by printing them from the generated model
lda.print_topics()

## It can be hard to get a good understanding of what's actually in these topics
## Visualizations are very helpful for this, let's use a package that does this:
vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.display(vis_data)

###############################################################################################################
# 3 Using sklearn and comparing with 'existing' categories
# Now you have seen how you can build a topic models with gensim and look at the contents visually with pyLDAVis.
# You can also use sklearn for topic modeling, both lda and nmf, and analyse results visually by comparing with existing categories, if you have them.
# NMF approaches can be very efficient, particularly with smaller datasets. Let's see what you think.

# We need a couple of functions to visualise the data
# Preparation for visualisation
# Written by Sonia Priou

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def display_topic_representation(model, dataframe):
    doc_topic = model  # example : model = lda_Tfidf.transform(tfidf)
    doc = np.arange(doc_topic.shape[0])
    no_topics = doc_topic.shape[1]
    dico = {'index': doc}
    for n in range(no_topics):
        dico["topic" + str(n)] = doc_topic[:, n]

    # Max topic
    Topic_max = []
    for i in range(doc_topic.shape[0]):
        Topic_max.append(doc_topic[i].argmax())
    dico["Topic most represented"] = Topic_max
    df_topic = pd.DataFrame(dico)

    # Link both DataFrame
    df_result = pd.merge(dataframe, df_topic, on='index')

    # Finding within the cluster found by LDA the original file
    fig, ax = plt.subplots()
    sns.set_style('whitegrid')
    sns.countplot(x='Topic most represented', data=df_result, hue='Category')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def display_file_representation(model, dataframe):
    # Within a file, what is the slipt between topics found
    doc_topic = model  # example : model = lda_Tfidf.transform(tfidf)
    doc = np.arange(doc_topic.shape[0])
    no_topics = doc_topic.shape[1]
    topic = np.arange(no_topics)
    dico = {'index': doc}
    for n in range(no_topics):
        dico["topic" + str(n)] = doc_topic[:, n]
    # Max topic
    Topic_max = []
    for i in range(doc_topic.shape[0]):
        Topic_max.append(doc_topic[i].argmax())
    dico["Topic most represented"] = Topic_max
    df_topic = pd.DataFrame(dico)

    # Link both DataFrame
    df_result = pd.merge(dataframe, df_topic, on='index')

    dico2 = {'Topic': topic}
    for i in df_result['Category'].value_counts().index:
        ser = df_result.loc[df_result['Category'] == i].mean()
        score = ser[1:no_topics + 1]
        dico2[i] = score

    df_score = pd.DataFrame(dico2)
    print('For each given file, we calculate the mean percentage of the documents depence to each topic')
    print('')
    print(df_score.head())

    fig, axs = plt.subplots(ncols=len(df_smaller['Category'].value_counts()))
    count = 0
    for i in df_result['Category'].value_counts().index:
        sns.barplot(x='Topic', y=i, data=df_score, ax=axs[count])
        count = count + 1

    plt.tight_layout()

# Let's look at a smaller sample, to make the analysis a bit easier. You can choose other categories of course!
categories_to_keep = ['17-dentistry', '46-ophthalmology', '72-psychiatrypsychology', '71-podiatry']
df_smaller = df.loc[df['Category'].isin(categories_to_keep)]
df_smaller['index'] = range(0,len(df_smaller))
df_smaller.head()

#Now let's use sklearn's function for converting corpora to document-term-matrices
stopwords = nltk.corpus.stopwords.words('english')
min_df = 5
max_df = 100000
lowercase = True
ngram_range = 2

bow_transformer = CountVectorizer(stop_words=stopwords,
                                  min_df=min_df,
                                  max_df=max_df,
                                  lowercase = lowercase).fit(df['Document Content'])
document_bow = bow_transformer.transform(df_smaller['Document Content'])
feature_names = bow_transformer.get_feature_names()
tfidf_transformer=TfidfTransformer().fit(document_bow)
document_tfidf= tfidf_transformer.transform(document_bow)

#How many topics do you want the model to generate? How many discriminative words from each topic do you want to look at?

no_topics = 4
no_top_words = 10
#Now let's build an lda model

lda = LatentDirichletAllocation(n_components=no_topics).fit(document_tfidf)
#Let's look at the most discriminative words for each topic. Do you see a pattern? Do you think more work needs to be done with the underlying representation?

display_topics(lda,feature_names, no_top_words)
#We can now look at the main topic for each document. Does this look reasonable to you?

print('Representation of the main topic for each document')
display_topic_representation(lda.transform(document_tfidf),df_smaller)

#Now let's look at the distribution of topics in the files in relation to the 'existing' categories

display_file_representation(lda.transform(document_tfidf),df_smaller)
#Now let's compare with NMF.

nmf = NMF(n_components=no_topics,
          random_state=1,
          alpha=.1,
          l1_ratio=.5,
          init='nndsvd').fit(document_tfidf)

W = nmf.transform(document_tfidf)
H = nmf.components_
display_topics(nmf, feature_names, no_top_words)
print('Representation of the main topic for each document')
display_topic_representation(W,df_smaller)
display_file_representation(W,df_smaller)