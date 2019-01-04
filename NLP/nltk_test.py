import nltk
from nltk.corpus import names
import random


##########################################################################
## EXTRACT GENDER FROM LIST OF NAMES
##########################################################################
#feature extractor: extracts last letters of word
def gender_features(word):
    word=word.lower()
    return {'last_letter':word[-1:],
            'suffix_3':word[-3:]}

#list of examples and corresponding class labels
labeled_names=[(name,'male') for name in names.words('male.txt')]
labeled_names+=[(name,'female') for name in names.words('female.txt')]
len_set=len(labeled_names)
random.shuffle(labeled_names)

#use features extractor to process names data, and divide result in training/test sets
#use training set to train naive bayesian classifier
featuresets=[(gender_features(n),gender) for (n,gender) in labeled_names]
train_set=featuresets[int(len(featuresets)/2):]
test_set=featuresets[:int(len(featuresets)/2)]

#when large corpora: better to use apply_features to spare memory usage
from nltk.classify import apply_features
train_set=apply_features(gender_features,labeled_names[int(len_set/2):])
test_set=apply_features(gender_features,labeled_names[:int(len_set/2)])

#train/test classifier
classifier=nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Neo'))
nltk.classify.accuracy(classifier,test_set)

#show features most informative for classification (words ending in a 35.5x more likely to be female)
classifier.show_most_informative_features(5)

#error analysis: important to have separate dev set
devtest_set=apply_features(gender_features,labeled_names[500:1500])
train_set=apply_features(gender_features,labeled_names[1500:])
test_set=apply_features(gender_features,labeled_names[:500])
classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,devtest_set))

#errors in devtest_set
errors=[]
for (name,tag) in labeled_names[500:1500]:
    guess=classifier.classify(gender_features(name))
    if guess!=tag: errors.append(tag,guess,name)


#include more features
def gender_features_all(name):
    features={}
    features['first_letter']=name[0].lower()
    features['last_letter']=name[-1].lower()
    for letter in 'qwertyuiopasdfghjklzxcvbnm':
        features['count({})'.format(letter)]=name.lower().count(letter)
        features['has({})'.format(letter)]=(letter in name.lower())
        
    return features

#train/test classifier on extended features set
train_set=apply_features(gender_features_all,labeled_names[int(len_set/2):])
test_set=apply_features(gender_features_all,labeled_names[:int(len_set/2)])
classifier_all=nltk.NaiveBayesClassifier.train(train_set)
classifier_all.classify(gender_features_all('Neo'))
nltk.classify.accuracy(classifier_all,test_set)

##########################################################################
## FUNCTIONS TO CLEAN DOCUMENTS
##########################################################################
from nltk.corpus import stopwords
import string
def cleanupDoc(s):
    stopset = set(stopwords.words('english')+list(string.punctuation))
    tokens = nltk.word_tokenize(s)
    cleanup = " ".join(filter(lambda word: word not in stopset, s.split()))
    return cleanup
s = "I am going to disco and bar tonight!!"
tokens = nltk.word_tokenize(s)
x = cleanupDoc(s)

def cleanupWord(word,stopset=None):
    if stopset is None: stopset = set(stopwords.words('english')+list(string.punctuation))
    word=word.lower() 
    if word not in stopset: return word

##########################################################################
## DOCUMENT CLASSIFICATION
##########################################################################

from nltk.corpus import movie_reviews
stopset = set(stopwords.words('english')+list(string.punctuation))
documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories() #category: positive or negative review
           for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

documents_clean=[(list(cleanupDoc(movie_reviews.raw(fileid))),category)
           for category in movie_reviews.categories() #category: positive or negative review
           for fileid in movie_reviews.fileids(category)]
#features extractor: most frequently used words
all_words=nltk.FreqDist(w.lower() for w in movie_reviews.words() if w not in stopset)
word_features=list(all_words)[:15]

#check if word occurs in document (faster on set than list)
def document_features(document, words_list):
    document_words=set(document)
    features={}
    for word in words_list:
        features['contains({})'.format(word)]=(word in document_words)
    return features

document_features(movie_reviews.words('pos/cv957_8737.txt'),['good','movie'])

#find most informative features
featuresets=[(document_features(doc,word_features),category) for (doc,category) in documents]
train_set=featuresets[100:]
test_set=featuresets[:100]

#train and test classifier
classifier_movies=nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier_movies,test_set)
classifier_movies.show_most_informative_features(5)

##########################################################################
## PART OF SPEECH TAGGING WITH CONTEXT EXPLOITATION
##########################################################################

from nltk.corpus import brown

# list common suffixes
def common_suffixes(words_list,size=100):
    suffix_fdist=nltk.FreqDist()
    for word in words_list:
        word=word.lower()
        suffix_fdist[word[-1:]] +=1
        suffix_fdist[word[-2:]] +=1
        suffix_fdist[word[-3:]] +=1
    return [suffix for (suffix,count) in suffix_fdist.most_common(size)]

most_common_suffixes=common_suffixes(words_list=brown.words(),size=100)

#checks if given word ends with common suffix
def pos_features(word, common_suffixes=most_common_suffixes):
    features={}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)]=word.lower().endswith(suffix)
    return features

#classifies tagged words (verb, noun etc) depending on their suffix
tagged_words=brown.tagged_words(categories='news')
featuresets=[(pos_features(word),word_type) for (word,word_type) in tagged_words]
size=int(len(featuresets)*0.1)
train_set,test_set=featuresets[size:],featuresets[:size]

#train and test classifier
classifier_word_type=nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier_word_type,test_set)
classifier_word_type.classify(pos_features('cats'))
classifier_word_type.pseudocode(depth=4)
    



        



