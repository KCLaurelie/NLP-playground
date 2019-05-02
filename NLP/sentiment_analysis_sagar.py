## Creating a Stigma Classification Model for Tweets

import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
pd.set_option('display.max_columns', 500)


##########################################
## (0) Import Data
##########################################
fn = r'C:\Users\K1774755\PycharmProjects\data\tweet_data_testing.csv'
data = pd.read_csv(fn, header=None, encoding='Latin1', names=['label', 'tweet_nb', 'date', 'topic', 'user', 'body_text'])
data = data[['label', 'body_text']]
print("The first 5 rows of the dataset look like:")
data.head()

print("the number of rated tweets by category:")
data.groupby('label')['body_text'].nunique()
data.label.replace('o' , "0", inplace=True)


## Now we are making this a binary classification problem by only keeping the 0s and 2s.
# We will also drop the tweets with missing labels
# 0 = Not stigmatizing
# 1 = Stigmatizing
# 0 = negative
# 2 = neutral
# 4 = positive

data = data[data.label != '1']
data['label'] = data.label.replace('2', '1')
data = data.dropna()
print("the data after pre-processing and cleaning, which includes converting the classifications into a binary model, looks like:")

##########################################
## (0) Feature extraction
##########################################

# Calculate sentiment and subjectivity for each tweet. We will use these values as features in our model.
# i.e. are tweets that are classed as stigmatizing more negative?
def sentAnal(df):
    for index, row in df.iterrows():
        temp = TextBlob(row['body_text'])
        df.loc[index,'Sentiment'] = temp.sentiment.polarity
        df.loc[index,'Subjectivity'] = temp.sentiment.subjectivity
    return df


data = sentAnal(data)


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100


data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

# Average Word Length. simply take the sum of the length of all the words and divide it by the total length of the tweet as defined in function above
data['avg_word'] = data['body_text'].apply(lambda x: avg_word(x))
# Number of Words in tweet
data['word_count'] = data['body_text'].apply(lambda x: len(str(x).split(" ")))
# Number of characters. Here, we calculate the number of characters in each tweet. This is done by calculating the length of the tweet.
data['char_count'] = data['body_text'].str.len() ## this also includes spaces
# number of special characters like hashtags. we make use of the ‘starts with’ function because hashtags (or mentions) always appear at the beginning of a word.
data['hastags'] = data['body_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
# number of numerics in tweet
data['numerics'] = data['body_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
# number of UPPERCASE words. Anger or rage is quite often expressed by writing in UPPERCASE words which makes this a necessary operation to identify those words.
data['upper'] = data['body_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

## Add spelling correction? This takes ages and maybe we lose meaning?
#data['body_text'].apply(lambda x: str(TextBlob(x).correct()));

## remove rare words?
freq = pd.Series(' '.join(data['body_text']).split()).value_counts()[-10:]

freq = list(freq.index)
data['body_text'] = data['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

print("lets correlate our features. Here is a correlation matrix of our features:")
corr = data.corr()
corr.style.background_gradient()
print("label 0 = not stigmatizing tweets")
print("label 1 = stigmatizing tweets")
print("higher sentiment value = positive sentiment")
sns.catplot(data = data, x = 'label' , y = 'Sentiment', height = 3, kind = 'bar');
sns.catplot(data = data, x = 'label' , y = 'Subjectivity', height = 3, kind = 'bar')

##########################################
# (0) Split Data into training/testing sets
##########################################
#  making cols variable so that we have all feature columns names except our label
cols = data[data.columns.difference(["label"])].columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[cols], data['label'], test_size=0.2)

print("The size of each training and testing datasets are:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

##########################################
# (0) vectorize text
##########################################
#Using inverse document frequency weighting (TF_IDF). This vectorizing method still creates a document term matrix (i.e. one row per tweet and columns representing single unique words), but with tfidf, instead of the cells representing the count, they represent a weighting that's meant to identify how important a word is to an individual tweet. The rarer the word is, the higher the tfifd value will be. This method helps to pull out important (but seldom used) words.

cols = data[data.columns.difference(["label", "body_text"])].columns

#Instantiate the v
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])

tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])

X_train_vect = pd.concat([X_train[cols].reset_index(drop=True),
           pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vect.get_feature_names())], axis=1)

X_test_vect = pd.concat([X_test[cols].reset_index(drop=True),
           pd.DataFrame(tfidf_test.toarray(), columns=tfidf_vect.get_feature_names())], axis=1)

print(X_train_vect.shape)
print("Our vectorized data looks like:")
X_train_vect

print("the number of times each word appears in our document term matrix:")
tfidf_vect_fit.vocabulary_


'''
Make and evaluate models
To evaluate each model, we will use three evaluation metrics:
(1) Accuracy = # predicted correctly / total # of observations
(2) Precision = # predicted as 1 that are actually 1 (i.e. true positives) / total # that the model predicted as 1.
(3) Recall = # predicted to be 1 that are actually 1 (i.e. true positives; the same numerator as precision) / total # that are actually 1 (instead of the total number that are predicted as 1)
'''

##########################################
# (1) Random Forest Classifier
##########################################
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time

rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1, random_state=0)
# n_estimators  is how many decision trees that will be built within your random forest, so the default is 10. These defaults mean, your random forest would build 10 decision trees of unlimited depth. Then, there would be a vote among these 10 trees to determine the final prediction.
# max depth basically means that it will build each decision tree until it minimizes some loss criteria.

start = time.time()
rf_model = rf.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = rf_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1)
print('Fit time: {} / Predict time: {} ----\n\n **Precision: {} \n\n **Recall: {} \n\n **Accuracy: {}'.format(
    fit_time, pred_time, precision.mean(), recall.mean(), (y_pred==y_test).sum()/len(y_pred)))
print("The most important features are:")
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)#[0:10]

##########################################
# (2) Random Forest with Gradient Boost
##########################################

gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)

start = time.time()
gb_model = gb.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = gb_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average= 'weighted')
print('Fit time: {} / Predict time: {} ----\n\n **Precision: {} \n\n **Recall: {} \n\n **Accuracy: {}'.format(
    fit_time, pred_time, precision, recall, ((y_pred==y_test).sum()/len(y_pred))))

print("The most important features are:")
sorted(zip(gb_model.feature_importances_, X_train.columns), reverse=True)#[0:10]


##########################################
## (3) K Nearest Neighbour
##########################################
from sklearn.neighbors import KNeighborsClassifier
knClas = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)
#Fit the model using X as training data and y as target values
knClas.fit(X_train_vect, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
           weights='uniform')
# Predict the class labels for the provided data
pred = knClas.predict(X_test_vect)
len(pred)
from sklearn.metrics import accuracy_score
kn_accuracy = accuracy_score( y_test,pred)

print('Accuracy of K Nearest Neighbours model: {}%'.format(kn_accuracy*100))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


from sklearn.metrics import average_precision_score,precision_score, recall_score
kn_precision1 = pd.DataFrame(y_test).reset_index().astype(float)
kn_precision2 = pd.DataFrame(pred).astype(float)
kn_precision = pd.concat([kn_precision1, kn_precision2], axis = 1).rename(columns = {0: 'label_comp'})
## Kn precision score
# Precision = # predicted as stigmatizing that are actually stigmatizing / total # predicted as stigmatizing

kn_precision_score = (((kn_precision['label']== 1) & (kn_precision['label_comp']==1)).sum()) / ((kn_precision['label_comp']==1).sum()) * 100
print('Precision of K Nearest Neighbours model: {}%'.format(kn_precision_score))

# Kn recall score
# Recall = # predicted as stigmatizing that are actually stigmatizing / total # that are actually stigmatizing

kn_recall_score = (((kn_precision['label']== 1) & (kn_precision['label_comp']==1)).sum()) / ((kn_precision['label']==1).sum()) * 100
print('Recall of K Nearest Neighbours model: {}%'.format(kn_recall_score))

##########################################
## 4) Support Vector Machine (SVM) Classifier
##########################################
from sklearn.svm import SVC
svmClas = SVC()
svmClas.fit(X_train_vect, y_train);
pred = svmClas.predict(X_test_vect);
svmAccuracy = accuracy_score( y_test,pred)
print('Accuracy of SVM model: {}%'.format(svmAccuracy*100))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


svm_precision1 = pd.DataFrame(y_test).reset_index().astype(float)
svm_precision2 = pd.DataFrame(pred).astype(float)
svm_precision = pd.concat([svm_precision1, svm_precision2], axis = 1).rename(columns = {0: 'label_comp'})

## svm precision score
# Precision = # predicted as stigmatizing that are actually stigmatizing / total # predicted as stigmatizing

svm_precision_score = (((svm_precision['label']== 1) & (svm_precision['label_comp']==1)).sum()) / ((svm_precision['label_comp']==1).sum()) * 100
print('Precision of svm model: {}%'.format(svm_precision_score))
# svm recall score
# Recall = # predicted as stigmatizing that are actually stigmatizing / total # that are actually stigmatizing

svm_recall_score = (((svm_precision['label']== 1) & (svm_precision['label_comp']==1)).sum()) / ((svm_precision['label']==1).sum()) * 100
print('Recall of svm model: {}%'.format(svm_recall_score))

##########################################
## (5) SVM with a linear kernel
##########################################
svmClas = SVC(kernel='linear')
svmClas.fit(X_train_vect, y_train)
pred = svmClas.predict(X_test_vect)
svmAccuracy = accuracy_score( y_test,pred)
print('Accuracy of linear SVM classifier: {}%'.format(svmAccuracy*100))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

##########################################
## (6) SVM with a sigmoid kernel
##########################################
# Now trying new parameters
svmClas = SVC(kernel='sigmoid')
svmClas.fit(X_train_vect, y_train)
pred = svmClas.predict(X_test_vect)
svmAccuracy = accuracy_score( y_test,pred)
print('Accuracy of linear SVM classifier: {}%'.format(svmAccuracy*100))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

##########################################
## (7) SVM with a poly kernel
##########################################
# Now trying new parameters
svmClas = SVC(kernel='poly')
svmClas.fit(X_train_vect, y_train)
pred = svmClas.predict(X_test_vect)
svmAccuracy = accuracy_score( y_test,pred)
print('Accuracy of linear SVM classifier: {}%'.format(svmAccuracy*100))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


##########################################
## (8) Naive Bayesian Classifer
##########################################
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gnb = GaussianNB()
gnb.fit(X_train_vect, y_train)
GaussianNB(priors=None)
pred = gnb.predict(X_test_vect)
gnbAccuracy = accuracy_score( y_test,pred)
print('Accuracy of gnb classifier: {}%'.format(gnbAccuracy*100))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
