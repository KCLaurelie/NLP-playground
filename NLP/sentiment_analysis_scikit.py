# from https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)

######################################################
# DATA ANALYSIS
######################################################
airline_tweets.head()
airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')

# distribution of sentiments (positive, negative, neutral) across tweets
airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')

#airline sentiment confidence
import seaborn as sns
sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)

######################################################
# DATA CLEANING
######################################################
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

######################################################
# REPRESENT TEXT IN NUMERIC FORM
######################################################

# bag of word approach
Doc1 = "I like to play football"
Doc2 = "It is a good game"
Doc3 = "I prefer football over rugby"
Vocab = ["I", "like", "to", "play", "football", "it", "is", "a", "good", "game", "prefer", "over", "rugby"]

# TFIDF approach
# TF  = (Frequency of a word in the document)/(Total words in the document)
# IDF = Log((Total number of docs)/(Number of docs containing the word))

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

######################################################
# DIVIDE TRAINING / TEST SETS
######################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

######################################################
# RANDOM FOREST CLASSIFIER
######################################################
from sklearn.ensemble import RandomForestClassifier

# train model
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# make predictions and evaluate model
predictions = text_classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

######################################################
# LOGISTIC REGRESSION
######################################################

######################################################
# SVM
######################################################

######################################################
# KNN
######################################################