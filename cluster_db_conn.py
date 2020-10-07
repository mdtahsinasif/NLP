# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:26:39 2020

@author: tahsin.asif
"""
from datetime import datetime, timedelta
import collections
import joblib
import gensim
import pandas as pd
import time
import os
#import schedule
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import matplotlib.pyplot as plt
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import logging


def word_tokenizer(text):
    # tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens

def cluster_sentences(sentences, nb_of_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                       stop_words=stopwords.words('english'),
                                       max_df=0.9,
                                       min_df=0.1,
                                       lowercase=True)
    # builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    prediction = kmeans.predict(tfidf_matrix)
    clusters = collections.defaultdict(list)
    plt.clf()  # clear the figure
    cost = []
    for i in range(1, 11):
        KM = KMeans(n_clusters=i, max_iter=500)
        KM.fit(tfidf_matrix)

        # calculates squared error
        # for the clustered points
        cost.append(KM.inertia_)

    # the point of the elbow is the
    # most optimal value for choosing k
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)


def clusterOperation():
    sentences = []
    try:
        db = connection.core
        collection = db.rss_feed_entry
        #collection = db.phishing_theme_template
        last_hour_date_time = datetime.now() - timedelta(hours=24 * 365)
        query = "{pub_date:ISODate('+last_hour_date_time+')}"
        # mydoc = collection.find(query)
        mydoc =   collection.find({
                'pub_date': {'$lte': datetime.now() - timedelta(hours=24)}
            })

       # print(mydoc)
        #print(last_hour_date_time.strftime('%Y-%m-%d %H:%M:%S'))

        print("Connected successfully!!!")
        for x in mydoc:
            #print(x['summary'])
            #sentences.append(x['title'] + x['newsCategory']+x['summary'])
           # sentences.append(x['title'] + x['newsCategory'])
            sentences.append(x['title'])
        connection.close()
    except Exception as e:
        logging.error("Exception Occured",exc_info=True)

    nclusters = 6
    #nclusters = 2
    clusters = cluster_sentences(sentences, nclusters)
    clusterdict = {}
    for cluster in range(nclusters):
        output = []
        # print ("cluster ",cluster,":")
        clusterNumber = "Cluster::" + str(cluster)
        key = (clusterNumber)
        for i, sentence in enumerate(clusters[cluster]):
            output.append((sentences[sentence]))
            value = (output)
        #print(value)
        clusterdict[key] = value

    try:
        db = connection.core
        collection = db.ai_news_cluster
     #   collection.remove()
        for key, value in clusterdict.items():
            #myquery = {"_id": key}
            myquery = {"cluster_id": key}
            newvalues = {"$set": {"titles": value}}
            collection.insert(myquery, newvalues)
            collection.update_many(myquery, newvalues)
        connection.close()
    except Exception as e:
        logging.error("Exception Occured", exc_info=True)


def preprocess(text):
    textStr = str(text)
    result = []
    for token in gensim.utils.simple_preprocess(textStr):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def score_dbConnection():
    try:
        db = connection.core
        collection = db.ai_news_cluster
        mydoc = collection.find({'titles': {'$exists': True}})
        dictoutput = {}
        for x in mydoc:
            #scoringDict_key = (x['_id'])
            scoringDict_key = (x['cluster_id'])
            titleWithScoreValArray = []
            for l in x['titles']:
                data_text = pd.DataFrame([l])
                data_text['index'] = data_text.index
                # print('documents=======>',data_text['index'])
                document = data_text
                #   print('document=======>',document)
                processed_docs = document[0].map(preprocess)
                #  print('processed_docs=========>',processed_docs)
                dictionary = gensim.corpora.Dictionary(processed_docs)
                # print(l)
                bow_vector = dictionary.doc2bow(preprocess(document))
                score_list = []

                for index, score in sorted(log_estimator[bow_vector], key=lambda tup: -1 * tup[1]):
                    list_key = score
                    score_list.append(score)
                    list_val = log_estimator.print_topic(index, 3)
                    dictoutput[list_key] = list_val
                titleWithScoreVal = (l + '--->' + str(max(score_list)))
                titleWithScoreValArray.append(titleWithScoreVal)
            scoringDict_value = titleWithScoreValArray
            scoringDict[scoringDict_key] = scoringDict_value
        update()
        print("Operation Ends")
    except Exception as e:
        logging.error("Exception Occured", exc_info=True)


def update():
    try:
        db = connection.core
        collection = db.ai_news_cluster
        mydoc = collection.find()
        for key in mydoc:
            #  print(key['_id'])
            for key, value in scoringDict.items():
                #print(key)
               # myquery = {"_id": key}
                myquery = {"cluster_id": key}
                newvalues = {"$set": {"titles": value,"Current Date": datetime.now()}}
                collection.update_many(myquery, newvalues)
    except Exception as e:
        logging.error("Exception Occured",exc_info=True)


if __name__ == '__main__':
    connection = MongoClient('localhost', 27017)
    path = 'C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/TextCLustering/'
    ModelFile = 'lda_model_tfidf.pkl'
    log_estimator = joblib.load(ModelFile)
    scoringDict = {}
    clusterOperation()
    score_dbConnection()

# def startMethod():
#      clusterOperation()
#      score_dbConnection()
#
# connection = MongoClient('localhost', 27017)
# ModelFile = 'lda_model_tfidf.pkl'
# log_estimator = joblib.load(ModelFile)
# scoringDict = {}
# schedule.every(1).minutes.do(startMethod)
#    #operation()
# # Loop so that the scheduling task
# # keeps on running all time.
# while True:
#     # Checks whether a scheduled task
#     # is pending to run or not
#     schedule.run_pending()
#     time.sleep(1)