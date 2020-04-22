# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:26:39 2020

@author: tahsin.asif
"""

import collections
from sklearn.externals import joblib
import gensim
import pandas as pd
import time
import os
import schedule
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas import np
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

        # plot the cost against K values
    # plt.plot(range(1, 11), cost, color='g', linewidth='3')
    # plt.xlabel("Value of K")
    # plt.ylabel("Sqaured Error (Cost)")
    # plt.show()  # clear the plot

    # the point of the elbow is the
    # most optimal value for choosing k
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)


def clusterOperation():

    sentences = []
    try:
        connection = MongoClient('localhost', 27017)
        db = connection.core
        collection = db.rss_feed_entry
        mydoc = collection.find().limit(50)
        print("Connected successfully!!!")
        for x in mydoc:
            sentences.append(x['title'] + x['newsCategory'])
     #   print(sentences)
        connection.close()
    except:
        print("Could not connect to MongoDB")

    # print(sentences)

    nclusters = 6
    clusters = cluster_sentences(sentences, nclusters)
    clusterdict = {}
    for cluster in range(nclusters):
        output = []
        # print ("cluster ",cluster,":")
        clusterNumber = "Cluster::" + str(cluster)
        key = (clusterNumber)
        for i, sentence in enumerate(clusters[cluster]):
            #   print ("\tsentence ",i,": ",sentences[sentence])
            #  output.append(("sentence:" + str(i) + ": ", sentences[sentence]))
            output.append((sentences[sentence]))

            value = (output)
        #print(value)
        clusterdict[key] = value

    try:
        connection = MongoClient('localhost', 27017)
        db = connection.core
        collection = db.ai_news_cluster
        collection.remove()
        for key, value in clusterdict.items():
            myquery = {"_id": key}
            newvalues = {"$set": {"titles": value}}
            collection.insert(myquery, newvalues)
            collection.update_many(myquery, newvalues)
          #  collection.remove(myquery, newvalues)
    except:
        print("Could not connect to MongoDB")


def preprocess(text):
    # print('Inside Preprocess')
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

        connection = MongoClient('localhost', 27017)
        db = connection.core
        collection = db.ai_news_cluster
        mydoc = collection.find({'titles': {'$exists': True}})
        #   mydoc = collection.find().limit(50)
        # db.inventory.find(
       # print((mydoc))
        print("Connected successfully!!!")
        #   print(mydoc)
        dictoutput = {}

        for x in mydoc:
            scoringDict_key = (x['_id'])
            titleWithScoreValArray = []

            for l in x['titles']:

                data_text = pd.DataFrame([l])
                #   print('data_text=>',data_text)
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
                    # #     #  print("Score: {}\t Topic: {}".format(score, log_estimator.print_topic(index, 5)))
                    list_key = score
                    score_list.append(score)
                    list_val = log_estimator.print_topic(index, 3)
                    dictoutput[list_key] = list_val

                # dictoutput={"Score: {}\t Topic: {}".format(score, log_estimator.print_topic(index, 5))}

                titleWithScoreVal = (l + '--->' + str(max(score_list)))

                titleWithScoreValArray.append(titleWithScoreVal)
            scoringDict_value = titleWithScoreValArray
            scoringDict[scoringDict_key] = scoringDict_value

        update()


    except:
        print("Could not connect to MongoDB")


def update():
    try:
        connection = MongoClient('localhost', 27017)
        db = connection.core
        collection = db.ai_news_cluster
        #  mydoc = collection.find({'titles': {'$exists': True}}).limit(6)
        mydoc = collection.find()
     #   print('Inside update method')

        for key in mydoc:
            #  print(key['_id'])
            for key, value in scoringDict.items():
                #print(key)
                myquery = {"_id": key}
                newvalues = {"$set": {"titles": value}}
                collection.update_many(myquery, newvalues)
    except:
        print('----------')


# if __name__ == '__main__':
def startMethod():
    clusterOperation()
    score_dbConnection()

path = 'C:................/AI/TextCLustering/'
log_estimator = joblib.load(os.path.join(path, 'lda_model_tfidf.pkl'))
scoringDict = {}
schedule.every(1).minutes.do(startMethod)
#    #operation()
# # Loop so that the scheduling task
# # keeps on running all time.
while True:
    # Checks whether a scheduled task
    # is pending to run or not
    schedule.run_pending()
    time.sleep(1)
