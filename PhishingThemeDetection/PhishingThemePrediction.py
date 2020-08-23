# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:26:36 2020

@author: TahsinAsif
"""

# -*- coding: utf-8 -*-
import logging
import time

import schedule

"""
Created on Thu Aug 19 20:04:07 2020

@author: TahsinAsif
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import joblib
from nltk.corpus import stopwords
from pymongo import MongoClient
from datetime import datetime, timedelta

def phishingTemplatePrediction():
    news_df = pd.read_csv("C:/Users/TahsinAsif/PhishingThemeDetection.csv",encoding='ISO-8859-1')
    news_df.dropna(subset=["link"], inplace=True)
    last_hour_date_time = datetime.now() - timedelta(hours=24 * 365)
    query = "{pub_date:ISODate('+last_hour_date_time+')}"
    print("Before MyDoc")
    mydoc = urlCollection.find()
    #mydoc = urlCollection.find({'phishingTemplateInfo': {'$exists': False}})
    print("After MyDoc")
    # mydoc =   urlCollection.find({
    #             'pub_date': {'$lte': datetime.now() - timedelta(hours=24)}
    #         })
    # Preprocess data
    news_df['title_des_cat'] = news_df['title'].astype(str).str.lower()+  news_df['description'].astype(str).str.lower() +news_df['newsCategory'].astype(str).str.lower()
    #Transform categories into discrete numerical values;
    #Transform all words to lowercase;
    #Remove all punctuations
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    news_df['title_des_cat']  = [w for w in news_df['title_des_cat'] if not w in stop_words]

    X_train, X_test, y_train, y_test = train_test_split(
       (news_df['title_des_cat']),
        news_df['PhishingCategory'],
        random_state = 0
    )
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    training_data = tfidf_transformer.fit_transform(X_train_counts)
    try:
        count = 0
        for x in mydoc:
            #print("x----------",x)
            dw ={}
            count=count+1
            print(count)
            sentences = []
           # print(x['title'])
            #sentences.append(x['title'] + x['newsCategory']+x['summary'])
            try:
               # print(x)
                if('newsCategory' not in x):
                    print("newsCategory Not  present")
                    sentences.append(x['title'] + x['description'])
                elif('description' not in x):
                    sentences.append(x['title'] + x['newsCategory'])
                elif('title' not in x):
                    print("newsCategory  not present")
                    sentences.append(x['title'] + x['description'])
                else:
                    sentences.append(x['title'] + x['description']+ x['newsCategory'])

                text = [w for w in sentences if not w in stop_words]
                print(text)
                testing_data = count_vect.transform(text)
                tfidf_transformer = TfidfTransformer()
                training_data = tfidf_transformer.fit_transform(testing_data)
                #predictions = final_estimator_loaded.predict_proba(training_data)
                predictions = final_estimator_loaded.predict(training_data)
                makeitastring = ''.join(map(str, predictions))
               # x['phishingTemplateInfo'] = makeitastring
                query = {"title": x['title']}
                print('query----------',query)
                new_values = {"$set": {"title": x['title'],"phishingTemplateInfo": makeitastring}}
                urlCollection.update_one(query,new_values)
                print("db updated")
               # print(dw)
            except Exception as e:
                print(e)
          #  print(str.sentences)
           # sentences = text

        connection.close()
    except Exception as e:
        logging.error("Exception Occured",exc_info=True)




if __name__ == '__main__':
    MODEL_FILE = "ada_grid_estimator_model.pkl"
    final_estimator_loaded = joblib.load(MODEL_FILE)
    connection = MongoClient('localhost', 27017)
    db = connection.core
    # url input collection
    urlCollection = db.rss_feed_entry_phishing
    dw = {}
    phishingTemplatePrediction()

# MODEL_FILE = "ada_grid_estimator_model.pkl"
# final_estimator_loaded = joblib.load(MODEL_FILE)
# connection = MongoClient('localhost', 27017)
# db = connection.core
# # url input collection
# urlCollection = db.rss_feed_entry_phishing
#
# def startMethod():
#     phishingTemplatePrediction()
#
# schedule.every(1).minutes.do(startMethod)
# # Every day at 12am or 00:00 time bedtime() is called.
# #schedule.every().day.at("00:00").do(startMethod)
# # # Loop so that the scheduling task
# # # keeps on running all time.
# while True:
#     # Checks whether a scheduled task
#     # is pending to run or not
#     schedule.run_pending()
#     time.sleep(1)
