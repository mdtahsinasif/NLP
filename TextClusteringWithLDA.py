# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:03:04 2020

@author: tahsin.asif
"""

import pandas as pd
data = pd.read_csv("Nov_rss_feed-csv.csv",encoding='ISO-8859-1')
#print(data)
data_text = data[['title']]
data_text['index'] = data_text.index
documents = data_text

len(documents)
documents[:5]

#Data Preprocessing

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')


print(WordNetLemmatizer().lemmatize('went', pos='v'))

stemmer = SnowballStemmer('english')
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]
pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    textStr = str(text)
    result = []
    for token in gensim.utils.simple_preprocess(textStr):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


doc_sample = documents[documents['index'] == 4310].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

print(documents['title'])

processed_docs = documents['title'].map(preprocess)

processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)    
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[10]

bow_doc_10 = bow_corpus[10]

for i in range(len(bow_doc_10)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_10[i][0], 
                                                     dictionary[bow_doc_10[i][0]], 
                                                     bow_doc_10[i][1]))
    


from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)    

corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

import os
from sklearn.externals import joblib
path = 'TextCLustering/'

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

#lda_cluster_model = joblib.dump(lda_model,os.path.join(path, 'lda_model.pkl') )

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
#Running LDA using TF-IDF

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)    
#lda_tfidf_model = joblib.dump(os.path.join(path, 'lda_model_tfidf.pkl') )

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
#Classification of topics

processed_docs[10]    

#Performance evaluation by classifying sample document using LDA Bag of Words model

for index, score in sorted(lda_model[bow_corpus[10]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))
    
#Performance evaluation by classifying sample document using LDA TF-IDF model
    
processed_docs[10]    
    

for index, score in sorted(lda_model_tfidf[bow_corpus[10]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10))) 
    
#Testing model on unseen document

unseen_document = 'How a Pentagon deal became an identity a attack malware Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))  
    
    
    
    
