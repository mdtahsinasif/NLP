from sklearn.externals import joblib
import pandas as pd
from flask import Flask, jsonify, request


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np


import os
app = Flask(__name__)

result = []
@app.route('/cluster', methods=['POST'])
def cluster():
    #print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_)
    #inputData = json_
    url_test = pd.DataFrame([json_])
    data_text = url_test[['data']]
    data_text['index'] = data_text.index
    documents = data_text
    processed_docs = documents['data'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)

    bow_vector = dictionary.doc2bow(preprocess(url_test['data']))
    dictoutput = {}
    for index, score in sorted(log_estimator[bow_vector], key=lambda tup: -1*tup[1]):
      #  print("Score: {}\t Topic: {}".format(score, log_estimator.print_topic(index, 5)))
        list_key = score
        list_val = log_estimator.print_topic(index, 5)
        dictoutput[list_key] = list_val
        #dictoutput={"Score: {}\t Topic: {}".format(score, log_estimator.print_topic(index, 5))}
    print("dictoutput--------------->",dictoutput)    
    return jsonify(pd.Series(dictoutput).to_json(orient='keys','values'))
   # output = log_estimator.predict(count_vect.transform(url_test['data']))
   

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


   

#MODEL_FILE = 'lda_model_tfidf.pkl'
#log_estimator = joblib.load(MODEL_FILE)
if __name__ == '__main__':
    path = '......../TextCLustering/'
    log_estimator = joblib.load(os.path.join(path, 'lda_model_tfidf.pkl') )
#    log_estimator = joblib.load(MODEL_FILE)
    app.run(port=8086)
