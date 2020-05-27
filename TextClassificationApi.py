# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:37:26 2020

@author: tahsin.asif
"""

from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, jsonify, request
import string
app = Flask(__name__)


@app.route('/newsClassification', methods=['POST'])
def predict():
    #print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_)
    #inputData = json_
    url_test = pd.DataFrame([json_])
    print('url_test item data---->',url_test['data'])
    df = pd.read_excel("............/CywareClassificationData.xlsx",encoding='ISO-8859-1')
  #  df.dropna(subset=["TITLE"], inplace=True)
    df['TITLE'] = df.TITLE.astype(str).map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))

    X_train, X_test, y_train, y_test = train_test_split(
           (df['TITLE']), 
            df['CATEGORY'], 
            random_state = 0
        )
    df['CATEGORY'] = df.CATEGORY.map({ 'Cyber Education and Awareness': 1,
       'Cyber Hacks and Incidents': 2, 
       'Cyber Innovation and Technology': 3,
       'Cyber Law and Regulation': 4,
       'Cyber Policy and Process': 5,
       'Emerging Threats and Cyberattacks': 6,
       'Major Release and Events': 7,
       'Threat Actors and Tools': 8,
       'Vulnerabilities and Exploits': 9,
       })
        
    print("Training dataset: ", X_train.shape[0])
    print("Test dataset: ", X_test.shape[0])
    count_vector = CountVectorizer(stop_words = 'english')
    count_vector.fit_transform(X_train)
    input = url_test['data']
    print('input----------->',input)
    output = log_estimator.predict(count_vector.transform(input))
    print('output----------->',output) 
    return jsonify(pd.Series(output).to_json(orient='values'))

MODEL_FILE = '----------/naive_bayes_text_classifierV3.pkl'
log_estimator = joblib.load(MODEL_FILE)
if __name__ == '__main__':
    log_estimator = joblib.load(MODEL_FILE)
    #app.run(debug=True)
    app.run(port=8086)
