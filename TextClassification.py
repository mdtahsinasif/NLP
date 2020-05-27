# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:25:05 2020

@author: tahsin.asif
"""

import pandas as pd
from sklearn import svm
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


news_df = pd.read_excel("C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/JavaCategorisation/PythonModel/CywareClassificationData.xlsx",encoding='ISO-8859-1')
news_df.CATEGORY.unique()


news_df.dropna(subset=["TITLE"], inplace=True)
# Preprocess data

#Transform categories into discrete numerical values;
#Transform all words to lowercase;
#Remove all punctuations



news_df['CATEGORY'] = news_df.CATEGORY.map({ 'Cyber Education and Awareness': 1,
       'Cyber Hacks and Incidents': 2, 
       'Cyber Innovation and Technology': 3,
       'Cyber Law and Regulation': 4,
       'Cyber Policy and Process': 5,
       'Emerging Threats and Cyberattacks': 6,
       'Major Release and Events': 7,
       'Threat Actors and Tools': 8,
       'Vulnerabilities and Exploits': 9,
       })

#news_df['TITLE'] = news_df.TITLE.astype(str)

news_df['TITLE'] = news_df.TITLE.astype(str).map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)

#news_df['TITLE']

#Split into train and test data sets



X_train, X_test, y_train, y_test = train_test_split(
   (news_df['TITLE']), 
    news_df['CATEGORY'], 
    random_state = 0
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

#Extract features
#Apply bag of words processing to the dataset



count_vector = CountVectorizer(stop_words = 'english')

training_data = count_vector.fit_transform(X_train)


testing_data = count_vector.transform(X_test)


#Train Multinomial Naive Bayes classifier


naive_bayes = MultinomialNB()
#y_train.fillna(y_train.mean())
#training_data.fillna(training_data.mean(),inplace=True)
#np.isnan(training_data.values.any())
#print(training_data.isnull().values.any())  
print(training_data)
#training_data = training_data.astype(float)

#y = y.as_matrix().astype(np.float)
training_data.shape
np.isnan(y_train).any()
y_train= np.nan_to_num(y_train)
training_data.data = np.nan_to_num(training_data.data)
naive_bayes = MultinomialNB()
naive_bayes_text_classifier= naive_bayes.fit((training_data), y_train)
##Generate Prediction
predictions = naive_bayes_text_classifier.predict(testing_data)
predictions
#
#svc = svm.SVC(kernel='linear', C=5,gamma='auto').fit(training_data, y_train)

##svc = svm.SVC(kernel='linear', C=5,gamma='auto').fit(training_data, y_train)
##Generate Prediction
#predictions = svc.predict(testing_data)
#predictions


#Ensemble hard voting 
#create estimators for voting classifier
#svm_estimator = svm.SVC(kernel='linear', C=5,gamma='auto')
#naive_bayes_estimator = MultinomialNB()
#
#voting_estimator = ensemble.VotingClassifier(estimators=[('sv', svm_estimator), ('nb', naive_bayes_estimator)], voting='soft', weights=[1,2,2])
##voting_grid = {'dt__max_depth':[3,5,7], 'rf__n_estimators':[50], 'rf__max_features':[5,6], 'rf__max_depth':[5]}
#grid_voting_estimator = model_selection.GridSearchCV(voting_estimator,param_grid={'C': [1, 10]}, cv=10,n_jobs=1)
#grid_voting_estimator.fit(training_data, y_train)
#print(grid_voting_estimator.grid_scores_)
#print(grid_voting_estimator.best_score_)
#print(grid_voting_estimator.best_params_)
#print(grid_voting_estimator.score(X_train, y_train))
###################

path = 'C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/JavaCategorisation/PythonModel'
#copy the model to pkl file and keep the model file at required server location
#joblib.dump(svc,os.path.join(path, 'svm_text_classifierV2.pkl') )
joblib.dump(naive_bayes_text_classifier,os.path.join(path, 'naive_bayes_text_classifierV3.pkl') )

#cross check the dumped model with load
#svm_text_classifier_loaded = joblib.load(os.path.join(path, 'svm_text_classifierV2.pkl') )
naive_bayes_text_classifier_loaded = joblib.load(os.path.join(path, 'naive_bayes_text_classifierV3.pkl') )
text = "Event : FS-ISAC 2016 Fall Summit"


count_vector_test_data = count_vector.transform([text])

predictedResult = naive_bayes_text_classifier_loaded.predict(count_vector_test_data)
print('Predicted Result:---->',predictedResult)


# Evaluate model performance

#This is a multi-class classification. So, for these evaulation scores, explicitly specify average = weighted

#np.isnan(y_test).any()
#y_test= np.nan_to_num(y_test)
#print("Accuracy score: ", accuracy_score(y_test, predictions))
#print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
#print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
#print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))
#    
############################
#
#gbm_estimator = ensemble.GradientBoostingClassifier(random_state=2017)
#gbm_grid = {'n_estimators':[50, 100], 'max_depth':[3,4,5], 'learning_rate':[0.001,0.01,0.2,0.3]}
#grid_gbm_estimator = model_selection.GridSearchCV(gbm_estimator, gbm_grid, cv=10,n_jobs=1)
#grid_gbm_estimator.fit(training_data, y_train)
#joblib.dump(grid_gbm_estimator,os.path.join(path, 'grid_gbm_estimator_V1.pkl') )
grid_gbm_estimator_V1_classifier_loaded = joblib.load(os.path.join(path, 'grid_gbm_estimator_V1.pkl') )

#print(grid_gbm_estimator.grid_scores_)
#print(grid_gbm_estimator.best_score_)
#print(grid_gbm_estimator.best_params_)
print(grid_gbm_estimator_V1_classifier_loaded.score(training_data, y_train))
############################
#
#
##cross check the dumped model with load
##grid_gbm_estimator_V1_loaded = joblib.load(os.path.join(path, 'grid_gbm_estimator_V1.pkl') )
#naive_bayes_text_classifier_loaded = joblib.load(os.path.join(path, 'naive_bayes_text_classifierV2.pkl') )
#text = "Cyber Education and Awareness"
#
#
#count_vector_test_data = count_vector.transform([text])
#
#predictedResult = naive_bayes_text_classifier_loaded.predict(count_vector_test_data)
#print('Predicted Result:---->',predictedResult)
#predictions = naive_bayes_text_classifier_loaded.predict(testing_data)
#predictions
#print("Accuracy score: ", accuracy_score(y_test, predictions))
##importances = grid_rf_estimator.best_estimator_.feature_importances_
