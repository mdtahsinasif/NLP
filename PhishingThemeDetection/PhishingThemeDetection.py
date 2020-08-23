# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:04:07 2020

@author: TahsinAsif
"""
import pandas as pd
from sklearn import svm,linear_model,naive_bayes, model_selection
import numpy as np
from sklearn import ensemble,preprocessing
from sklearn import tree, model_selection
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree, model_selection, preprocessing, ensemble, feature_selection, neighbors, naive_bayes
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


news_df = pd.read_csv("C:/Users/TahsinAsif/PhishingThemeDetection.csv",encoding='ISO-8859-1')

news_df.dropna(subset=["link"], inplace=True)
# Preprocess data

#Transform categories into discrete numerical values;
#Transform all words to lowercase;
#Remove all punctuations


news_df['newsCategory'] = news_df.newsCategory.map({ 'Cyber Education and Awareness': 1,
       'Cyber Hacks and Incidents': 2, 
       'Cyber Innovation and Technology': 3,
       'Cyber Law and Regulation': 4,
       'Cyber Policy and Process': 5,
       'Emerging Threats and Cyberattacks': 6,
       'Major Release and Events': 7,
       'Threat Actors and Tools': 8,
       'Vulnerabilities and Exploits': 9,
       'Cyber Insights':10,
       'Cyber Strategy, Policy and Process':11,
       })

print(news_df)


features = ['newsCategory']
tmp = news_df[features].values

news_df['newsCategory'] = news_df.newsCategory.astype(int)

#
#news_df['TITLE'] = news_df.TITLE.astype(str).map(
#    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
#)

#news_df['TITLE']

#Split into train and test data sets
ohe_features = ['newsCategory']
ohe = preprocessing.OneHotEncoder()
ohe.fit(news_df[ohe_features])
print(ohe.categories_)
tmp1 = ohe.transform(news_df[ohe_features]).toarray()
print(tmp1)

#import matplotlib.pyplot as plt
#plt.scatter(news_df['description'],news_df['PhishingCategory'])

news_df['title_des'] = news_df['title'].astype(str) +  news_df['description'].astype(str)
news_df['title_des_cat'] = news_df['title'].astype(str).str.lower()+  news_df['description'].astype(str).str.lower() +news_df['newsCategory'].astype(str).str.lower()

print(news_df['title_des_cat'])

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
news_df['title_des_cat']  = [w for w in news_df['title_des_cat'] if not w in stop_words]
print(news_df['title_des_cat'] )

X_train, X_test, y_train, y_test = train_test_split(
   (news_df['title_des_cat']), 
    news_df['PhishingCategory'], 
    random_state = 0
)
print(news_df['PhishingCategory'])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
training_data = tfidf_transformer.fit_transform(X_train_counts)

print(training_data)
print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

#Extract features
#Apply bag of words processing to the dataset

testing_data = count_vect.transform(X_test)
print(testing_data)
#training_data = training_data.astype(float)

#y = y.as_matrix().astype(np.float)
X_train_counts.shape
X_train.shape
y_train.shape
y_train= np.nan_to_num(y_train)
np.isnan(y_train).any()
training_data.data = np.nan_to_num(training_data.data)
##################################


#ada boost
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),random_state=100)
ada_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'base_estimator__max_depth':[1,3,5], 'base_estimator__criterion':['entropy', 'gini'] }
ada_grid_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, scoring='accuracy', cv=10, return_train_score=True)
ada_grid_estimator.fit(training_data, y_train)

print(ada_grid_estimator.best_score_)
print(ada_grid_estimator.best_params_)
final_estimator = ada_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.estimator_weights_)
print(final_estimator.estimator_errors_)
print(final_estimator.score(training_data, y_train))
#score: 0.9035262807717898
print(final_estimator.score(testing_data, y_test))
#score:0.7584830339321357
path = 'C:\\Users\\TahsinAsif\\OneDrive - CYFIRMA INDIA PRIVATE LIMITED\\21112019\\AI\\JavaCategorisation\\PythonModel'
#copy the model to pkl file and keep the model file at required server location
joblib.dump(final_estimator,os.path.join(path, 'ada_grid_estimator_model.pkl') )





##################################
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':list(range(3,10,1)), 'criterion':['entropy', 'gini'], 'max_features':[5,10,20,30], 'min_samples_split':[2,5,10] }
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='accuracy', cv=10, refit=True, return_train_score=True)
dt_grid_estimator.fit(training_data, y_train)
result = dt_grid_estimator.cv_results_

#######
#multiNBClassifier = MultinomialNB().fit(X_train_tfidf, y_train)  

naive_bayes = MultinomialNB()
naive_bayes_text_classifier= naive_bayes.fit((training_data), y_train)
print(naive_bayes_text_classifier.score(training_data, y_train))
#0.7658017298735862
print(naive_bayes_text_classifier.score(testing_data, y_test))
#0.7325349301397206
##Generate Prediction
predictions = naive_bayes_text_classifier.predict(testing_data)
predictions

#########Ensemble Random Forest###############
rf_estimator = ensemble.RandomForestClassifier(random_state=100)
rf_estimator_clf = rf_estimator.fit(training_data, y_train)  
print(result)
print(result.get('params'))
print(result.get('mean_train_score'))
print(result.get('mean_test_score'))
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_params_)
final_estimator = dt_grid_estimator.best_estimator_
print(final_estimator.score(training_data, y_train))
print(final_estimator.score(testing_data, y_test))
print(final_estimator.feature_importances_)

######################Ensemble bagging########################


#bagged tree
bt_estimator = ensemble.BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),random_state=100)
bt_grid = {'n_estimators':list(range(50,201,50)), 'base_estimator__max_depth':[3,5,9], 'base_estimator__criterion':['entropy', 'gini'] }
bt_grid_estimator = model_selection.GridSearchCV(bt_estimator, bt_grid, scoring='accuracy', cv=10, return_train_score=True)
bt_grid_estimator.fit(training_data, y_train)
final_estimator = bt_grid_estimator.best_estimator_
print(final_estimator.score(training_data, y_train))
print(final_estimator.score(testing_data, y_test))
##########################################

#Random forest
rf_estimator = ensemble.RandomForestClassifier(random_state=100, n_jobs=4)
rf_grid = {'n_estimators':list(range(50,201,50)), 'max_depth':[3,5,9], 'max_features':[4,5,6], 'criterion':['entropy', 'gini'] }
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='accuracy', cv=10, return_train_score=True)
rf_grid_estimator.fit(training_data, y_train)
final_estimator = bt_grid_estimator.best_estimator_
print(final_estimator.score(training_data, y_train))
print(final_estimator.score(testing_data, y_test))

##############################################

#extremely randomized trees
et_estimator = ensemble.ExtraTreesClassifier(random_state=100, n_jobs=4)
et_grid = {'n_estimators':list(range(50,201,50)), 'max_depth':[3,5,9], 'max_features':[4,5,6], 'criterion':['entropy', 'gini'] }
et_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='accuracy', cv=10, return_train_score=True)
et_grid_estimator.fit(training_data, y_train)

print(et_grid_estimator.best_score_)
print(et_grid_estimator.best_params_)
final_estimator = et_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(training_data, y_train))
#Score :: 0.7491683300066534

print(final_estimator.score(testing_data, y_test))
#Score :: 0.7245508982035929

######################################




#gradient boosting
gb_estimator = ensemble.GradientBoostingClassifier(random_state=100)
gb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5]}
gb_grid_estimator = model_selection.GridSearchCV(gb_estimator, gb_grid, scoring='accuracy', cv=10, return_train_score=True)
gb_grid_estimator.fit(training_data, y_train)

print(gb_grid_estimator.best_score_)
print(gb_grid_estimator.best_params_)
final_estimator = gb_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(training_data, y_train))
#score: 0.8908848968729208
print(final_estimator.score(testing_data, y_test))
#score: 0.7285429141716567

import xgboost as xgb
#pip install xgboost
#xgb = boosting + regulaization for overfit control
xgb_estimator = xgb.XGBClassifier(random_state=100, n_jobs=-1)
xgb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5], 'gamma':[0,0.01,0.1,0.2], 'reg_alpha':[0,0.5,1], 'reg_lambda':[0,0.5,1]}
xgb_grid_estimator = model_selection.GridSearchCV(xgb_estimator, xgb_grid, scoring='accuracy', cv=10, return_train_score=True)
xgb_grid_estimator.fit(training_data, y_train)

print(xgb_grid_estimator.best_score_)
print(xgb_grid_estimator.best_params_)
final_estimator = xgb_grid_estimator.best_estimator_
print(final_estimator.score(training_data, y_train))
#score: 0.9108449767132402
print(final_estimator.score(testing_data, y_test))
#score: 0.7025948103792415
############################################################
#Objective base learning

#logistic regression
lr_estimator = linear_model.LogisticRegression(random_state=100)
lr_grid = {'penalty':['l1','l2'], 'C':[0.1,0.2,0.5,1,2], 'max_iter':list(range(100,1000,500))}
grid_lr_estimator = model_selection.GridSearchCV(lr_estimator, lr_grid, cv=10)
grid_lr_estimator.fit(training_data, y_train)
print(grid_lr_estimator.best_params_)
final_estimator = grid_lr_estimator.best_estimator_
print(final_estimator.coef_)
print(final_estimator.intercept_)
print(grid_lr_estimator.best_score_)
print(final_estimator.score(training_data, y_train))
#score:8948769128409847
print(final_estimator.score(testing_data, y_test))
#score: 0.6047904191616766


#linear svm
lsvm_estimator = svm.LinearSVC(random_state=100)
lsvm_grid = {'C':[0.1,0.2,0.5,1] }
grid_lsvm_estimator = model_selection.GridSearchCV(lsvm_estimator, lsvm_grid, cv=10)
grid_lsvm_estimator.fit(training_data, y_train)
print(grid_lsvm_estimator.best_params_)
final_estimator = grid_lsvm_estimator.best_estimator_
print(final_estimator.coef_)
print(final_estimator.intercept_)
print(grid_lsvm_estimator.best_score_)
print(final_estimator.score(training_data, y_train))
#0.9634065202927479
print(final_estimator.score(testing_data, y_test))
#0.6087824351297405


#perceptron algorithm
perceptron_estimator = linear_model.Perceptron(random_state=100)
perceptron_grid = {'penalty':['l1','l2','elasticnet'], 'alpha':[0.001,0.002,0.005] }
grid_perceptron_estimator = model_selection.GridSearchCV(perceptron_estimator, perceptron_grid, cv=10)
grid_perceptron_estimator.fit(training_data, y_train)
print(grid_perceptron_estimator.best_params_)
final_estimator = grid_perceptron_estimator.best_estimator_
print(final_estimator.coef_)
print(final_estimator.intercept_)
print(grid_perceptron_estimator.best_score_)
print(final_estimator.score(training_data, y_train))
#0.7578176979374585
print(final_estimator.score(testing_data, y_test))
#0.6866267465069861
################
#kernel svm - poly kernel
ksvm_estimator = svm.SVC(random_state=100)
ksvm_grid = {'C':[0.1,0.2,0.5,1], 'kernel':['poly'], 'degree':[3,4,5] }
grid_ksvm_estimator = model_selection.GridSearchCV(ksvm_estimator, ksvm_grid, cv=10)
grid_ksvm_estimator.fit(training_data, y_train)
print(grid_ksvm_estimator.best_params_)
final_estimator = grid_ksvm_estimator.best_estimator_
print(final_estimator.dual_coef_)
print(final_estimator.intercept_)
print(final_estimator.n_support_)
print(final_estimator.support_)
print(grid_ksvm_estimator.best_score_)
print(final_estimator.score(training_data, y_train))
#0.9813705921490352
print(final_estimator.score(testing_data, y_test))
#6866267465069861

#kernel svm - rbf kernel
ksvm_estimator = svm.SVC(random_state=100)
ksvm_grid = {'C':[0.1,0.2,0.5,1], 'kernel':['rbf'] }
grid_ksvm_estimator = model_selection.GridSearchCV(ksvm_estimator, ksvm_grid, cv=10)
grid_ksvm_estimator.fit(training_data, y_train)
print(grid_ksvm_estimator.best_params_)
final_estimator = grid_ksvm_estimator.best_estimator_
print(final_estimator.dual_coef_)
print(final_estimator.intercept_)
print(final_estimator.n_support_)
print(final_estimator.support_)
print(grid_ksvm_estimator.best_score_)
print(final_estimator.score(training_data, y_train))
#0.9587491683300067
print(final_estimator.score(testing_data, y_test))
#7245508982035929

nb_estimator = naive_bayes.GaussianNB()
nb_estimator.fit(training_data.toarray(), y_train)

print(nb_estimator.class_prior_)
print(nb_estimator.sigma_)
print(nb_estimator.theta_)

res = model_selection.cross_validate(nb_estimator, training_data.toarray(), y_train, cv=10)
print(res.get('test_score').mean())
print(nb_estimator.score(training_data.toarray(), y_train))
#0.9687292082501663
print(nb_estimator.score(testing_data.toarray(), y_test))
#0.7485029940119761


############Deep learning model###################\

from tensorflow import keras
layers = keras.layers
models = keras.models
# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 2
drop_ratio = 0.5
max_words = 9864
num_classes = np.max(y_train) + 1

# Build the model
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(drop_ratio))
#model.add(layers.Dense(num_classes))
#model.add(layers.Activation('softmax'))
model.add(layers.Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(training_data, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# Evaluate the accuracy of our trained model
score = model.evaluate(testing_data, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


X_test = tmp[titanic_train.shape[0]:]
titanic_test['Survived'] = final_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)

#Ensemble hard voting 
#create estimators for voting classifier
#svm_estimator = svm.SVC(C=1, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
svm_estimator = svm.SVC(decision_function_shape='ovo')

naive_bayes_estimator = MultinomialNB()
#create estimators for voting classifier
voting_estimator = ensemble.VotingClassifier(estimators=[('sv', svm_estimator), ('nb', naive_bayes_estimator)], voting='soft', weights=[1,2,2])


voting_grid = {'dt__max_depth':[3,5,7], 'rf__n_estimators':[50], 'rf__max_features':[5,6], 'rf__max_depth':[5]}
grid_voting_estimator = model_selection.GridSearchCV(voting_estimator,param_grid={'C': [1, 10, 100, 1000]}, cv=10,n_jobs=1)
grid_voting_estimator.fit(training_data, y_train)
#model.fit(X, y, nb_epoch=40, batch_size=32, validation_split=0.2, verbose=1)
print(grid_voting_estimator.grid_scores_)
print(grid_voting_estimator.best_score_)
print(grid_voting_estimator.best_params_)
print(grid_voting_estimator.score(X_train, y_train))
##Generate Prediction
#predictions = naive_bayes_text_classifier.predict(testing_data)
#predictions
###################

path = 'C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/JavaCategorisation/PythonModel'
#copy the model to pkl file and keep the model file at required server location
#joblib.dump(svc,os.path.join(path, 'svm_text_classifierV2.pkl') )
#joblib.dump(naive_bayes_text_classifier,os.path.join(path, 'naive_bayes_text_classifierV3.pkl') )

#cross check the dumped model with load
#svm_text_classifier_loaded = joblib.load(os.path.join(path, 'svm_text_classifierV2.pkl') )
grid_voting_estimator = joblib.load(os.path.join(path,'grid_gbm_estimator_V1.pkl') )
print(grid_voting_estimator.grid_scores_)
print(grid_voting_estimator.best_score_)
print(grid_voting_estimator.best_params_)
print(grid_voting_estimator.score(X_train, y_train))
predictions = grid_voting_estimator.predict(testing_data)
predictions
#print("Accuracy score: ", accuracy_score(y_test, predictions))
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

