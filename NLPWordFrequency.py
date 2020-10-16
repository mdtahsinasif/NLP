# Importing libraries
import nltk
import re

import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from pymongo import MongoClient

# Input the file


dbtext = ''

#val = input("Enter word combination  Value:")
#val = int(val)
val =1

host = 'localhost'
port = 27017

print('We are in dbConnection.py')
try:
    connection = MongoClient('localhost', 27017)
    db = connection.core
    collection = db.rss_feed_entry
    for x in collection.find({}, {"_id": 0, "title": 1,"newsCategory":1}):
        if(x['newsCategory']=="Emerging Threats and Cyberattacks" or x['newsCategory']== "Cyber Hacks and Incident"
        or x['newsCategory']=="Threat Actors and Tools"):
            print(x['newsCategory'])
            dbtext = (dbtext + (unidecode.unidecode_expect_nonascii(x['title'])))
    dbtext1 = sent_tokenize(dbtext)
   # dbtext1 = list(dbtext.split(" "))
   # print('dbtext->', dbtext1)
    print("Connected successfully!!!")
except Exception as e:
    print(e)

# Preprocessing
def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)
    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()

    # Stopword removal

stop_words = set(stopwords.words('english'))
your_list = ['skills', 'ability', 'The', 'description','your','in','from','From','Their','euleros']
for i, line in enumerate(dbtext1):
    dbtext1[i] = ' '.join([x for
                        x in nltk.word_tokenize(line) if
                        (x not in stop_words) and (x not in your_list)])

print('dbtext->', dbtext1)
# Getting trigrams
vectorizer = CountVectorizer(ngram_range=(val, val))
X1 = vectorizer.fit_transform(dbtext1)
features = (vectorizer.get_feature_names())
#print("\n\nFeatures : \n", features)
#print("\n\nX1 : \n", X1.toarray())

# Applying TFIDF
vectorizer = TfidfVectorizer(ngram_range=(val, val))
X2 = vectorizer.fit_transform(dbtext1)
scores = (X2.toarray())
#print("\n\nScores : \n", scores)

# Getting top ranking features
sums = X2.sum(axis=0)
data1 = []
topKeywords=[]
for col, term in enumerate(features):
    data1.append((term, sums[0, col]))
ranking = pd.DataFrame(data1, columns=['term', 'rank'])
words = (ranking.sort_values('rank', ascending=False))
#print("\n\nWords head : \n", words['term'].head(10))
topKeywords.append(words['term'].head(30))
for topword in topKeywords:
    print(topword.values)

for x in collection.find({}, {"_id": 0, "title": 1,"link":1,"newsCategory":1}):
    for topword in topKeywords:
        titleVal = str(topword.values)
      #  print("titleVal::::::",titleVal)
        if(titleVal in  x['title'] and x['newsCategory']=="Emerging Threats and Cyberattacks" or x['newsCategory']== "Cyber Hacks and Incident"
        or x['newsCategory']=="Threat Actors and Tools" ) :
            print(x['link'])
