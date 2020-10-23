import joblib
import pandas as pd
from sklearn import tree, model_selection, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import urllib.request
from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
import json
from urllib.parse import urlparse
from tld import get_tld
import nltk

nltk.download('punkt')

import os

import string
import imgkit
import re
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_)
    # inputData = json_
    url_test = pd.DataFrame([json_])
    print('url_test item data---->', url_test['data'])
    df = pd.read_csv(
        "CveScore.csv",
        encoding='ISO-8859-1')

    df.dropna(subset=["baseScore"], inplace=True)

    # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
    #                       stop_words='english')
    # features = tfidf.fit_transform(df.description).toarray()
    # labels = df.baseScore
    # features.shape
    X_train, X_test, y_train, y_test = train_test_split(df['description'], df['baseScore'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    # print("before logestimator------------------>",count_vect.transform(url_test['data']))
    # print(log_estimator.predict(count_vect.transform(url_test['data'])))
    output = log_estimator.predict(count_vect.transform(url_test['data']))
    return jsonify(pd.Series(output).to_json(orient='values'))


@app.route('/summary', methods=['POST'])
def summary():
    headers = {}
    headers[
        'User-Agent'] = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
    json_ = request.json
    print('json:', json_['data'])
    # print('json_g---->',json_g)

    url_request = urllib.request.Request(
        json_['data'], headers=headers)

    text_str2 = urllib.request.urlopen(url_request)
    text_str = text_str2.read()
    # print('review_text------->',text_str)
    # 1.Remove HTML
    soup = BeautifulSoup(text_str, "html.parser")

    # 1.Remove Tags
    [x.extract() for x in soup.findAll(['script', 'style', 'nonscript'])]
    [y.decompose() for y in soup.findAll(['span', 'li', 'noscript', 'footer',
                                          'title', 'a', 'h3', 'h2', 'h4', 'header'])]

    [x.extract() for x in (soup.select('[style~="visibility:hidden"]'))]
    [z.extract() for z in (soup.select('[style~="display:none"]'))]

    # print(soup.select('[style~="display:none"]'))

    for div in soup.findAll("div.cookie_accpt"):
        div.decompose()

    for div in soup.findAll("div", {'class': 'info'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'tags'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'cookie_stng hide'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'subscribe-me'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'glob_nav'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'subscribe hideit widget'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'agreement hide-on-submit'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'header'}):
        div.decompose()

    for div in soup.find_all("div", {'id': 'id_cookie_statement'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'column-22 col-md-3'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'col-md-4'}):
        div.decompose()

    for div in soup.find_all("div", {'class': 'col-md-1 col-12'}):
        div.decompose()

    for hidden in soup.find_all(style='display:none'):
        hidden.decompose()

        # review_text = soup.get_text(strip=True)
    review_text = soup.get_text(strip=True)
    # result = [sentence.delete() for sentence in review_text if "sign" in sentence]
    print('--->', review_text)
    summaryString = str(review_text)
    summaryString = summaryString.replace("\|`~-=_+", " ")
    summaryResult = run_summarization(summaryString.replace("\r", ""))
    summaryResult = summaryResult.replace("\n", "")
    summaryResult = summaryResult.replace("\t", "")
    summaryResult = summaryResult.replace("\r", "")

    return jsonify(pd.Series(summaryResult).to_json(orient='values'))


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    # adding beautiful soap logic to get clean text from html

    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.

        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    global average
    sumValues = 0

    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    try:
        # Average value of a sentence from original text
        average = (sumValues / len(sentenceValue))

    except ZeroDivisionError:
        print("Description is Empty:::Phising Webpage!")

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    # print('Inside Review Text',text)
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)
    # print('Inside freq_table------------',)
    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 0.8 * threshold)
    # print(summary)

    return summary

#Adding Content Summarization Model

@app.route('/contentSummary', methods=['POST'])
def contentSummary():

    print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_['data'])
    content = json_['data']
    text_str  =  content
    print('review_text------->',text_str)
    output =  run_content_summarization(text_str)
    #print("=====================output====================",output)
    return jsonify(pd.Series(output).to_json(orient='values'))

def run_content_summarization(text):
   # print('run_summarization=========>',text)
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)
   # print('Inside freq_table------------',freq_table)
    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 0.99 * threshold)
    print(summary)

    return summary
# Adding Text Classification Model API
@app.route('/newsClassification', methods=['POST'])
def newsClassification():
    # print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_)
    # inputData = json_
    url_test = pd.DataFrame([json_])
    print('url_test item data---->', url_test['data'])
    df = pd.read_excel("CywareClassificationData.xlsx", encoding='ISO-8859-1')
    #  df.dropna(subset=["TITLE"], inplace=True)
    df['TITLE'] = df.TITLE.astype(str).map(
        lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))

    X_train, X_test, y_train, y_test = train_test_split(
        (df['TITLE']),
        df['CATEGORY'],
        random_state=0
    )
    df['CATEGORY'] = df.CATEGORY.map({'Cyber Education and Awareness': 1,
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
    count_vector = CountVectorizer(stop_words='english')
    count_vector.fit_transform(X_train)
    input = url_test['data']
    print('input----------->', input)
    output = ClassificationModel_estimator.predict(count_vector.transform(input))
    print('output----------->', output)
    return jsonify(pd.Series(output).to_json(orient='values'))


# Malicious URL Prediction
@app.route('/urlPredict', methods=['POST'])
def urlPredict():
    json_ = request.json
    print('json:', json_)
    logger.info('json_ %s' % json_)
    # print('json_g---->',json_g)
    json_obj = json.dumps(json_)
    print('json_obj', json_['data'])
    logger.info('json_ %s' % json_obj)
    number_reviews_json = len(json_obj)  # Calculating the number of reviews  json_obj.encode("utf-8"))
    print('Length of _json', number_reviews_json)
    logger.info('Length of _json %s' % number_reviews_json)
    # testing = {'Url':https://zyxytr.com/acompanhamento}
    url_test = pd.DataFrame([json_])
    print('url_test item data---->', url_test['data'])
    logger.info('url_test item data----> %s' % str(url_test['data']))
    # Hostname Length
    url_test['hostname_length'] = len(urlparse(json_['data']).netloc)
    print('hostname_length------------>', url_test['hostname_length'])
    logger.info('hostname_length----> %s' % str(url_test['hostname_length']))

    # fd length
    def fd_length(url):
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    url_test['fd_length'] = fd_length(json_['data'])
    print('fd_length====================>>>', url_test['fd_length'])
    logger.info('fd_length---->%s' % str(url_test['fd_length']))

    # tld length
    # Length of Top Level Domain
    url_test['tld'] = get_tld(json_['data'], fail_silently=True)
    print('tld_name====================>>>', url_test['tld'])
    logger.info('tld_name----> %s' % str(url_test['tld']))

    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    url_test['tld_length'] = tld_length(url_test['tld'])
    print('tld Length------------------>', url_test['tld_length'])
    logger.info('tld Length----> %s' % str(url_test['tld_length']))
    url_test['count-'] = json_['data'].count('-')
    print('Count---------------->', url_test['count-'])
    logger.info('Count- ---> %s' % str(url_test['count-']))
    url_test['count@'] = json_['data'].count('@')
    print('Count-@---------------->', (url_test['count@']))
    logger.info('Count-@ ---> %s' % url_test['count@'])
    url_test['count?'] = json_['data'].count('?')
    print('Count?---------------->', (url_test['count?']))
    logger.info('Count-? --->%s' % url_test['count?'])
    url_test['count%'] = json_['data'].count('%')
    print('Count%---------------->', url_test['count%'])
    logger.info('Count ---> %s' % str(url_test['count%']))
    url_test['count.'] = json_['data'].count('.')
    print('Count..---------------->', url_test['count.'])
    logger.info('Coun.. --->%s' % str(url_test['count.']))
    url_test['count='] = json_['data'].count('=')
    print('Count=---------------->', url_test['count='])
    logger.info('Cout= ---> %s' % str(url_test['count=']))
    url_test['count-http'] = json_['data'].count('http')
    print('Count-http---------------->', url_test['count-http'])
    logger.info('Cout-http %s' % str(url_test['count-http']))
    url_test['count-https'] = json_['data'].count('https')
    print('Count-https---------------->', url_test['count-https'])
    logger.info('Cout-https %s' % str(url_test['count-https']))
    url_test['count-www'] = json_['data'].count('www')
    print('Count-www---------------->', url_test['count-www'])
    logger.info('Cout-www %s' % str(url_test['count-www']))

    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
            return digits

    url_test['count-digits'] = digit_count(json_['data'])
    logger.info('count-digits %s' % str(url_test['count-digits']))

    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
            return letters

    url_test['count-letters'] = letter_count(json_['data'])
    logger.debug('count-letters %s' % str(url_test['count-letters']))

    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    url_test['count_dir'] = no_of_dir(json_['data'])
    print("url_test['count_dir']============>", url_test['count_dir'])

    # logger.debug('count_dir %s' % str(url_test['count_dir']))

    # Use of IP or not in domain
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
            return 1

    # url_test['use_of_ip'] = url_test.apply(lambda i: having_ip_address(i))
    url_test['use_of_ip'] = having_ip_address(json_['data'])
    print("Use OF IP=============>", url_test['use_of_ip'])
    logger.debug('Use OF IP %s' % str(url_test['use_of_ip']))

    # Predictor Variables
    x = url_test[['hostname_length',
                  'fd_length', 'tld_length', 'count-', 'count@', 'count?',
                  'count%', 'count=', 'count-http', 'count-https', 'count-www', 'count-digits',
                  'count-letters', 'count_dir', 'use_of_ip']]

    print('query_df_array::----->', x)
    prediction = maliciousUrlModel_estimator.predict(x)
    print('Predicted Value--->', prediction)
    logger.debug('Predicted Value %s' % str(prediction))
    logger.info('Predicted Value %s' % str(prediction))
    return jsonify(pd.Series(prediction).to_json(orient='values'))

# for screen shot capturing  api
@app.route('/screenShot', methods=['POST'])
def screenShot():
    json_ = request.json
    print('json:', json_)
    logger.info('json_ %s'%json_)
    # print('json_g---->',json_g)
    json_obj = json.dumps(json_)
    print(json_['data'])
    inputurl = json_['data']
    logger.info('json_ %s'%json_obj)
    # adding imagekit lib for capturing scree shot on server
    inputurl = str(inputurl)
   # s = 'Part 1. Part 2. Part 3 then more text'
    if('https' in inputurl ):
        inputurlimg = re.search(r'https:(.*?)co', inputurl).group(1)
        inputurlimg = re.sub(r'[^\w]', '', inputurlimg)
    else:
        inputurlimg = re.search(r'http:(.*?)co', inputurl).group(1)
        inputurlimg = re.sub(r'[^\w]', '', inputurlimg)

    print(inputurlimg)
    try:
        options = {'xvfb': ''}
        imgOutput = inputurlimg + ".jpg"
        print(imgOutput)
        imgkit.from_url(inputurl, imgOutput, options=options)
        # print("image name::---->",inputurl+'.jpg')
        output = "Success"
    except Exception as e:
        output = "Failure"
        print(e)
    return output


MODEL_FILE = 'regression_model-v4.pkl'
ClassificationModel = 'naive_bayes_text_classifierV3.pkl'
log_estimator = joblib.load(MODEL_FILE)
maliciousUrlModel = 'log_model-v1.pkl'
cleanHeadlines_request = []
if __name__ == '__main__':
    logging.basicConfig(filename="MaliciouusUrlDetection.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_estimator = joblib.load(MODEL_FILE)
    json_obj = ''
    obj = ''
    ClassificationModel_estimator = joblib.load(ClassificationModel)
    maliciousUrlModel_estimator = joblib.load(maliciousUrlModel)
    #app.run(debug=True)
    #app.run(host='0.0.0.0')
    # to run in local
    app.run(port=8086)
