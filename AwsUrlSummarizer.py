# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:02:15 2020

@author: tahsin.asif
"""
from flask import Flask, jsonify, request
import pandas as pd
import urllib.request



app = Flask(__name__)



@app.route('/summary', methods=['POST'])
def summary():
     headers = {}
     headers[
         'User-Agent'] = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
     json_= request.json
     print ('json:',json_['data'])
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
                                                  'title', 'a', 'h3','h2','h4','header'])]

     [x.extract() for x in (soup.select('[style~="visibility:hidden"]'))]
     [z.extract() for z in (soup.select('[style~="display:none"]'))]

            #print(soup.select('[style~="display:none"]'))

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

     for hidden in soup.find_all(style='display:none'):
         hidden.decompose()
        
            #review_text = soup.get_text(strip=True)
     review_text = soup.get_text(strip=True)
            # result = [sentence.delete() for sentence in review_text if "sign" in sentence]
     print('--->',review_text)
            
     ###############################
     #output = log_estimator.predict(count_vect.transform(url_test['data']))
     return jsonify(pd.Series(review_text).to_json(orient='values'))


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
     
     
if __name__ == '__main__':
  
     app.run(port=8086)
