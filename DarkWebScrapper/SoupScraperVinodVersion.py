import requests
from bs4 import BeautifulSoup
import pymongo
import re
from pymongo import MongoClient
# import MySQLdb
import datetime
import logging

from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


class SoupScraper():
 
    client = MongoClient('localhost', 27017)
    db = client.core
    collection = db.dark_web
    print('Connected to dark-web collection')
    logging.basicConfig(filename="SoupScraper.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # getLinks(self)

    def getLinks(self):
        print('Inside Getlinks')
        org_name = 'Mitsubushi'
        # self.logger.info('Starting getLinks for ' + org_name + ' org id ' + str(org_id))
        # url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name
        # url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1M+TO+*]&sdh=&'
        #        url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1d+TO+*]&sdh=&'
        # print(url)
        url = 'https://www.cars.com/shopping/'
        print(url)
        # url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1y+TO+*]&sdh=&'
       
        # proxies = {"http": "localhost:9150"}
        try:
            #  page= requests.get(url, proxies=proxies, timeout=30)
            driver.get(url)
            print('inside try-----------------')
            (driver.get(url))
            # print(driver.find_element_by_xpath('/html').text)
            pageText = (driver.find_element_by_xpath('/html').text)
            print('Inside try::', pageText)
            result = scraper.run_summarization(pageText)
            print('Result:::--------->', result)
        except requests.Timeout:
            self.logger.info('Timed out')
            return
        soup = BeautifulSoup(pageText, 'html.parser')
        self.logger.info('Org:' + str(org_name))
        for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
            self.logger.info(link['href'])
            query = {"url": link['href']}
            dws = self.collection.find(query)
            self.logger.info(dws.count())
            if dws.count() > 0:
                continue

            try:
                # pg = requests.get(link['href'], proxies=proxies)
                pg = driver.get(link['href'])
                print('pg-------------->', pg)
            except:
                continue
            try:
                pg_soup = BeautifulSoup(pg.text, 'html.parser')
                print('pg_soup---------->', pg_soup)
              #  result =  scraper.run_summarization(pg_soup)
                #print('Result:::--------->',result)
            except:
                continue

            [x.extract() for x in pg_soup.findAll(['script', 'style', 'nonscript'])]
            [y.decompose() for y in pg_soup.findAll(['span', 'li', 'noscript', 'footer',
                                                     'title', 'a', 'h3'])]
            # dw = {}
            # dw['org_id'] = org_id
            # dw['org_name'] = org_name
            # dw['asset'] = asset_type
            # dw['url'] = link['href']
            # dw['content'] = pg_soup.get_text()
            # dw['created_date'] = datetime.datetime.utcnow()
           # self.collection.insert_one(dw)
        self.client.close()

    def _create_frequency_table(self,text_string) -> dict:
        """
        we create a dictionary for the word frequency table.
        For this, we should only use the words that are not part of the stopWords array.
        Removing stop words and making frequency table
        Stemmer - an algorithm to bring words to its root word.
        :rtype: dict
        """
        print('Inside Frequenct Table')
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

    def _score_sentences(self,sentences, freqTable) -> dict:
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

    def _find_average_score(self,sentenceValue) -> int:
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

    def _generate_summary(self,sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1

        return summary

    def run_summarization(self,text):
        print('Inside Review Text',text)
        # 1 Create the word frequency table
        freq_table = scraper._create_frequency_table(text)
        # print('Inside freq_table------------',)
        '''
        We already have a sentence tokenizer, so we just need 
        to run the sent_tokenize() method to create the array of sentences.
        '''

        # 2 Tokenize the sentences
        sentences = sent_tokenize(text)

        # 3 Important Algorithm: score the sentences
        sentence_scores =scraper._score_sentences(sentences, freq_table)

        # 4 Find the threshold
        threshold = scraper._find_average_score(sentence_scores)

        # 5 Important Algorithm: Generate the summary
        summary = scraper._generate_summary(sentences, sentence_scores, 0.8 * threshold)
        return summary


#
#    def getOrgs(self):
#        self.logger.info('Starting getOrgs')
#       
#
#        cursor = db.cursor()
#        cursor.execute("SELECT * FROM org where status='Active'")
#        numrows = cursor.rowcount
#        for x in range(0,numrows):
#            row = cursor.fetchone()
#            self.getLinks(row[1], row[0], 'Company')
#        db.close()
binary = FirefoxBinary(......Tor Browser/Browser/firefox')
service_args = ['--proxy=localhost:9150', '--proxy-type=socks5', ]

driver = webdriver.PhantomJS(service_args=service_args)
scraper = SoupScraper()
scraper.getLinks()
# scraper.getOrgs()
