import csv
from csv import reader
from bs4 import BeautifulSoup
import re
from pymongo import MongoClient
import logging
from nltk.tokenize import word_tokenize, sent_tokenize
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By


class SoupScraper():
    # client = MongoClient('mongodb://appmongouser:XWDXJd2WIc3MFGI8@prod-shard-00-00-agp94.mongodb.net:27017,prod-shard-00-01-agp94.mongodb.net:27017,prod-shard-00-02-agp94.mongodb.net:27017,prod-shard-00-03-agp94.mongodb.net:27017,prod-shard-00-04-agp94.mongodb.net:27017,prod-shard-00-05-agp94.mongodb.net:27017,prod-shard-00-06-agp94.mongodb.net:27017/test?ssl=true&replicaSet=Prod-shard-0&authSource=admin&retryWrites=true&w=majority&readPreference=nearest')
    # db = client.core
    # collection = db.dark_web_search
    logging.basicConfig(filename="SoupScraper.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    def count_page(self, org_name_uri):
        url = ''
        try:
            last_page = ""
            if(url == ('http://dnpscnbaix6nkwvystl3yxglz7nteicqrou3t75tpcc5532cztc46qyd.onion')):
               print("")
               driver.get(url)
               page = driver.page_source
               soup = BeautifulSoup(page, 'html.parser')
               # print(soup)
               paging = soup.find("div", {"class": "row", "id": ""}).find("ul", {"class": "pagination"}).find_all("a")
            else:
                url = "http://searchcoaupi3csb.onion/search/?q=" + org_name + "&num=10&sort="
                driver.get(url)
                page = driver.page_source
                soup = BeautifulSoup(page, 'html.parser')
                # print(soup)
                paging = soup.find("div", {"class": "row", "id": ""}).find("ul", {"class": "pagination"}).find_all("a")

            #    print(paging)
            start_page = paging[1].text
            #   print("Start Page", start_page)
            last_page = paging[len(paging) - 2].text
        #  print("Last Page", last_page)
        except :
            self.logger.info('Timed out')


        return last_page

    def getLinks(self, domain, org_name):
        try:
            if domain == ("http://searchcoaupi3csb.onion/search/"):
                url = "http://searchcoaupi3csb.onion/search/?q="+org_name+"&num=10&sort="
                counter = scraper.count_page(url)
                print("counter value:::->", counter)
                self.logger.info('Total search pages found::: ' + counter)
                for x in range(int(counter)):
                    if x == 0:
                        url = "http://searchcoaupi3csb.onion/search/?q=" + org_name + "&num=10&sort="
                        try:

                            driver.get(url)
                            linkPageText = driver.page_source

                            soup = BeautifulSoup(linkPageText, 'lxml')
                            i = 0
                            for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
                                i = (i + 1)
                            #    print("linkCounter::->",i)
                            #    print("Link in for loop ::->", link['href'])

                                linkTextVal = driver.find_element_by_xpath('/html/body/main/div[2]/ol/'+'li['+str(i)+']/div[1]/div').text
                                linkTextkey = link.text
                                self.logger.info(link['href'])
                                keys = linkTextkey
                                dw[keys] = linkTextVal
                        except:
                            continue
                    else:

                        try:
                            url = "http://searchcoaupi3csb.onion/search/move/?q=" + org_name + "&pn=" + str(
                                x + 1) + "&num=10&sdh=&"

                            driver.get(url)
                            linkPageText = driver.page_source

                            soup = BeautifulSoup(linkPageText, 'lxml')
                            i = 0
                            for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
                                i = (i + 1)
                          #      print("linkCounter::->", i)
                                #    print("Link in for loop ::->", link['href'])
                                #Uncomment on server
                                # t = html.fromstring(linkPageText.content)
                                # linkTextVal = t.xpath(
                                #     '/html/body/main/div[2]/ol/' + 'li[' + str(i) + ']/div[1]/div/text()')

                                #Comment on server
                                linkTextVal = driver.find_element_by_xpath(
                                    '/html/body/main/div[2]/ol/' + 'li[' + str(i) + ']/div[1]/div').text
                                ##

                                linkTextkey = link.text
                                self.logger.info(link['href'])
                                keys = linkTextkey
                                dw[keys] = linkTextVal
                        except:
                            continue

                    print("Length of Dictionary:::", len(dw))


            elif(domain ==('http://nzxj3il7lr6qmouq.onion/trending/month')):
                counter = scraper.count_page(domain)
                print("counter value:::->", counter)
                for x in range (int(counter)):
                    domain = 'http://nzxj3il7lr6qmouq.onion/trending/month?page'+str(x+1)
                    print("Domain:::->", domain)
                    scraper._getUrlInfo(domain)
            elif (domain == ('http://dnpscnbaix6nkwvystl3yxglz7nteicqrou3t75tpcc5532cztc46qyd.onion')):
                counter = scraper.count_page(domain)
                print("counter-->", counter)
                xpath = xpathInput
                for x in range(int(counter)):
                    if (x != 0):
                        url = domain + "?page=" + str(x)
                        print("inside Domain")
                        driver.get(url)
                        linkTextVal = driver.find_element_by_xpath(
                            xpath).text
                        #   print("linkTextVal:::->",linkTextVal)
                        linkTextkey = url
                        #   self.logger.info(link['href'])
                        dw[linkTextkey] = linkTextVal
                        print(url.title() + "==========================")
            else:
                scraper._getUrlInfo(domain)
        except:
         self.logger.info('Exception at get Links method')

    def _getUrlInfo(self, url):
        print("URL in geturlinfo::----------->", url)
        try:
            i = 0
            homelink = ''
            if (url == "http://searchcoaupi3csb.onion/search/"):
                xpath = '/html/body/main/div[2]/ol/' + 'li[' + str(i) + ']/div[1]/div'
            elif (url == "http://depastedihrn3jtw.onion/top.php"):
                xpath = xpathInput
                homelink = "http://depastedihrn3jtw.onion/"
            elif (url == "http://suprbayoubiexnmp.onion/"):
                homelink = "http://suprbayoubiexnmp.onion/"
                xpath = xpathInput
            elif ("http://nzxj3il7lr6qmouq.onion/trending/month" in url):
                homelink = ""
                xpath =xpathInput
                xpathkey = '/html/body/div[2]/section/div/div/div[1]/div/div[1]'
                print("URL in strongPaste::::-", url)
            elif(url == ("http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion/")):
                xpath = '/html/body/div/div/div/div[2]/div/div/div'
            elif(url ==("http://hxt254aygrsziejn.onion/")):
                xpath = xpathInput
            driver.get(url)
            linkPageText = driver.page_source
            soup = BeautifulSoup(linkPageText, 'lxml')
            try:
                for link in soup.findAll('a', attrs={'href': re.compile("^")}):
                   # print("Link in for loop::->",link)

                    try:
                        i = (i + 1)
                        if(url == ("http://hxt254aygrsziejn.onion/") and (link['href'].__contains__('?p='))):
                                print(link['href'])
                                newlink = link['href']
                                driver.get(newlink)
                                linkTextVal = driver.find_element_by_xpath(
                                    xpath).text
                                linkTextkey = link.text
                                self.logger.info(link['href'])
                                dw[linkTextkey] = linkTextVal
                        else:
                            linklower = str(link.text).lower()

                        keywords = ["news", 'Announcements', "download", "upload", "files", "application"
                            , "software", "hack", "threat", "program", "cyber", "hardware", "network"
                            , "discuss", "communitie", "rss", "bitcoin", "paypal", "credit", "card",
                                    "btc", "onion", "investment", "market", "letter", "money", "prepaid", "lazrus",
                                    "hitman", "top", "privacy", "censor", "discussion", 'show paste', 'security',
                                    'deepweb', 'vpn','project','code','algorithm','virus','malware','windows','password'
                                    ,'read more','com','uk','p','Application']
                        if any(x in linklower for x in keywords):
                            if ('http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion/' in url):
                                xpath = xpathInput
                                homelink = 'http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion/'
                                newlink = (homelink + link['href']).replace("../../", "")
                                #newlink = (homelink + newlink)
                                #print("link['href']:::-----", link['href'])
                                #print("newlink after filter:::->", newlink)
                            elif('http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion/' in url):
                                homelink = 'http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion'

                                newlink = (homelink + link['href'])

                               # newlink = homelink
                                #Xpath for subpages
                                xpath = xpathInput
                               # Xpath for Mainpage
                              #  xpath = '/html/body/div/div/div'
                               # print("New Link in ransomeware-->", newlink)
                            elif (url == ('http://ekbgzchl6x2ias37.onion/') ):
                                homelink = 'http://ekbgzchl6x2ias37.onion'
                               # newlink = (homelink + link['href'])
                               # print("newlink:::->", newlink)
                                xpath =xpathInput
                            elif(url ==('http://dnpscnbaix6nkwvystl3yxglz7nteicqrou3t75tpcc5532cztc46qyd.onion')):
                                print("========================")
                                #xpath = '//*[@id="content-101"]'
                                homelink = ""
                                newlink = url
                            elif (url == ('http://hxt254aygrsziejn.onion/')):
                                newlink == link['href']
                                print("new link::->", newlink)
                            else:

                                # if ("http" in link['href']):
                                #     newlink =link['href']
                                #     #print("Link in if block ::->", newlink)
                                # else:
                                print("Inside else condition")
                                #newlink =  homelink + link['href']
                                    #print("Link in else block ::->", newlink)
                              #  newlink = (homelink + link['href'])
                            try:
                                if ("http" in link['href']):
                                    newlink = link['href']
                                   # print("Link in if block ::->", newlink)
                                else:
                                    newlink = homelink + link['href']
                                print("Link  ::->", newlink)
                                if (url == (
                                    "http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion/")):
                                    driver.get(newlink)
                                    wait = WebDriverWait(driver, 2)
                                    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "p")))
                                else:
                                   # print("newlink:::->", newlink)
                                    driver.get(newlink)
                                linkTextVal = driver.find_element_by_xpath(
                                    xpath).text
                                if ("http://suprbayoubiexnmp.onion/" in url):
                                    subPageSource = driver.page_source
                                    scraper.scrapSubLinks(subPageSource)
                                if("http://nzxj3il7lr6qmouq.onion/trending/month" in url):
                                    linkTextValkey = driver.find_element_by_xpath(
                                        xpathkey).text
                                  #  print(url+ "::->", linkTextValkey)
                                    linkTextkey = linkTextValkey
                                else:
                                    linkTextkey = link.text
                                self.logger.info(link['href'])
                                dw[linkTextkey] = linkTextVal
                            except:
                                self.logger.info('Timed Out')

                    except:
                        self.logger.info("Timed Out")
            except:
                self.logger.info("Timed Out")
        except:
            self.logger.info("Timed Out")

    def scrapSubLinks(self,linkTextVal):
        soup = BeautifulSoup(linkTextVal, 'lxml')
        i = 0
        xpath ="//*[@id='posts']"
        homelink ='http://suprbayoubiexnmp.onion/'
        for link in soup.findAll('a', attrs={'href': re.compile("^")}):
            i = (i + 1)
         #   print("linkCounter::->", i)
            if("http" not in link['href'] ):
                try:
                   newlink = homelink + link['href']
                   driver.get(newlink)
                   #print(newlink)
                   linkTextVal = driver.find_element_by_xpath('//*[@id="content"]').text
                   #print(linkTextVal)
                 #  value =  scraper.run_summarization(linkTextVal)
                    # ##############
                   linkTextkey = link.text
                  # self.logger.info(link['href'])
                   # print("Inside new code")
                   keys = linkTextkey
                   print(linkTextVal)
                   dw[keys] = linkTextVal
                   print("Dicti:::->",dw)
                except:
                    self.logger.info("Exeption")

    def _create_frequency_table(self, text_string) -> dict:
        """
        we create a dictionary for the word frequency table.
        For this, we should only use the words that are not part of the stopWords array.
        Removing stop words and making frequency table
        Stemmer - an algorithm to bring words to its root word.
        :rtype: dict
        """
      #  print('Inside Frequenct Table')
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

    def _score_sentences(self, sentences, freqTable) -> dict:
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

    def _find_average_score(self, sentenceValue) -> int:
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

    def _generate_summary(self, sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1

        return summary

    def run_summarization(self, text):
        # print('Inside Review Text',text)
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
        sentence_scores = scraper._score_sentences(sentences, freq_table)

        # 4 Find the threshold
        threshold = scraper._find_average_score(sentence_scores)

        # 5 Important Algorithm: Generate the summary
        summary = scraper._generate_summary(sentences, sentence_scores, 0.8 * threshold)
        return summary

    def update(self):
        try:
            db = connection.core
            collection = db.dark_web
            mydoc = collection.find()

            for key in mydoc:
                #  print(key['_id'])
                for key, value in dw.items():
                    record = {
                        "Title":key,
                        "Data": value
                    }
                    # print(key)

                    collection.insert_one(record)
        except Exception as e:
            logging.error("Exception Occured", exc_info=True)

#binary = FirefoxBinary('C:/Users/tahsin.asif/Desktop/Tor Browser/Browser/firefox')
service_args = ['--proxy=localhost:9150', '--proxy-type=socks5', ]
dataDic = {}

# scraper.getOrgs()
if __name__ =='__main__':
    #server Proxies iput
    proxies = {"http": "127.0.0.1:8118"}
    connection = MongoClient('localhost',27017)
    db = connection.core
    collection = db.dark_web
    driver = webdriver.PhantomJS(service_args=service_args)

    scraper = SoupScraper()
    dw = {}
    org_name = "exploit"
    # open file in read mode
    with open('DarkWebCSVInput.csv',
              'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj,delimiter = ',')
        # Iterate over each row in the csv using reader object
        for row in list(csv_reader):
            # row variable is a list that represents a row in csv
            domainName = ''.join(map(str,row[0]))
            xpathInput = ''.join(map(str,row[1]))
            print("domainName from CSV:::-",domainName)
            print("xpathInput from CSV:::-", xpathInput)
            scraper.getLinks(domainName,org_name)
            scraper.update()

            with open('darkweboutput.csv', mode='w', encoding='utf-8') as darkweboutput:
                 for k, v in dw.items():
                     print("Keys::------", k.encode("utf-8"))
                     print("Values::----", v.encode("utf-8"))
                    # collection.insert_many(dw)
                     employee_writer = csv.writer(darkweboutput, delimiter=',', quotechar='"',
                                                  quoting=csv.QUOTE_MINIMAL)
                     employee_writer.writerow([str(k) + "----------->", str(v)])

        print("Length of dict::->",len(dw))
