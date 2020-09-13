import csv
import datetime
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
import hashlib
import json



class SoupScraper():
    # client = MongoClient('mongodb://appmongouser:XWDXJd2WIc3MFGI8@prod-shard-00-00-agp94.mongodb.net:27017,prod-shard-00-01-agp94.mongodb.net:27017,prod-shard-00-02-agp94.mongodb.net:27017,prod-shard-00-03-agp94.mongodb.net:27017,prod-shard-00-04-agp94.mongodb.net:27017,prod-shard-00-05-agp94.mongodb.net:27017,prod-shard-00-06-agp94.mongodb.net:27017/test?ssl=true&replicaSet=Prod-shard-0&authSource=admin&retryWrites=true&w=majority&readPreference=nearest')
    # connection = MongoClient('localhost',27017)
    # db = connection.core
    # collection = db.dark_web_search
    logging.basicConfig(filename="SoupScraper.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # def count_page(self, org_name_uri):
    #     url = org_name_uri
    #     print("org_name_uri::->",org_name_uri)
    #     last_page = ""
    #     try:
    #         driver.get(url)
    #         page = driver.page_source
    #         soup = BeautifulSoup(page, 'html.parser')
    #       #  print(soup)
    #         paging = soup.find("div", {"class": "", "id": ""}).find("ul", {"class": "pagination"}).find_all("a")
    #         print(paging)
    #         start_page = paging[1].text
    #         print("Start Page", start_page)
    #         last_page = paging[len(paging) - 2].text
    #         #  print("Last Page", last_page)
    #     except TimeoutError:
    #         self.logger.info('Timed out')
    #     return last_page

    def getLinks(self, domain, org_name):
        try:
            xpath = xpathInput

            if(nonPagination):
               print("Domain in Use::->", domain)
               subUrlReturnContainer = scraper.findSubUrls(domain)
               for subUrls in subUrlReturnContainer:
                   #print ("subUrls::->",subUrls['href'])
                   if (domain == "http://p6o7m73ujalhgkiv.onion"):
                       sublink = (subUrls)
                       newlink = domain + '/?' + sublink
                       scraper._getUrlInfo(newlink, sublink)
                   elif ("http" in subUrls['href'] ):
                       newlink = subUrls['href']
                       scraper._getUrlInfo(newlink, subUrls['href'])
                   elif(('http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion' in domain)):
                       newlink = str(subUrls['href']).replace(".","")
                       newlink = domain + newlink
                       scraper._getUrlInfo(newlink, subUrls['href'])
                   else:
                       if(domain == "http://depastedihrn3jtw.onion/top.php"):
                           domain = "http://depastedihrn3jtw.onion/"
                       newlink = domain + subUrls['href']
                  # print(" Updated Links ::->", newlink)
                       scraper._getUrlInfo(newlink,subUrls['href'])
            else:
                counter = 7
                print("counter-->", counter)
                for x in range(int(counter)):
                    url = domain + "?page=" + str(x+1)
                    print("Domains::->", url)
                    scraper._getUrlInfo(url,url['href'])
        except TimeoutError:
          self.logger.info('Exception at get Links method')


    def findSubUrls(self,url):
        try:
            jsonUrlLink =[]
            subUrlRepo =[]
            driver.get(url)
            linkPageText = driver.page_source
            soup = BeautifulSoup(linkPageText, 'lxml')
            #print(soup)
            if (url == ("http://hxt254aygrsziejn.onion")):
                regexVal = '[p=]'
            else:
                regexVal = '^'
            if (url == ("http://p6o7m73ujalhgkiv.onion")):
                    wait = WebDriverWait(driver, 2)
                    scripts = soup.find_all('script', {'type': 'text/javascript'})[-1]
                    scriptString = soup.text
                    print(scripts.text)
                    for item in scriptString.split("\n"):
                        if "var post_links " in item:
                            value = item.strip()
                            before_keyword, keyword, after_keyword = value.partition('=')
                            new_str = after_keyword.replace('[', '')
                            new_str = new_str.replace(']', '')
                            new_str = new_str.replace(';', '')
                            new_str = new_str.replace('},', '}|')
                            li = list(new_str.split("|"))
                            # print(li)
                            for i in li:
                                jsonValue = json.loads(i)
                                #  print( jsonValue['link'])
                                jsonUrlLink.append(jsonValue['link'])
                            print(jsonUrlLink)
                    return jsonUrlLink

            for link in soup.findAll('a', attrs={'href': re.compile(regexVal)}):
            #for link in soup.findAll('a'):
            #   print("Link before Filter::->",link.text.encode("utf-8"))
               linkCheck = (link.text.encode("utf-8")).lower()
           #    print("linkCheck:::->",linkCheck)
               keywords = ["sex","male","female","porn","mastur","drug"]
               excludes = ["sex", "porn", "p0rn", "girl", "taboo", "fetish","male","female","mastur","drug","fuck","tor","girl","boy"]
               a_match = [True for match in keywords if match in str(linkCheck)]
               if True in a_match:
                    self.logger.info("Contain improper words")
               else:
                    print("Link after Filter::->", link.text.encode("utf-8"))
                    title_lower = str(link.text.encode("utf-8")).lower()
                    if any(ele in title_lower for ele in excludes):
                        print("in else")
                        print(title_lower)
                        continue
                    subUrlRepo.append(link)
            return subUrlRepo
        except TimeoutError:
            self.logger.info("Time Out Error")


    def _getUrlInfo(self, url,subUrlsVal):
            print("Inside geturlinfo::----------->", url)
            try:
                i = 0
                dw = {}
               # homelink = url
               # print("before driver get url")
                xpath = ''
                if "http://hxt254aygrsziejn.onion/?p=" in url:
                    postNo = url[33:len(url)]
                    xpath = '//*[@id="post-' + postNo + '"]'
                driver.get(url)
                #print("After driver get url")
                if(waitforLoad):
                    wait = WebDriverWait(driver, 10)
                    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "p")))
                linkPageText = driver.page_source
                #print(linkPageText)
                #print("before Xpath Input")
                if xpath != '':
                    linkTextVal = driver.find_element_by_xpath(xpath).text
                else:
                    linkTextVal = driver.find_element_by_xpath(xpathInput).text
                print(linkTextVal)
                #print("After Xpath Input",linkTextVal)
                if (("http://p6o7m73ujalhgkiv.onion") in url):
                    soup = BeautifulSoup(linkPageText, 'lxml')
                    scripts = soup.find_all('script', {'type': 'text/javascript'})[-1]
                    scriptString = scripts.text
                    #print("scriptString______________",scriptString)
                    for item in scriptString.split("\n"):
                        if "var source_content" in item:
                            removeItems = ['insert', 'attributes', 'blockTitle','true'
                                , 'ops','Website','blockAuthor','textBreak',' n ']
                            value = item.strip()
                            #print("Script Value :::->",value)
                            before_keyword, keyword, after_keyword = value.partition('=')
                            #print("after_keyword----------->",after_keyword)
                            new_str = after_keyword.replace('(`', '')
                            new_str = re.sub(r"[^a-zA-Z0-9]+", ' ', new_str)
                            for remove in removeItems:
                                new_str = new_str.replace(remove, '')

                    linkTextVal = new_str
                    print(url+":::->", linkTextVal)
                    linkTextValLower = str(linkTextVal).lower()
                    linkTextkey = url
                else:
                    linkTextValLower = str(linkTextVal).lower()
                    #    print(url+":::->", linkTextVal)
                    linkTextkey = url
                self.logger.info(url)
                content_hash = ''
                excludes = ["sex", "porn", "p0rn", "girl", "taboo", "fetish","boys","mastur","drugs","suicide"]
                if any(ele in linkTextValLower for ele in excludes):
                    print("censored Content111::->",url)
                else:
                    query = {"url": linkTextkey}
                    dws = collection.find(query)
                    self.logger.info(dws.count())
                    new_hash = hashlib.sha224(linkTextVal.encode('utf-8')).hexdigest()
                    print("New hash:" + new_hash)
                    if dws.count() > 0:
                        existing_dws = dws[0]
                        existing_hash = dws[0]['hash']
                        print("Existing hash:" + existing_hash)
                        if new_hash == existing_hash:
                            print("CONTENT NOT CHANGED")
                        else:
                            existing = 'true'
                            dw = dws[0]
                            query = {"_id":existing_dws['_id']}
                            update_stmt = {"$set":{"modified_date":datetime.datetime.utcnow(),"content":linkTextVal,"hash":new_hash}}
                            self.collection.update_one(query,update_stmt)
                            print("updating in DB")
                    else:
                        dw['url'] = linkTextkey
                        dw['title'] = subUrlsVal.title()
                        dw['content'] = linkTextVal
                        dw['created_date'] = datetime.datetime.utcnow()
                        dw['source'] = source
                        dw['hash'] = new_hash
                        print("Inserting in DB 111")
                        #   dw[linkTextkey] = linkTextVal
                        collection.insert_one(dw)
                        self.logger.info('Inserted successfull in DB')
                if(subPageScrapping):
                     scraper.scrapSubLinks(linkPageText,subUrlsVal)
            except Exception as e:
                print("Error111")
                print(e)
                self.logger.info("::===========Time out error ============::")


    def scrapSubLinks(self,linkTextVal,subUrlsVal):
        print("In scrapeSubUrls")
        soup = BeautifulSoup(linkTextVal, 'lxml')
        i = 0
       # subXpath ="'//*[@id='content']'"
        homelink = domainName
        for link in soup.findAll('a', attrs={'href': re.compile("^")}):
            i = (i + 1)
            dw = {}
            print("subpage scrapping link::->", link['href'])
            if("http" not in link['href'] ):
                try:
                    if(str(link['href']).__contains__("../")):
                         correctLink = str(link['href']).replace("../", "/")
                    elif(str(link['href']).__contains__("./")):
                        correctLink = str(link['href']).replace("./","/")
                    elif (str(link['href']).__contains__("../../")):
                        correctLink = str(link['href']).replace("../../", "/")
                    else:
                        newlink = homelink + correctLink
                   # if(newlink.__contains__("../")):
                   #     newlink.replace("../", "/")
                    print("subpage new link::->",newlink)
                    driver.get(newlink)
                except:
                    self.logger.info("rror in link",newlink)
            else:
                try:
                    driver.get(link['href'])
                except:
                    self.logger.info(link,":::->Not workin")
            try:
                print(newlink)
                excludes = ["sex", "porn", "p0rn", "girl", "taboo", "fetish","boys","mastur","drugs","suicide"]
                linkTextVal = driver.find_element_by_xpath(subXpathInput).text
                linkTextValLower = str(linkTextVal).lower()
                if any(ele in linkTextVal for ele in excludes):
                    continue
                # print(linkTextVal)
                #  value =  scraper.run_summarization(linkTextVal)
                linkTextkey = link.text
                # self.logger.info(link['href'])
                # print("Inside new code")
                keys = linkTextkey
                #   print(linkTextVal)
                query = {"url": linkTextkey}
                dws = collection.find(query)
                self.logger.info(dws.count())
                if dws.count() > 0:
                    dw['content'] = linkTextVal
                    dw['updated_date'] = datetime.datetime.utcnow()
                    print("updating in DB")
                    #   dw[linkTextkey] = linkTextVal
                    collection.update(dw)
                else:
                    print("Inside After exlude condition in sub scrapping method------->",link)
                    dw['url'] = linkTextkey
                    dw['title'] = subUrlsVal.title()
                    dw['content'] = linkTextVal
                    dw['created_date'] = datetime.datetime.utcnow()
                    dw['source'] = source
                    print("Inserting in DB")
                    #   dw[linkTextkey] = linkTextVal
                    collection.insert_one(dw)
                    print("Inserted successfull in DB in subscrapping method")
                    #   print("Dicti:::->",dw)

            except:
                self.logger.info("Exception")

#service_args = ['--proxy=127.0.0.1:8118', '--proxy-type=http', ]
service_args = ['--proxy=localhost:9150', '--proxy-type=socks5', ]
dataDic = {}

# scraper.getOrgs()
if __name__ =='__main__':
    #server Proxies iput
    proxies = {"http": "127.0.0.1:8118"}
    connection = MongoClient('localhost',27017)
    db = connection.core
    dw = {}
    collection = db.dark_web_search_test
    driver = webdriver.PhantomJS(service_args=service_args)

    scraper = SoupScraper()

    org_name = "exploit"
    waitforLoad = False
    subPageScrapping = False
    nonPagination = False
    # open file in read mode
    with open('DarkWebCSVInput.csv',
              'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj,delimiter = ',')
        # Iterate over each row in the csv using reader object
        for row in list(csv_reader):
            # row variable is a list that represents a row in csv
            domainName = ''.join(map(str,row[0]))
            xpathInput = ''.join(map(str,row[2]))
            source = ''.join(map(str,row[1]))
            try :
                subXpathInput =''.join(map(str,row[3]))
                subPageScrapping = True
                print("subXpathInput from CSV:::-", subXpathInput)
            except:
                 print("No Subpath value")

            # if(domainName =="http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion"):
            #     subXpathInput = '/html/body/div[2]/div[2]/div/div[3]'
            # else:
            #     subXpathInput = '//*[@id="content"]'

            print("domainName from CSV:::-",domainName)
            print("xpathInput from CSV:::-", xpathInput)
            print("source from CSV:::-", source)


            if (domainName == ("http://ekbgzchl6x2ias37.onion") or ("http://hxt254aygrsziejn.onion")
                                                                or ("http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion")
                                                                or ("http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion")
                                                                or ("http://depastedihrn3jtw.onion")
                                                                or ("http://p6o7m73ujalhgkiv.onion")):
                nonPagination = True
            # if(domainName == ("http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion")or
            #         ("http://suprbayoubiexnmp.onion/")) :
            #     subPageScrapping = True
            #if (domainName == "http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion"):
            #    waitforLoad = True

            #if(domainName == "http://ekbgzchl6x2ias37.onion"):
            #    source = 'CL0P^_- LEAKS'

            #if(domainName == "http://hxt254aygrsziejn.onion"):
            #    source = 'CORPORATE LEAKS'

            #if(domainName == "http://dnpscnbaix6nkwvystl3yxglz7nteicqrou3t75tpcc5532cztc46qyd.onion"):
            #    source = 'Happy Blog'
            waitforLoad = True
            scraper.getLinks(domainName,org_name)
          #  scraper.update()

            with open('darkweboutput.csv', mode='w', encoding='utf-8') as darkweboutput:
                 for k, v in dw.items():
                     print("Keys::------")
                     print("Values::----" )
                    # collection.insert_many(dw)
                     employee_writer = csv.writer(darkweboutput, delimiter=',', quotechar='"',
                                                  quoting=csv.QUOTE_MINIMAL)
                     employee_writer.writerow([str(k) + "----------->", str(v)])

        print("Length of dict::->",len(dw))
