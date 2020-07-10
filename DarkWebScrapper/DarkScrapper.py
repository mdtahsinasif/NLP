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
                   if ("http" in subUrls['href'] ):
                       newlink = subUrls['href']
                   elif(('http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion' in domain)):
                       newlink = str(subUrls['href']).replace(".","")
                       newlink = domain + newlink
                   else:
                       if(domain == "http://depastedihrn3jtw.onion/top.php"):
                           domain = "http://depastedihrn3jtw.onion/"
                       newlink = domain + subUrls['href']
                  # print(" Updated Links ::->", newlink)
                   scraper._getUrlInfo(newlink)
            else:
                counter = 7
                print("counter-->", counter)
                for x in range(int(counter)):
                    url = domain + "?page=" + str(x+1)
                    print("Domains::->", url)
                    scraper._getUrlInfo(url)
        except TimeoutError:
          self.logger.info('Exception at get Links method')


    def findSubUrls(self,url):
        try:
            subUrlRepo =[]
            driver.get(url)
            linkPageText = driver.page_source
            soup = BeautifulSoup(linkPageText, 'lxml')
            if (url == ("http://hxt254aygrsziejn.onion")):
                regexVal = '[p=]'
            else:
                regexVal = '^'

            for link in soup.findAll('a', attrs={'href': re.compile(regexVal)}):
            #   print("Link before Filter::->",link.text.encode("utf-8"))
               linkCheck = (link.text.encode("utf-8")).lower()
           #    print("linkCheck:::->",linkCheck)
               keywords = ["sex","male","female","porn","mastur","drug"]
               a_match = [True for match in keywords if match in str(linkCheck)]
               if True in a_match:
                    self.logger.info("Contain improper words")
               else:
                    print("Link after Filter::->", link.text.encode("utf-8"))
                    subUrlRepo.append(link)
            return subUrlRepo
        except TimeoutError:
            self.logger.info("Time Out Error")


    def _getUrlInfo(self, url):
            print("Inside geturlinfo::----------->", url)
            try:
                i = 0
               # homelink = url
               # print("before driver get url")
                driver.get(url)
                #print("After driver get url")
                if(waitforLoad):
                    wait = WebDriverWait(driver, 2)
                    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "p")))
                linkPageText = driver.page_source
                #print("before Xpath Input")
                linkTextVal = driver.find_element_by_xpath(xpathInput).text
                #print("After Xpath Input",linkTextVal)
                print(url+":::->", linkTextVal)
                linkTextkey = url
                self.logger.info(url)
                dw[linkTextkey] = linkTextVal
                if(subPageScrapping):
                     scraper.scrapSubLinks(linkPageText)
            except :
                 self.logger.info("::===========Time out error ============::")




    def scrapSubLinks(self,linkTextVal):
        soup = BeautifulSoup(linkTextVal, 'lxml')
        i = 0
       # subXpath ="'//*[@id='content']'"
        homelink = domainName
        for link in soup.findAll('a', attrs={'href': re.compile("^")}):
            i = (i + 1)
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
                    #print(newlink)
                    linkTextVal = driver.find_element_by_xpath(subXpathInput).text
                   #print(linkTextVal)
                 #  value =  scraper.run_summarization(linkTextVal)
                    linkTextkey = link.text
                  # self.logger.info(link['href'])
                   # print("Inside new code")
                    keys = linkTextkey
                 #   print(linkTextVal)
                    dw[keys] = linkTextVal
                 #   print("Dicti:::->",dw)
                except:
                    self.logger.info("Exception")



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
            xpathInput = ''.join(map(str,row[1]))
            try :
                subXpathInput =''.join(map(str,row[2]))
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


            if (domainName == ("http://ekbgzchl6x2ias37.onion") or ("http://hxt254aygrsziejn.onion")
                                                                or ("http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion")
                                                                or ("http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion")
                                                                or ("http://depastedihrn3jtw.onion")):
                nonPagination = True
            # if(domainName == ("http://answerszuvs3gg2l64e6hmnryudl5zgrmwm3vh65hzszdghblddvfiqd.onion")or
            #         ("http://suprbayoubiexnmp.onion/")) :
            #     subPageScrapping = True
            if (domainName == "http://hpoo4dosa3x4ognfxpqcrjwnsigvslm7kv6hvmhh2yqczaxy3j6qnwad.onion"):
                waitforLoad = True

            scraper.getLinks(domainName,org_name)
          #  scraper.update()

            with open('darkweboutput.csv', mode='w', encoding='utf-8') as darkweboutput:
                 for k, v in dw.items():
                     print("Keys::------", k.encode("utf-8"))
                     print("Values::----", v.encode("utf-8"))
                    # collection.insert_many(dw)
                     employee_writer = csv.writer(darkweboutput, delimiter=',', quotechar='"',
                                                  quoting=csv.QUOTE_MINIMAL)
                     employee_writer.writerow([str(k) + "----------->", str(v)])

        print("Length of dict::->",len(dw))
