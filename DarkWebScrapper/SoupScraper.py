import requests
from bs4 import BeautifulSoup
import pymongo
import re
from pymongo import MongoClient
import MySQLdb 
import datetime
import logging
from csv import reader

from csv import reader
# open file in read mode
with open('......OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/DarkWebScrapper/DarkWebCSVInput.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        url = 'http://searchcoaupi3csb.onion/search/?q=' + str(row[0]) + '&ex_q=timestamp%3A[now%2Fd-1d+TO+*]&sdh=&'
        print(url)
       # print(row)
        
class SoupScraper():

#    client = MongoClient('mongodb://appmongouser:XWDXJd2WIc3MFGI8@prod-shard-00-00-agp94.mongodb.net:27017,prod-shard-00-01-agp94.mongodb.net:27017,prod-shard-00-02-agp94.mongodb.net:27017,prod-shard-00-03-agp94.mongodb.net:27017,prod-shard-00-04-agp94.mongodb.net:27017,prod-shard-00-05-agp94.mongodb.net:27017,prod-shard-00-06-agp94.mongodb.net:27017/test?ssl=true&replicaSet=Prod-shard-0&authSource=admin&retryWrites=true&w=majority&readPreference=nearest')
    client = MongoClient('localhost', 27017)
    db = client.core
    collection = db.dark_web_search
    logging.basicConfig(filename="SoupScraper.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    def getLinks(self, org_name, org_id, asset_type):
        self.logger.info('Starting getLinks for ' + org_name + ' org id ' + str(org_id))
        #url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name
        #url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1M+TO+*]&sdh=&'
        with open('............OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/DarkWebScrapper/DarkWebCSVInput.csv', 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
        # row variable is a list that represents a row in csv
                url = 'http://searchcoaupi3csb.onion/search/?q=' + row + '&ex_q=timestamp%3A[now%2Fd-1d+TO+*]&sdh=&'
                print(url)

        
        #url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1y+TO+*]&sdh=&'
                proxies = {}
                try:
                    page= requests.get(url, proxies=proxies, timeout=30)
                except requests.Timeout:
                    self.logger.info('Timed out')
            return
        soup= BeautifulSoup(page.text, 'html.parser')
        self.logger.info('Org:' + str(org_name))
        for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
            self.logger.info(link['href'])
            query = {"url" : link['href']}
            dws = self.collection.find(query)
            self.logger.info(dws.count())
            if dws.count() > 0:
                continue

            try:
                pg = requests.get(link['href'], proxies=proxies)
            except:
                continue
            try:
                pg_soup = BeautifulSoup(pg.text, 'html.parser')
            except:
                continue

            [x.extract() for x in pg_soup.findAll(['script', 'style', 'nonscript'])]
            [y.decompose() for y in pg_soup.findAll(['span', 'li', 'noscript', 'footer',
                                              'title', 'a','h3' ])]
            dw = {}
            dw['org_id'] = org_id
            dw['org_name'] = org_name
            dw['asset'] = asset_type
            dw['url'] = link['href']
            dw['content'] = pg_soup.get_text()
            dw['created_date'] = datetime.datetime.utcnow()
            self.collection.insert_one(dw)
        self.client.close()


    def getOrgs(self):
        self.logger.info('Starting getOrgs')
        db = MySQLdb.connect("jp-rds01.c7tohwlgi9s5.ap-northeast-1.rds.amazonaws.com","cyorguser","bkTQAp76YU29ZD45dBJ5CjvN","org")
        cursor = db.cursor()
        cursor.execute("SELECT * FROM org where status='Active'")
        numrows = cursor.rowcount
        for x in range(0,numrows):
            row = cursor.fetchone()
            self.getLinks(row[1], row[0], 'Company')
        db.close()

scraper = SoupScraper()
scraper.getOrgs()
