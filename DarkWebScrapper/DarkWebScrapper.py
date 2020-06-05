# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:25:06 2020

@author: tahsin.asif
"""

from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import pandas as pd

binary = FirefoxBinary('C:/Users/tahsin.asif/Desktop/Tor Browser/Browser/firefox')
service_args = [ '--proxy=localhost:9150', '--proxy-type=socks5', ]

driver = webdriver.PhantomJS(service_args=service_args)
#get page
#driver.get("https://stackoverflow.com/questions/40161921/how-to-open-and-close-tor-browser-automatically-with-python")
print(driver.page_source)

##for captcha reading
url = "http://finedumps3ml4k3u.onion/"
driver.get(url)
driver.implicitly_wait(10000)
#pip install cssselect
#pip install lxml
import lxml.html
import urllib.request as urllib2
import pprint
import http.cookiejar as cookielib
def form_parsing(html):
   tree = lxml.html.fromstring(html)
   data = {}
   for e in tree.cssselect('form input'):
      if e.get('name'):
         data[e.get('name')] = e.get('value')
   return data
REGISTER_URL = 'thehuboy27kracz6sdql2r7c324vrs5aok2e33gorrikccaqhvzfcvad.onion'
ckj = cookielib.CookieJar()
browser = urllib2.build_opener(urllib2.HTTPCookieProcessor(ckj))
driver.get(url)
html = (driver.find_element_by_xpath('/html').text)
print(html)
html = browser.open('http://avengersdutyk3xf.onion/').read()
form = form_parsing(html)
pprint.pprint(form)



#######
#driver = webdriver.Firefox(firefox_binary = binary)
#url = "http://suprbayoubiexnmp.onion/"
org_name = 'Mitsubushi'
url = 'http://searchcoaupi3csb.onion/search/?q=' + org_name + '&ex_q=timestamp%3A[now%2Fd-1d+TO+*]&sdh=&'
print(url)
driver.get(url)
#print(driver.find_element_by_class_name("mw-redirect").text)
print(driver.find_element_by_xpath('/html').text)
driver.find_elements_by_class_name("postMain")
driver.find_element_by_class_name('postContent').text
post_content_list = []
#postText = str(driver.find_element_by_class_name('mw-redirect').text)
postText = str(driver.find_element_by_xpath('/html').text)
print(postText)

from bs4 import BeautifulSoup
html = postText
soup = BeautifulSoup(html, "html.parser")
soup.findAll('a')
one_a_tag = soup.findAll("a")[36]
link = one_a_tag["href"]




print(post_content_list.append(driver.find_element_by_xpath('/html').text))

for i in range(1, MAX_PAGE_NUM + 1):
  page_num = i
  url = '*first part of url*' + str(page_num) + '*last part of url*'
  driver.get(url)
  
df['postURL'] = post_url_list
df['author'] = post_author_list
df['postTitle'] = post_title_list
df.to_csv('scrape.csv')

  

     