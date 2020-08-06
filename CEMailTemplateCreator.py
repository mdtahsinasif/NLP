import base64
import datetime
import glob
import os
import time
from csv import reader
import cv2
import pytesseract
import spacy
from pymongo import MongoClient
from selenium import webdriver

def clickPics():
    with open('C:\Backup\PycharmProjects\PycharmProjects\DarkWeb\HTMLURLInput.csv',
              'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj,delimiter = ',')
        # Iterate over each row in the csv using reader object
        for row in list(csv_reader):
            # row variable is a list that represents a row in csv
            try:
                domainName = ''.join(map(str,row[0]))
           #     print("domainName from CSV:::-",domainName)
                driver = webdriver.Chrome(DRIVER)
                driver.get(domainName)
                driver.set_window_size(1920, 2000)      #the trick
                time.sleep(2)
                date_stamp = str(datetime.datetime.now()).split('.')[0]
                date_stamp = date_stamp.replace(" ", "_").replace(":", "_").replace("-", "_")
                file_name = 'images/spear_phising_mail/'+date_stamp + ".png"
                screenshot = driver.save_screenshot(file_name)
          #      print("Image stored as ::->",screenshot)
                new_string = (file_name.split('/')[2])
                print('new_string------->',new_string)
                # domainNameKey = (domainName.split('\\')[8])
                # print('domainNameKey---------->',domainNameKey)
                data[new_string] = domainName
                driver.quit()
            except:
                print("Hello Bang There is an Error")


data = {}

def contentReader():

    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    for f1 in files:
        try:
           dw = {}
           print("image Name::->",f1.title())
           img = cv2.imread(f1)
           text = pytesseract.image_to_string(img)
          # data[f1] = text
           about_doc = nlp(text)
           sentences = list(about_doc.sents)
          # print("==============Text===========", text)
           dw['date'] = datetime.datetime.utcnow()
           dw['sentences']= text
           urlKey = (f1.title().split('\\')[8])
           print("Url--------->", urlKey)
           keyCheck = str(urlKey).lower()
           if keyCheck in imgUrlDict:
                print("---------url present----------", keyCheck)
                dw['URL'] = imgUrlDict[keyCheck]
         #   print("Data before Inserted=======>>>",data[keyCheck])
           collection.insert_one(dw)
           print("Data Inserted Successfully")
        except:
           print("image not clear")





if __name__ == "__main__":
    connection = MongoClient('localhost', 27017)
    db = connection.core
    collection = db.ce_mail_template_data
    #img_dir = "C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\covid_email_phishing_template"  # Enter Directory of all images
    img_dir = "C:\\Backup\\PycharmProjects\\PycharmProjects\\google-images-download\\google-images-download\\images\\bitcoin_phishing_template"  # Enter Directory of all images
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    #mypath = 'C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\covid_email_phishing_template'
    mypath = 'C:\\Backup\\PycharmProjects\\PycharmProjects\\google-images-download\\google-images-download\\images\\bitcoin_phishing_template'
    # webdriver.Chrome("C:\\Users\\TahsinAsif\\Downloads\\chromedriver.exe", options=options)
    DRIVER = 'C:\\Users\\TahsinAsif\\Downloads\\chromedriver.exe'
    nlp = spacy.load('en_core_web_sm')
    imgUrlDict ={
        "test1.png": "https://www.riskmanagementmonitor.com/spotting-coronavirus-related-phishing-emails/",
        "test2.png": "https://www.wsj.com/articles/hackers-target-companies-with-fake-coronavirus-warnings-11583267812",
        "test3.png": "https://www.proofpoint.com/us/corporate-blog/post/attackers-expand-coronavirus-themed-attacks-and-prey-conspiracy-theoriesd-coronavirus-themed-attacks-and-prey-conspiracy-theories",
        "test4.png": "https://www.aha.org/system/files/media/file/2020/02/coronavirus-themed-e-mail-phishing-health-sector-hc3-2-3-2020.pdf",
        "test5.png": "https://www.wired.com/story/coronavirus-phishing-scams/",
        "test6.png": "https://www.rexxfield.com/au/australian-scams-during-covid-19-crisis/",
        "test7.png": "https://www.rexxfield.com/au/australian-scams-during-covid-19-crisis/",
        "test8.png": "https://www.rexxfield.com/au/australian-scams-during-covid-19-crisis/",
        "test9.png": "https://www.rexxfield.com/au/australian-scams-during-covid-19-crisis/",
        "test10.png": "https://agio.com/newsroom/weaponized-coronavirus-phishing-how-to-protect-yourself/",
        "test11.png": "https://www.bbc.com/news/technology-51838468",
        "test12.png": "https://www.bbc.com/news/technology-51838468",
        "test14.png": "https://www.proofpoint.com/us/corporate-blog/post/attackers-expand-coronavirus-themed-attacks-and-prey-conspiracy-theoriesd-coronavirus-themed-attacks-and-prey-conspiracy-theories",
        "test15.png": "https://www.proofpoint.com/us/corporate-blog/post/attackers-expand-coronavirus-themed-attacks-and-prey-conspiracy-theoriesd-coronavirus-themed-attacks-and-prey-conspiracy-theories",
         ########################################Bitcoin#############################
        "bitcoin1.png":"https://securityboulevard.com/2019/09/bitcoin-phishing-the-n1ghtm4r3-emails/",
        "bitcoin2.png": "https://www.mailguard.com.au/blog/bitcoin-users-beware-phishing-email-scam-brandjacks-localbitcoins",
        "bitcoin3.png": "https://www.mailguard.com.au/blog/bitcoin-users-beware-phishing-email-scam-brandjacks-localbitcoins",
        "bitcoin4.png": "https://www.businessinsider.in/people-are-being-victimized-by-a-terrifying-new-email-scam-where-attackers-claim-they-stole-your-password-and-hacked-your-webcam-while-you-were-watching-porn-heres-how-to-protect-yourself/articleshow/65126033.cms",
        "bitcoin5.png": "https://www.reddit.com/r/Bitcoin/comments/6jcpgt/btc_phishing_email_warning_paymentbitpayorg_email/",
        "bitcoin6.png": "https://www.mailguard.com.au/blog/extortion-email-scam-demands-ransom-bitcoin-payment-uses-qr-code-to-provide-address",
        "bitcoin7.png": "https://securityboulevard.com/2019/09/bitcoin-phishing-the-n1ghtm4r3-emails/",
        "bitcoin8.png": "https://securityboulevard.com/2019/09/bitcoin-phishing-the-n1ghtm4r3-emails/",
        "bitcoin9.png": "https://securityboulevard.com/2019/09/bitcoin-phishing-the-n1ghtm4r3-emails/",
        "bitcoin10.png": "https://securityboulevard.com/2019/09/bitcoin-phishing-the-n1ghtm4r3-emails/",
        "bitcoin11.png": "https://www.kaspersky.com/blog/crypto-phishing/20765/",
        "bitcoin12.png": "https://www.kaspersky.com/blog/crypto-phishing/20765/",
        "bitcoin13.png": "https://blog.malwarebytes.com/scams/2019/08/the-lucrative-business-of-bitcoin-sextortion-scams/",
        "bitcoin14.png": "https://www.zdnet.com/article/this-is-what-happens-to-the-cryptocurrency-paid-out-through-sextortion-campaigns/",
        "bitcoin15.png": "https://blog.comodo.com/comodo-news/bitcoin-phishing-attack-on-cryptowallet-owner/",
        "bitcoin16.png": "https://symantec-enterprise-blogs.security.com/blogs/threat-intelligence/email-extortion-scams",
        "bitcoin17.png": "https://symantec-enterprise-blogs.security.com/blogs/threat-intelligence/email-extortion-scams",
        "bitcoin18.png": "https://www.pcrisk.com/removal-guides/15415-hacker-who-has-access-to-your-operating-system-email-scam",
        "bitcoin19.png": "https://blogs.cisco.com/security/your-money-or-your-life-digital-extortion-scams",
        "bitcoin20.png": "https://blogs.cisco.com/security/your-money-or-your-life-digital-extortion-scams",
        "bitcoin21.png":"https://www.bleepingcomputer.com/news/security/beware-of-extortion-scams-stating-they-have-video-of-you-on-adult-sites/"
    }
  #  clickPics()
    time.sleep(3)
    contentReader()
    #print(data)
