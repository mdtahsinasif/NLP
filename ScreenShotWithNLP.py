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
           #imgdb = cv2.imread(f1)
           text = pytesseract.image_to_string(img)
          # data[f1] = text
           about_doc = nlp(text)
           sentences = list(about_doc.sents)
           print("==============Text===========", text)
           dw['date'] = datetime.datetime.utcnow()
           encoded_string = base64.b64encode(cv2.imread(f1))
    #        print("==encoded_string==", encoded_string)
           dw['encoded_string'] = encoded_string
           dw['sentences']= text
           #   abc = collection.insert(dw)
           url = (f1.title().split('\\')[8])
         #  print("Url--------->", url)
           keyCheck = str(url).lower()
           if keyCheck in data:
               print("---------url present----------", url)
               dw['URL']= data[keyCheck]
           print("Data before Inserted=======>>>",data[keyCheck])
           collection.insert_one(dw)
           print("Data Inserted Successfully")
           # decode = encoded_string.decode()
           # print("==Decode==",decode)
           # img_tag = '<img alt="sample" src="data:image/png;base64,{0}">'.format(decode)
           # #Inserting Image to DB
           # img = cv2.imshow(img_tag)
        except:
           print("image not clear")


# def insert_image():
#     with open('C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\spear_phising_mail',
#               'r') as image_file:
#         encoded_string = base64.b64encode(image_file.read())


if __name__ == "__main__":
    connection = MongoClient('localhost', 27017)
    db = connection.core
    collection = db.ce_mail_template_data
    img_dir = "C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\spear_phising_mail"  # Enter Directory of all images
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    mypath = 'C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\spear_phising_mail'
    # webdriver.Chrome("C:\\Users\\TahsinAsif\\Downloads\\chromedriver.exe", options=options)
    DRIVER = 'C:\\Users\\TahsinAsif\\Downloads\\chromedriver.exe'
    nlp = spacy.load('en_core_web_sm')
    clickPics()
    time.sleep(3)
    contentReader()
    print(data)
