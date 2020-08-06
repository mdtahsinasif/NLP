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
                new_string = (file_name.split('/')[2])
                print('new_string------->',new_string)
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
           about_doc = nlp(text)
           sentences = list(about_doc.sents)
           #print("==============Text===========", text)
           dw['date'] = datetime.datetime.utcnow()
           image = open(f1, "rb")
           image_read = image.read()
           encoded_bytes = base64.encodebytes(image_read)
           print("encoded_bytes======>", encoded_bytes)
           #To Decode Image data
           # image_64_decode = base64.decodebytes(encoded_bytes)
           # image_result = open('deer_decode.gif', 'wb')  # create a writable image and write the decoding result
           # image_result.write(image_64_decode)

           dw['Image'] = encoded_bytes
           dw['sentences']= text
           url = (f1.title().split('\\')[8])
           print("Url--------->", url)
           keyCheck = str(url).lower()
           if keyCheck in data:
               print("---------url present----------", url)
               dw['URL']= data[keyCheck]
           print("Data before Inserted=======>>>",data[keyCheck])
           collection.insert_one(dw)
           print("Data Inserted Successfully")

        except:
           print("image not clear")


if __name__ == "__main__":
    connection = MongoClient('localhost', 27017)
    db = connection.core
    collection = db.ce_mail_template_data_check
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
