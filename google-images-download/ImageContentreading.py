#pip install tesseract
#pip install tesseract-ocr
import pytesseract
from PIL import Image

from os import listdir
from os.path import isfile, join
import numpy
import cv2
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
mypath='C:\Backup\PycharmProjects\PycharmProjects\DarkWeb\phising_photos'
# onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
# images = numpy.empty(len(onlyfiles), dtype=object)
# for n in range(0, len(onlyfiles)):
#   images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
#   #img = Image[n].open("PhishingEmail3.png")
#   text = pytesseract.image_to_string(images[n].getpixel((325,432)))
#   print(text)

import cv2
import os
import glob

import spacy


img_dir = "C:\Backup\PycharmProjects\PycharmProjects\google-images-download\google-images-download\images\spear_phising_mail"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = {}
from PIL import Image
for f1 in files:
    try:
     print("image Name::->",f1.title())
     img = cv2.imread(f1)
     text = pytesseract.image_to_string(img)
     data[f1] = text
   #  print(text)
    except:
       print("image not clear")
nlp = spacy.load('en_core_web_sm')

for key,value in data.items():
   # print(key,"------->",value)
    about_doc = nlp(value)
  #  print(str(about_doc.sents))
    sentences = list(about_doc.sents)
#    print(sentences)
   # print(len(data))


# for i,sentence in enumerate(sentences):
#         print(i,sentence)
