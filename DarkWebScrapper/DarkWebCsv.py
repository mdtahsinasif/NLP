# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:04:12 2020

@author: tahsin.asif
"""

import pandas as pd
from sklearn import tree, model_selection, ensemble
df = pd.read_csv("C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/DarkWebScrapper/DarkWebCSVInput.csv",encoding='ISO-8859-1')
#df = pd.read_csv("/content/drive/My Drive/App/CveScore.csv",encoding='ISO-8859-1')
df.head(5)


from csv import reader
# open file in read mode
with open('C:/Users/tahsin.asif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/AI/DarkWebScrapper/DarkWebCSVInput.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print(row)

