from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pickle
from keras.models import load_model
import numpy as np
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
import json
import nltk
from collections import OrderedDict
from cliff.api import Cliff
import requests
from bs4 import BeautifulSoup
import re
import logging
from flask import Flask
from flask import request
#######Lib for POS#####
import nltk
from nltk.tag import StanfordNERTagger
import spacy

# Logging file setup
logging.basicConfig(filename="logfile.log", format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)


class nlpclassification(object):

    def __init__(self, inp, inputurl):

        # nltk File checker (If not found downloads)
        nltk.downloader.download('maxent_ne_chunker')
        nltk.downloader.download('words')
        nltk.downloader.download('treebank')
        nltk.downloader.download('maxent_treebank_pos_tagger')
        nltk.downloader.download('punkt')
        # since 2020-09
        nltk.downloader.download('averaged_perceptron_tagger')

        if not inp:
            logger.error("Title Not Received")

        if not inputurl:
            logger.error("URL Not Received")

        self.title = inp
        self.url = inputurl
        self.count = 0
        self.keywordsTagging = []
        self.returnjson = {}
        self.scrapper()
        self.scrapper()
        self.nlpmodel()
        self.manualclassifcationtagging()
        self.clavin()
        self.printing()

    def scrapper(self):
        # Using BS4 module as a scrapper
        self.returnjson["urlActiveAndWorking"] = "Yes"

        try:
            page = requests.get(self.url)
            if not page:
                logger.error(
                    "Page Couldn't be fetched, Kindly check if the link is valid and you have an active internet connection ")
                self.returnjson["urlActiveAndWorking"] = "No"
            soup = BeautifulSoup(page.content, 'html.parser')
            self.body_page = soup.text
            if not self.body_page:
                logger.error("Page Couldn't be Scrapped")
        except:
            print("Invalid URL, Please try again")
            logger.error("URL is invalid")

    def manualclassifcationtagging(self):
        with open('keys.json') as json_file:
            data = json.load(json_file)
            if (data is None):
                print("Keyword file not found")
                logging.error("Keyword JSON File not found")
            Cybereducation = data['Cybereducation']
            Cyberhacks = data['Cyberhacks']
            Cyberinnovation = data['Cyberinnovation']
            Cyberlaw = data['Cyberlaw']
            Cyberpolicy = data['Cyberpolicy']
            Emergingthreats = data['Emergingthreats']
            Majorrelease = data['Majorrelease']
            Threatactors = data['Threatactors']
            Vulnerabilities = data['Vulnerabilities']

        # manual tagging
        self.tagging = []

        for i in self.title.split():
            # print(i)
            try:
                if any((re.compile('.*' + i + '.*')).match(line) for line in Cybereducation):
                    self.keywordsTagging.append('Cyber Education and Awareness')
                    self.tagging.append(i)
                    logging.info("Cyber Education and Awareness was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Cyberhacks)):
                    self.keywordsTagging.append('Cyber Hacks and Incidents')
                    self.tagging.append(i)
                    logging.info("Cyber Hacks and Incidents was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Cyberinnovation)):
                    self.keywordsTagging.append('Cyber Innovation and Technology')
                    self.tagging.append(i)
                    logging.info("Cyber Innovation and Technology was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Cyberlaw)):
                    self.keywordsTagging.append('Cyber Law and Regulation')
                    self.tagging.append(i)
                    logging.info("Cyber Law and Regulation was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Cyberpolicy)):
                    self.keywordsTagging.append('Cyber Policy and Process')
                    self.tagging.append(i)
                    logging.info("Cyber Policy and Process was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Emergingthreats)):
                    self.keywordsTagging.append('Emerging Threats and Cyber Attacks')
                    self.tagging.append(i)
                    logging.info("Emerging Threats and Cyber Attacks was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Majorrelease)):
                    self.keywordsTagging.append('Major Release and Events')
                    self.tagging.append(i)
                    logging.info("Major Release and Events was found")
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Threatactors)):
                    self.keywordsTagging.append('Threat Actors and Tools')
                    logging.info("Threat Actors and Tools was found")
                    self.tagging.append(i)
                    self.count = self.count + 1
                elif (any((re.compile('.*' + i + '.*')).match(line) for line in Vulnerabilities)):
                    self.keywordsTagging.append('Vulnerabilities and Exploits')
                    self.tagging.append(i)
                    logging.info("Vulnerabilities and Exploits was found")
                    self.count = self.count + 1
            except:
                print("error in manual tagging", '\n')
                logging.error(" error in manual tagging")


        self.tagging = list(set(self.tagging))
        if not self.keywordsTagging:
            logging.info("No tagging information could be found using Keyword Matching")

        print('\n')
        print("Categorization :")
        # for i in self.keywordsTagging:
        #     self.combinedOutput.append(i)
        # print(list(OrderedDict.fromkeys(self.combinedOutput)))
        print('\n')
        self.returnjson["categorizationByKeywords"] = list(OrderedDict.fromkeys(self.keywordsTagging))
        self.returnjson["keywordsUsedForCategorisation"] = list(OrderedDict.fromkeys(self.tagging))

    def nlpmodel(self):
        chunked = ne_chunk(pos_tag(word_tokenize(self.title)))
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        self.tagging = continuous_chunk

        with open('tokenizer_new.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            if (tokenizer is None):
                print("Tokenizer file not found")
                logging.error("Tokenizer file not found")
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load('classes_new.npy', allow_pickle=True)
        model = load_model('multi.h5')
        self.title2 = [self.title]
        if (model is None):
            print("Model not found")
            logging.error("Model not found")
        seq = tokenizer.texts_to_sequences(self.title2)
        seq_padded = pad_sequences(seq, 50)
        # print(seq_padded)
        out = model.predict(seq_padded)

        a = out[0]
        di = {}
        c = 0
        for i in a:
            if (i > 0.5):
                di[''.join(encoder.inverse_transform([c]))] = str(i)
            c = c + 1
        di = sorted(di.items(), key=lambda kv: kv[1])
        modelOutput = di

        modelOutput = modelOutput[-4:]
        modelOutput = modelOutput[::-1]
        # self.combinedOutput = modelOutput
        print(modelOutput)
        self.returnjson["categorizationByModel"] = dict(modelOutput)
        print(self.returnjson["categorizationByModel"])

    def clavin(self):
        my_cliff = Cliff('http://localhost:8080')
        dictionary = {}
        while True:
            try:
                dictionary = my_cliff.parse_text(self.body_page)
                break

            except:
                print("Clavin Docker not running or link not valid", '\n')
                logging.error("Clavin Docker not running or link not valid")
                break

        json_object = json.dumps(dictionary, indent=4)

        with open("clavin.json", "w") as outfile:
            outfile.write(json_object)
            logging.info("Clavin JSON file written")

        with open('clavin.json') as fi:
            # with open('sample.json') as fi:
            self.d = json.load(fi)
            if not self.d:
                logging.error("Clavin JSON File Empty")

    def printing(self):
        while True:
            try:
                d1 = self.d['results']
                d2 = d1['places']
                #    d3 = d2['mentions']
                print("Geo Tagging :")
                d3 = d2['focus']
                d4 = d3['countries']
                d5 = d3['states']
                d6 = d3['cities']
                country = d4[0]['name']
                state = d5[0]['name']
                city = d6[0]['name']
                # print('\n')
                print("Country: ", country)
                print("State: ", state)
                print("City: ", city)
                self.returnjson["Geo Tagging"] = {"Country": country, "State": state, "City": city}
                print('\n')
                break
            except:
                print("No location data found in the link", '\n')
                logging.warning("No location data found in the link")
                break

        print("Tagging :")
        print(self.tagging)
        self.returnjson["taggingBasedOnTitle"] = self.tagging
        print('\n')
        while True:
            try:
                organisations = d1['organizations']
                print("Organisations Mentioned: ", end=" ")
                self.returnjson["Organisations Mentioned"] = d1['organizations']
                for i in organisations:
                    print(i['name'], end=", ")
                break

            except:

                print("No organisation data found in the link", '\n')
                logging.warning("No organisation data found in the link")
                break

        while True:
            try:
                people = d1['people']
                print("People Mentioned: ", end=" ")
                self.returnjson["People Mentioned"] = d1['people']
                for i in people:
                    print(i['name'], end=", ")
                break

            except:

                print("No People mentioned in the link", '\n')
                logging.warning("No People mentioned in the link")
                break

        print('\n')

    def subObjFinder(text):
        print("Title inside subObjFinder function ------->", text)
        posData = {}
        doc = nlp(text)
        for token in doc:
            # check token pos
            # if token.pos_ == 'NOUN':
            #     # print token
            #     print(token.text, '->', token.pos_)
            #    # posData['NOUN'] = token.text
            # if token.pos_ == 'VERB':
            #     # print token
            #     print(token.text, '->', token.pos_)
            #    # posData['VERB'] = token.text
            # if (token.dep_ == 'nsubj'):
            #     print(token.dep_, '--->', token.text)
            #    # posData['SUBJECT'] = token.text
            # # extract object
            # if (token.dep_ == 'dobj'):
            #     print(token.dep_, '--->', token.text)
            # posData['OBJECT'] = token.text
            posData[token.dep_] = token.text
            #logger.info("====== posData ========", posData[token.dep_])

        return posData

    def tagFinder(desc):
        print("Title inside tagFinder ------->", desc)
        tagData = {}
        results = stanford_ner_tagger.tag(str(desc).split())
        logging.info("====== Results ========", results)
        #print('Original Sentence: %s' % (text))
        for result in results:
            tag_value = result[0]
            tag_type = result[1]
            if tag_type != 'O':
                logging.info("====== Tag_type ========", tag_type, tag_value)
               # print('Type: %s, Value: %s' % (tag_type, tag_value))
                if tag_type in tagData.keys():
                    #print('True', tag_type)
                    tagData[tag_type] = tagData[tag_type] + "," + (tag_value)
                    logging.info("====== Tag_type ========",tagData[tag_type])
                else:
                    # create a new array in this slot
                    tagData[tag_type] = tag_value
                    logging.info("====== Tag_type ========", tagData[tag_type])
        return tagData

# Taking user input manually (Debug Only)

# while True:
#     try:
#         inp = input("enter the title: ")
#         if not inp:
#             logging.error("Empty Title String Entered")
#             raise ValueError('Empty Title String')
#
#         break
#     except ValueError as e:
#         print(e)
#
#
# while True:
#     try:
#         inputurl = input('enter url :')
#         if not inputurl:
#             logging.error("Empty URL Entered")
#             raise ValueError('Empty URL')
#         break
#     except ValueError as e:
#         print(e)
#
# inp = "China attacks India APT21"
# inputurl = "https://en.wikipedia.org/wiki/Karsten_Nohl"
#
# printobject = nlpclassification(inp, inputurl)
# d = printobject.returnjson
# d= json.dumps(d)
# print(d)
#
#


# FLASK API

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def summary():
    data = request.json
    # return data
    #print("data:::-",data)
    #print(type(data))
    if not data["Title"]:
        logging.critical("Title not available in the JSON or wrong format")
    if not data["URL"]:
        logging.critical("URL not available in the JSON or wrong format")
    if not data["Description"]:
        logging.critical("Description not available in the JSON or wrong format")

    inp = data["Title"]
    print("Title::::-", inp)
    url = data["URL"]
    description = data["Description"]
    printobject = nlpclassification(inp, url)
    d = printobject.returnjson
    # print("---------------d------------",d)
    d = json.dumps(d)
    categoryjson = json.loads(d)
    posData = nlpclassification.subObjFinder(inp)
    posDatajson = json.dumps(posData)
    #print("posDatajson------------",posDatajson)
    posjson = json.loads(posDatajson)

    # tagging fetching code
    desTagData = nlpclassification.tagFinder(description)

    descriptionJson = json.dumps(desTagData)
    #print("descriptionJson------------", descriptionJson)
    desJson = json.loads(descriptionJson)

    jsondata = dict(list(categoryjson.items()) + list(posjson.items()) + list(desJson.items()))

    finaljson =json.dumps(jsondata)
    #print("finaljson::::::::::::::->", finaljson)
    return finaljson




if __name__ == "__main__":
    # load english language model
    nlp = spacy.load("C:/Users/TahsinAsif/AppData/Roaming/Python/Python39/site-packages/en_core_web_sm/en_core_web_sm-2.3.1", disable=['ner', 'textcat'])

    stanford_ner_tagger = StanfordNERTagger(
        'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
        'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'stanford-ner-3.9.2.jar'
    )
    app.run(host='127.0.0.1', port=5000, debug=True)