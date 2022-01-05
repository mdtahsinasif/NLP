import logging
# -*- coding: utf-8 -*-
from pymongo import MongoClient
import nltk
from nltk.tag import StanfordNERTagger
import spacy

print('NTLK Version: %s' % nltk.__version__)

class SubObjWithDb():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    def dbConnection(self):
        try:
            dbtext = ''
           # query = "{pub_date:ISODate('+last_hour_date_time+')}"

            # mydoc = collection.find({
            #     'pub_date': {'$gte': datetime.now() - timedelta(hours=24)}
            # })
            # for x in collection.find({}, {"_id": 0, "title": 1,"newsCategory":1}):
            self.logger.info("====== Inside dbConnection Method ========")
            count = 0
            mydoc = collection.find()
            for x in mydoc:
                subobjdict={}
                tagdict = {}
                subobjdict['titlePOSDetails'] = subObjWithDbObj.subObjFinder(str(x['title'].encode("utf-8")))
                #print(subobjdict)
                tagdict = subObjWithDbObj.tagFinder(str(x['description'].encode("utf-8")))
                # print(tagdict)
                subobjdictValue = subobjdict.get('titlePOSDetails')
                #print("subobjdictValue--------->", subobjdictValue)

                collection.update_one({"titlePOSDetails": {"$exists": False}},
                                       {'$set': {"titlePOSDetails": subobjdictValue}})

                tagdictValue = tagdict
                collection.update_one({"desTagInfo": {"$exists": False}},
                                            {'$set': {"desTagInfo": tagdictValue}})
                print("tagdictValue--------->", tagdictValue)

        except Exception as e:
            print(e)
            self.logger.info("====== Preprocessing the data set ========", e)

    def tagFinder(self,text):
        #print("Title inside tagFinder ------->", text)
        tagData = {}
        results = stanford_ner_tagger.tag(text.split())
        self.logger.info("====== Results ========", results)
        #print('Original Sentence: %s' % (text))
        for result in results:
            tag_value = result[0]
            tag_type = result[1]
            if tag_type != 'O':
                self.logger.info("====== Tag_type ========", tag_type, tag_value)
               # print('Type: %s, Value: %s' % (tag_type, tag_value))
                if tag_type in tagData.keys():
                    #print('True', tag_type)
                    tagData[tag_type] = tagData[tag_type] + "," + (tag_value)
                    self.logger.info("====== Tag_type ========",tagData[tag_type])
                else:
                    # create a new array in this slot
                    tagData[tag_type] = tag_value
                    self.logger.info("====== Tag_type ========", tagData[tag_type])
        return tagData

    def subObjFinder(self,text):
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
            self.logger.info("====== posData ========",posData[token.dep_])

        return posData







if __name__ == "__main__":
    logging.basicConfig(filename="SubObjWithDb.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    headers = {
        'Referer': 'https://itunes.apple.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
    }
    stanford_ner_tagger = StanfordNERTagger(
        'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
        'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'stanford-ner-3.9.2.jar'
    )

    # load english language model
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'textcat'])

    subObjWithDbObj = SubObjWithDb()
    val = 1
    host = 'localhost'
    port = 27017
    try:
        connection = MongoClient(host, port)
        #connection = MongoClient('10.97.158.161', 27017)
        #db = connection.core_stag
        db = connection.core
        collection = db.rss_feed_entry_s
        subObjWithDbObj.dbConnection()
    except Exception as e:
        print(e)
        logging.exception("Exception in main():")
    finally:
        connection.close()
        logging.info("=====Closing the db Connection")


