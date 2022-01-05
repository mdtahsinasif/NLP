

# import spacy
import spacy

# load english language model
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

text = "This is a sample sentence."

# create spacy 
doc = nlp(text)

for token in doc:
    print(token.text,'->',token.pos_)
    
    
for token in doc:
    # check token pos
    if token.pos_=='NOUN':
        # print token
        print(token.text)    
        
text = "Attackers compromise Microsoft Exchange servers to hijack internal email chains"

# create spacy 
doc = nlp(text)

for token in doc:
    print(token.text,'->',token.pos_)        
    

for token in doc:
    # extract subject
    if (token.dep_=='nsubj'):
        print("Sub::->"+token.text)
    # extract object
    elif (token.dep_=='dobj'):
        print("Obj::->"+token.text)
    elif (token.pos_=='VERB'):
        print("Verb::->"+token.text)
    elif (token.pos_=='NOUN'):
        # print token
        print("NOUN-->"+token.text)
    elif (token.pos_=='PROPN'):
        # print token
        print("PROPN-->"+token.text)      


##############################################
        
        
article = "Attackers compromise Microsoft Exchange servers to hijack internal email chains Pakistan , India , US"


import nltk
from nltk.tag import StanfordNERTagger

print('NTLK Version: %s' % nltk.__version__)

stanford_ner_tagger = StanfordNERTagger(
    'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    'C:/Users/TahsinAsif/OneDrive - CYFIRMA INDIA PRIVATE LIMITED/antuitBackUp@3march/Asif/AI/NameEntityRecog/stanford_ner/' + 'stanford-ner-3.9.2.jar'
)

results = stanford_ner_tagger.tag(article.split())

print('Original Sentence: %s' % (article))
for result in results:
    tag_value = result[0]
    tag_type = result[1]
    if tag_type != 'O':
        print('Type: %s, Value: %s' % (tag_type, tag_value))