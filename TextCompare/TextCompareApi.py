import json
import logging
from flask import Flask, request, jsonify
import spacy
import pandas as pd
app = Flask(__name__)


@app.route('/textCompare', methods=['POST'])
def textCompare():
    try:
        # print('json---------------->>>>>>>>>>>>>>:')
        json_ = request.json
        print('json:', json_)
      #  logger.info('json_ %s' % json_)
        # print('json_g---->',json_g)
        json_obj = json.dumps(json_)
        print(json_['inputSentence'])
        ref_sentence = json_['inputSentence']
        ref_sentence = str(ref_sentence)
        print("inputSentence:", ref_sentence)
        querySentence = json_['querySentence']
        querySentence = str(querySentence)
        print("querySentence:",querySentence)
        ref_sentence = "Siemens Energy's new cybersecurity monitor and detection service - Smart Energy."
        ref_sentence_vec = nlp(ref_sentence)  # vectorize token
        #querySentence = "UNSW, Telstra collaborate on cyber security skills study courses - iTWire."
        query_sentence_vec = nlp(querySentence)
        sim = query_sentence_vec.similarity(ref_sentence_vec)
        output = sim
        print(output)
    except:
        output ='Failure'
    return jsonify(pd.Series(output).to_json(orient='values'))

if __name__ == '__main__':
    logging.basicConfig(filename="TextComparision.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    nlp = spacy.load(
        'C:\\Users\\TahsinAsif\\Anaconda3\\envs\\keras-tf\\Lib\\site-packages\\en_core_web_lg\\'
        'en_core_web_lg-2.2.5')
    # app.run(debug=True)
    #    app.run(host='0.0.0.0')
    # to run in local
    app.run(port=8086)

