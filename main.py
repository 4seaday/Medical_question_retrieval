import json
import logging

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran
import re

from flask import Flask, render_template, request
from flask import make_response
from retrieval import RetrievalEncoder


# from flask_bootstrap import Bootstrap
# from keras import backend as K
# from keras import layers, models

logger = logging.getLogger()
logger.setLevel(logging.INFO)
app = Flask(__name__)


# Bootstrap(app)

# @app.route('/', methods=['GET','POST'])
# def index():
#     return render_template('/index.html')

# @app.route('/index.html', methods=['GET','POST'])
# def index2():
#     return render_template('/index.html')

# @app.route('/chatboot.html', methods=['GET','POST'])
# def chatboot():
#     return render_template('/chatboot.html')

# @app.route('/chatboot2.html', methods=['GET','POST'])
# def chatboot2():
#     return render_template('/chatboot2.html')

# @app.route('/chatboot3.html', methods=['GET','POST'])
# def chatboot3():
#     return render_template('/chatboot3.html')

# retrieval cahsing
retireval_model = RetrievalEncoder()

# QA model
@app.route('/webhook', methods=['GET', 'POST'])
def get_answer():
    data = request.get_json(silent=True)
    sessionId = data['session']
    print(f"intput Text : {data['queryResult']['queryText']}")
    text = retireval_model.find_answer(data['queryResult']['queryText'])
    print(f"output Text : {text}")
    response = get_intent_and_go(text)
    response = create_response(response)
    return response


def create_response(response):
    """ Creates a JSON with provided response parameters """

    # convert dictionary with our response to a JSON string
    res = json.dumps(response, indent=4)

    logger.info(res)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'

    # print(r)
    return r


def get_intent_and_go(text):
    "intent parsing해서 해당 response 생성"
    response = {
        "fulfillmentText":
            text
    }
    return response


if __name__ == '__main__':
    app.run('0.0.0.0', port=8088, threaded=True)