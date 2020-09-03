import flask
import pandas as pd
import pickle
import base64
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from transformation import data_transformation
from transform_to_image import getImage

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'number'
app.config['CORS_HEADERS'] = 'Guess-Type'
app.config["DEBUG"] = True

cors = CORS(app, resources={r"/guess": {"origins": "http://localhost:5000"}})

@app.route('/guess', methods=['POST'])
@cross_origin(origin='localhost',headers=['Guess-Type','number'])
def guess():
    data = request.data
    image = getImage(data)
    print(image.shape)
    #features = data_transformation(data)
    
    #loaded_model = pickle.load(open('../Models/SVM_98%_Kaggle.sav', 'rb'))
    #pred = loaded_model.predict(features)

    #return pred