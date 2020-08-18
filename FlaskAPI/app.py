import flask
from flask import Flask, jsonify, jsonify, request
import json
import numpy as np
import pickle
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys

app = Flask(__name__)



class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)
    
    
def load_models():
    file_name = 'model_path'
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

@app.route('/guess', methods=['POST'])
def guess():
    #stub input features
    imagefile = flask.request.files.get('imagefile', '')
    #x = request_json['input']
    #x_in = np.array(x).reshape(1, -1)
    #load model & make a guess
    model = load_models()
    #guessing = model.predict(x_in)[0]
    #send a response
    #response = json.dumps({'response': guessing})
    #return response, 200

if __name__ == '__main__':
    test(CORSRequestHandler, HTTPServer, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
    app.run(debug=True)
