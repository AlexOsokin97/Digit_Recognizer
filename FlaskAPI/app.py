import flask
import pandas as pd
import pickle
from flask_cors import CORS
from transformation import data_transformation

data = pd.read_csv('../test.csv')
features = data_transformation(data)

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/guess', methods=['POST', 'GET'])
def guess():
    loaded_model = pickle.load(open('../Models/SVM_98%_Kaggle.sav', 'rb'))
    pred = loaded_model.predict(features)

    return pred

app.run()
