import flask
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/guess', methods=['POST'])
def home():
    return '1'

app.run()
