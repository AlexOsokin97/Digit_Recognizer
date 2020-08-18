import os,io
from flask import Flask, request, redirect, url_for,Response,render_template,send_file,make_response,jsonify,send_from_directory
from flask_cors import CORS,cross_origin


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static\\upload')
VIDEO_FOLDER = os.path.join(APP_ROOT,'static\\video')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




cors = CORS(app, allow_headers='Content-Type', CORS_SEND_WILDCARD=True)


def allowed_file(filename):
    """
    :param filename: 
    :return: 
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET','POST'])
@cross_origin(origins='*', send_wildcard=True)
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            byte_io = io.BytesIO()
            byte_io.write(file.read())
            byte_io.seek(0)
            response = make_response(send_file(byte_io,mimetype='image/jpg'))
            response.headers['Content-Transfer-Encoding']='base64'
            return response 

    return render_template('upload.html')




if __name__ == '__main__':
    app.run(debug=True)
